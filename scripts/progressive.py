"""Progressive (bit-plane) kvtc coding with a hot tier and question-time refinement.

Each compressed token's kvtc coefficients (cross-layer PCA, RoPE removed) are
quantized once: per (token, group of 64 components) an FP16 offset and range,
and 8-bit codes on a 256-bin grid. The top m bits of a code are an m-bit code
on a coarser grid (decoded at its bin centre), so precision can be cut by
dropping bit-planes without re-encoding.

The stream is a fixed order of (group, bit-plane) pieces, built once on
calibration data by error removed per byte (``fit_progressive.py``). A
token stored with its first k pieces decodes at that precision; more pieces
only refine it. Pieces cost 8 bytes (64 one-bit values) plus 4 bytes for the
group's offset and range with its first plane.

- Hot tier: each token keeps a prefix of the stream; DimmerKV picks the
  prefix length per token under a byte budget (flat, or weighted by scores).
- Cold tier: the remaining pieces of every token (up to all 8 planes).
- Question time: score the hot cache with the question, fetch the cold
  pieces for the top tokens only, and rebuild those tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

import dimmer
from kvtc import KVTCCodec, allocate_rungs, rerotate, rope_tables

PLANES = 8
HEADER_BYTES = 4  # FP16 offset + FP16 range, sent with a group's first plane.
RUNG_ID_BYTES = 1


def quantize8(coefficients: torch.Tensor, group: int):
    """[tokens, features] -> codes uint8 [tokens, groups, group], offset and range FP16 [tokens, groups]."""
    tokens, features = coefficients.shape
    grouped = coefficients.view(tokens, features // group, group)
    low = grouped.amin(dim=-1).half()
    span = (grouped.amax(dim=-1) - low.float()).clamp_min(1e-8).half()
    codes = ((grouped - low.float()[..., None]) / span.float()[..., None] * 256).floor().clamp(0, 255)
    return codes.to(torch.uint8), low, span


def dequantize(codes: torch.Tensor, low: torch.Tensor, span: torch.Tensor, planes: torch.Tensor) -> torch.Tensor:
    """Decode with ``planes`` [tokens or 1, groups] top bit-planes per group (0 = dropped -> 0)."""
    m = planes.to(torch.int32)[..., None]
    shift = (PLANES - m).clamp(0, PLANES)
    index = codes.to(torch.int32) >> shift
    bins = (2**m).float()
    value = low.float()[..., None] + (index.float() + 0.5) * span.float()[..., None] / bins
    out = torch.where(m > 0, value, torch.zeros_like(value))
    return out.reshape(codes.shape[0], -1)


@dataclass
class Order:
    """Progressive order for one stream."""

    planes: torch.Tensor  # [pieces + 1, groups]: planes of each group after the first k pieces.
    prefix_bytes: torch.Tensor  # [pieces + 1]: bytes of the first k pieces.
    rungs: list[int]  # Prefix lengths k offered to the allocator, coarse -> full.


def build_order(coefficients: torch.Tensor, group: int, targets, exact_bytes: int) -> Order:
    """Greedy (group, plane) order by calibration error removed per byte."""
    codes, low, span = quantize8(coefficients, group)
    groups = codes.shape[1]
    error = torch.zeros(groups, PLANES + 1, dtype=torch.float64)
    grouped = coefficients.view(coefficients.shape[0], groups, group)
    for m in range(PLANES + 1):
        rebuilt = dequantize(codes, low, span, torch.full((1, groups), m, device=codes.device))
        error[:, m] = (grouped - rebuilt.view_as(grouped)).square().sum(dim=(0, 2)).double().cpu()
    level = torch.zeros(groups, dtype=torch.long)
    planes, sizes = [level.clone()], [0.0]
    rows = torch.arange(groups)
    for _ in range(groups * PLANES):
        # Score each group by its best run of upcoming planes (gain per byte), so a
        # first plane that only pays off together with later ones is not postponed.
        score = torch.full((groups,), float("-inf"), dtype=torch.float64)
        for ahead in range(1, PLANES + 1):
            target = level + ahead
            valid = target <= PLANES
            gain = error[rows, level] - error[rows, target.clamp_max(PLANES)]
            cost = 8.0 * ahead + torch.where(level == 0, HEADER_BYTES, 0).double()
            score = torch.where(valid, torch.maximum(score, gain / cost), score)
        best = int(score.argmax())
        sizes.append(sizes[-1] + 8.0 + (HEADER_BYTES if level[best] == 0 else 0))
        level[best] += 1
        planes.append(level.clone())
    prefix_bytes = torch.tensor(sizes)
    rungs = []
    for target in targets:
        fitting = (prefix_bytes <= target * exact_bytes).nonzero().flatten()
        if len(fitting) and int(fitting[-1]) > 0 and (not rungs or int(fitting[-1]) > rungs[-1]):
            rungs.append(int(fitting[-1]))
    rungs.append(len(sizes) - 1)  # Full stream: all 8 planes.
    return Order(torch.stack(planes).to(torch.uint8), prefix_bytes, rungs)


@dataclass
class TokenStore:
    """Encoded middle tokens of one stream."""

    codes: torch.Tensor
    low: torch.Tensor
    span: torch.Tensor
    hot_pieces: torch.Tensor  # [tokens] prefix length kept in the hot tier.


class ProgressiveCodec:
    """Hot/cold progressive kvtc codec on top of a ``fit_kvtc.py`` basis."""

    def __init__(self, kvtc_path: str | Path, order_path: str | Path, device, model) -> None:
        self.base = KVTCCodec(kvtc_path, device, model)
        saved = torch.load(order_path, map_location="cpu", weights_only=False)
        if saved["model"] != self.base.model_id:
            raise RuntimeError(f"{order_path} was fit for {saved['model']}.")
        self.orders = {
            kind: Order(saved[kind]["planes"].to(device), saved[kind]["prefix_bytes"].to(device), saved[kind]["rungs"])
            for kind in ("key", "value")
        }
        self.device = torch.device(device)
        self._model = model

    @property
    def model(self) -> str:
        return self.base.model_id

    def full_bytes(self, kind: str) -> int:
        return int(self.orders[kind].prefix_bytes[-1])

    def encode(self, keys, values, budget: float, exact_recent: int, weights=None) -> "ProgressiveCache":
        base, tokens = self.base, keys[0].shape[2]
        start = min(base.sinks, tokens)
        end = max(start, tokens - exact_recent)
        middle = end - start
        cos, sin = rope_tables(self._model, tokens, self.device)
        budget_bytes = int(budget * tokens * 2 * base.exact_bytes)
        remaining = budget_bytes - (tokens - middle) * 2 * base.exact_bytes

        stores, errors, level_bytes = {}, [], []
        for kind, tensors in (("key", keys), ("value", values)):
            stream, order = base.streams[kind], self.orders[kind]
            matrix = base.to_matrix(tensors, kind, cos, sin)
            coefficients = (matrix[start:end] - stream.mean) @ stream.basis
            codes, low, span = quantize8(coefficients, base.group)
            energy = coefficients.square().sum(dim=1).mean().clamp_min(1e-12)
            # Error of every (token, group) at 0..8 planes, then each rung's error by lookup.
            grouped = coefficients.view(middle, -1, base.group)
            groups = grouped.shape[1]
            table = torch.stack([
                (grouped - dequantize(codes, low, span, torch.full((1, groups), m, device=self.device))
                 .view_as(grouped)).square().sum(dim=-1)
                for m in range(PLANES + 1)
            ], dim=-1)  # [tokens, groups, 9]
            rung_planes = order.planes[order.rungs].long().T  # [groups, rungs]
            rung_error = table.gather(2, rung_planes[None].expand(middle, -1, -1)).sum(dim=1) / energy
            del table, grouped
            if weights is not None:
                importance = weights[start:end].float()
                rung_error = rung_error * (importance / importance.mean().clamp_min(1e-12))[:, None]
            errors.append(rung_error)
            level_bytes += [int(order.prefix_bytes[k]) + RUNG_ID_BYTES for k in order.rungs]
            stores[kind] = TokenStore(codes, low, span, torch.zeros(middle, dtype=torch.long, device=self.device))
        chosen = allocate_rungs(torch.cat(errors), torch.tensor(level_bytes, device=self.device), middle, remaining)
        for index, kind in enumerate(("key", "value")):
            rung_index = chosen[index * middle : (index + 1) * middle]
            stores[kind].hot_pieces = torch.tensor(self.orders[kind].rungs, device=self.device)[rung_index]
        return ProgressiveCache(self, keys, values, stores, start, end, cos, sin, budget_bytes)


class ProgressiveCache:
    """A compressed conversation: hot prefix per token, cold remainder."""

    def __init__(self, codec, keys, values, stores, start, end, cos, sin, budget_bytes):
        self.codec, self.stores = codec, stores
        self.start, self.end, self.cos, self.sin = start, end, cos, sin
        self.originals = {"key": keys, "value": values}
        self.budget_bytes = budget_bytes
        base = codec.base
        exact_tokens = keys[0].shape[2] - (end - start)
        self.hot_bytes = exact_tokens * 2 * base.exact_bytes + sum(
            int(codec.orders[kind].prefix_bytes[store.hot_pieces].sum()) + RUNG_ID_BYTES * len(store.hot_pieces)
            for kind, store in stores.items()
        )
        if self.hot_bytes > budget_bytes:
            raise AssertionError(f"Hot tier {self.hot_bytes} B over budget {budget_bytes} B.")
        self.cold_bytes = sum(
            int((codec.orders[kind].prefix_bytes[-1] - codec.orders[kind].prefix_bytes[store.hot_pieces]).sum())
            for kind, store in stores.items()
        )

    def _rows(self, kind: str, rows: torch.Tensor, pieces: torch.Tensor) -> torch.Tensor:
        """Decode middle-token ``rows`` at per-token prefix lengths -> pre-RoPE matrix rows."""
        store, base = self.stores[kind], self.codec.base
        stream = base.streams[kind]
        planes = self.codec.orders[kind].planes[pieces]
        coefficients = dequantize(store.codes[rows], store.low[rows], store.span[rows], planes)
        return coefficients @ stream.basis.T + stream.mean

    def _write(self, tensors: list[torch.Tensor], kind: str, rows: torch.Tensor, matrix_rows: torch.Tensor) -> None:
        base = self.codec.base
        positions = rows + self.start
        blocks = matrix_rows.view(-1, base.layers, base.heads, base.head_dim)
        for layer, tensor in enumerate(tensors):
            part = blocks[:, layer].transpose(0, 1)  # [heads, n, dim]
            if kind == "key":
                part = rerotate(part, self.cos[positions], self.sin[positions])
            tensor[0, :, positions] = part.to(tensor.dtype)

    def hot(self):
        """Cache tensors as restored from the hot tier only."""
        out = {}
        for kind, originals in self.originals.items():
            tensors = [t.clone() for t in originals]  # Sinks and recent window stay exact.
            rows = torch.arange(self.end - self.start, device=self.codec.device)
            self._write(tensors, kind, rows, self._rows(kind, rows, self.stores[kind].hot_pieces))
            out[kind] = tensors
        return out["key"], out["value"]

    def refine(self, keys, values, tokens: torch.Tensor):
        """Fetch the cold pieces of middle ``tokens`` (absolute positions); returns new tensors and bytes fetched."""
        rows = tokens - self.start
        rows = rows[(rows >= 0) & (rows < self.end - self.start)]
        out, fetched = {}, 0
        for kind, tensors in (("key", keys), ("value", values)):
            order, store = self.codec.orders[kind], self.stores[kind]
            full = torch.full_like(rows, order.planes.shape[0] - 1)
            fetched += int((order.prefix_bytes[-1] - order.prefix_bytes[store.hot_pieces[rows]]).sum())
            updated = [t.clone() for t in tensors]
            self._write(updated, kind, rows, self._rows(kind, rows, full))
            out[kind] = updated
        return out["key"], out["value"], fetched


def question_importance(model, keys, values, tail_ids: torch.Tensor, context_length: int, evicted=None):
    """Attention each context token receives from the question tokens, summed over layers and heads."""
    from common import build_cache

    cache = build_cache([k.clone() for k in keys], [v.clone() for v in values], model)
    ids = tail_ids[None].to(keys[0].device)
    positions = torch.arange(context_length, context_length + ids.shape[1], device=ids.device)[None]
    dimmer.set_eviction(evicted)
    dimmer.STATE.scores, dimmer.STATE.window_scores = {}, {}
    dimmer.STATE.scoring = True
    try:
        model(input_ids=ids, past_key_values=cache, position_ids=positions)
    finally:
        dimmer.STATE.scoring = False
        dimmer.set_eviction(None)
    return torch.stack([s[:, :context_length] for s in dimmer.STATE.scores.values()]).sum(dim=(0, 1))
