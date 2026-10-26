"""DimmerKV codec: kvtc transform coding with per-token precision, to and from bytes.

Engine-agnostic port of ``scripts/kvtc.py`` (``KVTCCodec.compress``) and
``scripts/kvtc_storage.py`` (``encode_bytes`` / ``decode_bytes``). Caches go
in and out as stacked tensors ``[layers, kv_heads, tokens, head_dim]``, so any
engine that can gather a sequence's keys and values can use it.

Per token, keys (RoPE removed) of every layer and KV head form one vector,
likewise values; a PCA basis fitted once per model decorrelates it, and the
coefficients are quantized in groups with a per-group bit width taken from a
ladder of rungs. The DimmerKV allocator picks each token's rung under one
byte budget (``per_token=True``), after first giving every token the rung
uniform kvtc could afford with ``floor_share`` of the budget. The first
``sinks`` tokens and the last ``recent`` tokens stay exact.

Differences from the scripts, none of which change the bytes:
- RoPE comes from the calibration file (``rope.py``), not the HF model.
- The basis keeps only the columns some rung uses (``used_features``).
- Rung errors are computed in token chunks to bound temporary memory.
- ``position_offset`` codes a segment that starts mid-conversation (no sinks).
- Budgets too small for the coarsest rung fall back to the coarsest rung.

The blob is a small header (magic, version, codec fingerprint, position
offset) followed by the zlib-DEFLATE of exactly the scripts' byte layout.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .rope import RoPE

BIT_OPTIONS = (0, 2, 4, 8)
GROUP_BYTES_OVERHEAD = 4  # FP16 shift + FP16 scale per (token, group) with bits > 0.
MAGIC = b"DKV1"
FORMAT_VERSION = 1
BLOB_HEADER = struct.Struct("<4sB16sq")  # magic, version, fingerprint, position offset.
PAYLOAD_HEADER = np.dtype(np.int32)  # tokens, start, end (as in kvtc_storage.py).


def quantize_groups(coefficients: torch.Tensor, bits: torch.Tensor, group: int) -> torch.Tensor:
    """Quantize-dequantize [tokens, features] with per-group bit widths (b = 0 drops the group)."""
    tokens, features = coefficients.shape
    grouped = coefficients.view(tokens, features // group, group)
    out = torch.zeros_like(grouped)
    for width in BIT_OPTIONS[1:]:
        chosen = (bits == width).nonzero().flatten()
        if chosen.numel() == 0:
            continue
        part = grouped[:, chosen]
        low = part.amin(dim=-1, keepdim=True).half().float()
        scale = ((part.amax(dim=-1, keepdim=True) - low) / (2**width - 1)).clamp_min(1e-8).half().float()
        out[:, chosen] = ((part - low) / scale).round().clamp(0, 2**width - 1) * scale + low
    return out.view(tokens, features)


def rung_bytes(bits: torch.Tensor, group: int) -> int:
    """Stored bytes per token vector for one bit allocation [groups]."""
    used = bits > 0
    return int((bits.float() * group / 8).sum() + GROUP_BYTES_OVERHEAD * used.sum())


def allocate_rungs(errors: torch.Tensor, level_bytes: torch.Tensor, middle: int, budget: int) -> torch.Tensor:
    """Greedy per-slot rung choice (DimmerKV flat) under a byte budget.

    ``errors`` is [slots, levels] (first ``middle`` slots are keys, the rest
    values) and ``level_bytes`` holds each stream's level sizes back to back.
    Every slot starts at its coarsest level; an upgrade's value is error drop
    per extra byte, made non-increasing along levels with ``cummin`` so the
    stable greedy pick always takes each slot's upgrades in order.
    """
    slots, levels = errors.shape
    sizes = level_bytes.view(2, levels).repeat_interleave(middle, dim=0).double()
    base = int(sizes[:, 0].sum())
    if base > budget:
        raise ValueError("Budget cannot cover the coarsest rung for every token.")
    if levels == 1:
        return torch.zeros(slots, dtype=torch.long, device=errors.device)
    gains = (errors[:, :-1] - errors[:, 1:]).clamp_min(0).double()
    costs = sizes[:, 1:] - sizes[:, :-1]
    value = (gains / costs).cummin(dim=1).values
    order = torch.sort(value.reshape(-1), descending=True, stable=True).indices
    taken = order[costs.reshape(-1)[order].cumsum(0) <= budget - base]
    counts = torch.zeros(slots, dtype=torch.long, device=errors.device)
    counts.index_add_(0, taken // (levels - 1), torch.ones_like(taken))
    return counts


def _pack(codes: torch.Tensor, width: int) -> torch.Tensor:
    """Pack uint8 values below 2**width into bytes, 8 // width per byte."""
    per = 8 // width
    flat = codes.flatten().to(torch.int32)
    flat = torch.cat([flat, flat.new_zeros((-flat.numel()) % per)]).view(-1, per)
    shifts = torch.arange(per, device=codes.device, dtype=torch.int32) * width
    return (flat << shifts).sum(dim=1).to(torch.uint8)


def _unpack(packed: torch.Tensor, width: int, count: int) -> torch.Tensor:
    per = 8 // width
    shifts = torch.arange(per, device=packed.device, dtype=torch.int32) * width
    values = (packed.to(torch.int32)[:, None] >> shifts) & (2**width - 1)
    return values.flatten()[:count].to(torch.uint8)


def _packed_length(count: int, width: int) -> int:
    per = 8 // width
    return (count + per - 1) // per


def _numpy_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().cpu().numpy().tobytes()


def checkpoint_fingerprint(saved: dict) -> bytes:
    """16-byte identity of a calibration: model, geometry, rungs, means and RoPE."""
    digest = hashlib.sha256()
    digest.update(str(saved["model"]).encode())
    digest.update(np.array([saved["layers"], saved["heads"], saved["head_dim"], saved["group"], saved["sinks"]],
                           dtype=np.int64).tobytes())
    for kind in ("key", "value"):
        digest.update(_numpy_bytes(saved[kind]["rungs"].to(torch.int64)))
        digest.update(_numpy_bytes(saved[kind]["mean"].float()))
        digest.update(_numpy_bytes(saved[kind]["basis"][:, :1].float()))
    if "rope" in saved:
        digest.update(_numpy_bytes(saved["rope"]["inv_freq"].float()))
    return digest.digest()[:16]


def read_fingerprint(path: str | Path) -> bytes:
    """Fingerprint of a calibration file without loading its basis into memory."""
    saved = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return saved.get("fingerprint") or checkpoint_fingerprint(saved)


@dataclass
class Stream:
    mean: torch.Tensor  # [features]
    basis: torch.Tensor  # [features, used_features], columns ordered by variance
    rungs: torch.Tensor  # [rungs, used_features // group] bit widths, coarse -> fine
    rung_bytes: list[int]


class DimmerKVCodec:
    """Compresses one sequence's KV cache to bytes and back.

    ``checkpoint`` is a calibration file from ``dimmerkv calibrate`` (or a
    legacy ``scripts/fit_kvtc.py`` file plus ``rope``). ``max_rung_fraction``
    drops rungs finer than that fraction of FP16, which shrinks the basis
    kept in memory (tokens can still be stored exact).
    """

    def __init__(self, checkpoint: str | Path | dict, device="cpu", rope: RoPE | None = None,
                 max_rung_fraction: float | None = None, error_chunk: int = 2048) -> None:
        saved = checkpoint if isinstance(checkpoint, dict) else torch.load(
            checkpoint, map_location="cpu", weights_only=False, mmap=True)
        self.model_id = saved["model"]
        self.layers, self.heads, self.head_dim = saved["layers"], saved["heads"], saved["head_dim"]
        self.group, self.sinks = saved["group"], saved["sinks"]
        self.features = self.layers * self.heads * self.head_dim
        self.exact_bytes = self.features * 2  # One FP16 stream (K or V) of one token.
        self.device = torch.device(device)
        self.error_chunk = error_chunk
        self.fingerprint = saved.get("fingerprint") or checkpoint_fingerprint(saved)
        if rope is None:
            if "rope" not in saved:
                raise ValueError("Calibration file has no RoPE tables; run `dimmerkv convert` on it, "
                                 "or pass rope=RoPE.from_hf_config(config).")
            rope = RoPE(saved["rope"]["inv_freq"], saved["rope"]["attention_scaling"])
        if rope.rotary_dim > self.head_dim:
            raise ValueError(f"RoPE rotary_dim {rope.rotary_dim} exceeds head_dim {self.head_dim}.")
        self.rope = rope.to(self.device)
        self.streams: dict[str, Stream] = {}
        for kind in ("key", "value"):
            rungs, sizes = saved[kind]["rungs"].long(), list(saved[kind]["rung_bytes"])
            if max_rung_fraction is not None:
                keep = [i for i, size in enumerate(sizes) if size <= max_rung_fraction * self.exact_bytes] or [0]
                rungs, sizes = rungs[keep], [sizes[i] for i in keep]
            used_groups = int((rungs > 0).any(dim=0).nonzero().max()) + 1
            used = used_groups * self.group
            self.streams[kind] = Stream(
                saved[kind]["mean"].float().to(self.device),
                saved[kind]["basis"][:, :used].float().to(self.device),
                rungs[:, :used_groups].to(self.device),
                sizes,
            )

    # ----- layout -----
    def _matrix(self, tensors: torch.Tensor, cos=None, sin=None) -> torch.Tensor:
        """[layers, heads, tokens, dim] -> [tokens, features] float32, keys derotated when cos is given."""
        tensors = tensors.to(self.device)
        if cos is not None:
            tensors = self.rope.derotate(tensors, cos, sin)
        tokens = tensors.shape[2]
        return tensors.float().permute(2, 0, 1, 3).reshape(tokens, self.features)

    def _unmatrix(self, matrix: torch.Tensor) -> torch.Tensor:
        return matrix.view(matrix.shape[0], self.layers, self.heads, self.head_dim).permute(1, 2, 0, 3)

    def _check(self, tensors: torch.Tensor) -> None:
        expected = (self.layers, self.heads, self.head_dim)
        if tensors.dim() != 4 or (tensors.shape[0], tensors.shape[1], tensors.shape[3]) != expected:
            raise ValueError(f"Expected [layers={self.layers}, heads={self.heads}, tokens, dim={self.head_dim}], "
                             f"got {tuple(tensors.shape)}.")

    def _inverse(self, coefficients: torch.Tensor, stream: Stream) -> torch.Tensor:
        """``coefficients @ basis.T + mean``, multiplying only groups non-zero for some row."""
        rows = coefficients.shape[0]
        used = (coefficients.view(rows, -1, self.group) != 0).any(dim=2).any(dim=0)
        if bool(used.all()):
            return coefficients @ stream.basis.T + stream.mean
        columns = used.repeat_interleave(self.group).nonzero().flatten()
        return coefficients[:, columns] @ stream.basis[:, columns].T + stream.mean

    # ----- allocation -----
    def _rung_errors(self, coefficients: torch.Tensor, tail: torch.Tensor | None, stream: Stream) -> torch.Tensor:
        """Squared error [tokens, rungs] of every token at every rung."""
        out = torch.empty(coefficients.shape[0], len(stream.rung_bytes), device=self.device)
        for begin in range(0, coefficients.shape[0], self.error_chunk):
            part = coefficients[begin : begin + self.error_chunk]
            out[begin : begin + self.error_chunk] = torch.stack(
                [(part - quantize_groups(part, bits, self.group)).square().sum(dim=1) for bits in stream.rungs],
                dim=1)
        # Components outside the kept basis are dropped at every rung.
        return out if tail is None else out + tail[:, None]

    def _choose(self, centered, coefficients, middle, remaining, per_token, floor_share, key_weight, weights):
        level_bytes = {kind: self.streams[kind].rung_bytes + [self.exact_bytes] for kind in ("key", "value")}
        levels = len(level_bytes["key"])
        cost = [level_bytes["key"][r] + level_bytes["value"][r] for r in range(levels)]
        if not per_token:
            fits = [r for r in range(levels) if middle * cost[r] <= remaining]
            rung = max(fits) if fits else 0
            return {kind: torch.full((middle,), rung, device=self.device) for kind in ("key", "value")}
        errors = []
        for kind in ("key", "value"):
            stream = self.streams[kind]
            energy = centered[kind].square().sum(dim=1)
            tail = None
            if stream.basis.shape[1] < self.features:
                tail = (energy - coefficients[kind].square().sum(dim=1)).clamp_min(0)
            # Relative error so key and value errors are on one scale; exact level = 0.
            rung_err = self._rung_errors(coefficients[kind], tail, stream) / energy.mean()
            rung_err = torch.cat([rung_err, torch.zeros_like(rung_err[:, :1])], dim=1)
            if weights is not None:
                importance = weights.float().to(self.device)
                rung_err = rung_err * (importance / importance.mean().clamp_min(1e-12))[:, None]
            if kind == "key":
                rung_err = rung_err * key_weight
            errors.append(rung_err)
        first = 0
        if floor_share > 0:
            fits = [r for r in range(levels) if middle * cost[r] <= floor_share * remaining]
            first = max(fits) if fits else 0
        remaining = max(remaining, middle * cost[first])
        # The greedy pick sorts in float64, which MPS lacks; the [tokens, levels] table is small.
        where = "cpu" if self.device.type == "mps" else self.device
        sizes = torch.cat([torch.tensor(level_bytes[kind][first:], device=where) for kind in ("key", "value")])
        errors = torch.cat([e[:, first:] for e in errors]).to(where)
        chosen = (allocate_rungs(errors, sizes, middle, remaining) + first).to(self.device)
        return {"key": chosen[:middle], "value": chosen[middle:]}

    # ----- public API -----
    def compress(self, keys: torch.Tensor, values: torch.Tensor, *, budget: float = 0.03, recent: int = 32,
                 position_offset: int = 0, sinks: int | None = None, per_token: bool = True,
                 floor_share: float = 1.0, key_weight: float = 1.0, weights: torch.Tensor | None = None,
                 zlib_level: int = 6) -> bytes:
        """Compress keys and values ``[layers, kv_heads, tokens, head_dim]`` (keys with RoPE applied).

        ``budget`` is the stored size as a fraction of the FP16 K+V cache of
        these tokens (exact tokens included). ``position_offset`` is the
        absolute position of the first token; segments that do not start at 0
        keep no sinks unless ``sinks`` says so. ``weights`` [middle tokens]
        scales each token's errors by its importance (e.g. attention scores).
        The defaults are the repository's recommended ``dkvtc@0.03:floor1``
        without question scores.
        """
        self._check(keys)
        self._check(values)
        tokens = keys.shape[2]
        if sinks is None:
            sinks = self.sinks if position_offset == 0 else 0
        start = min(sinks, tokens)
        end = max(start, tokens - recent)
        middle = end - start
        cos, sin = self.rope.tables(position_offset, tokens)
        matrices = {"key": self._matrix(keys, cos, sin), "value": self._matrix(values)}
        centered, coefficients = {}, {}
        for kind in ("key", "value"):
            stream = self.streams[kind]
            centered[kind] = matrices[kind][start:end] - stream.mean
            coefficients[kind] = centered[kind] @ stream.basis
        budget_bytes = int(budget * tokens * 2 * self.exact_bytes)
        remaining = budget_bytes - (tokens - middle) * 2 * self.exact_bytes
        if middle:
            choice = self._choose(centered, coefficients, middle, remaining, per_token, floor_share, key_weight,
                                  None if weights is None else weights[:middle])
        else:
            choice = {kind: torch.zeros(0, dtype=torch.long, device=self.device) for kind in ("key", "value")}
        del centered

        parts = [np.array([tokens, start, end], dtype=PAYLOAD_HEADER).tobytes()]
        for kind, tensors in (("key", keys), ("value", values)):
            stream = self.streams[kind]
            rungs = choice[kind]
            parts.append(_numpy_bytes(rungs.to(torch.uint8)))
            exact_rung = len(stream.rung_bytes)
            for rung in sorted(rungs.unique().tolist()):
                rows = rungs == rung
                if rung == exact_rung:
                    parts.append(_numpy_bytes(matrices[kind][start:end][rows].half()))
                    continue
                grouped = coefficients[kind][rows].view(int(rows.sum()), -1, self.group)
                bits = stream.rungs[rung]
                for width in BIT_OPTIONS[1:]:
                    chosen = (bits == width).nonzero().flatten()
                    if chosen.numel() == 0:
                        continue
                    part = grouped[:, chosen]
                    low = part.amin(dim=-1, keepdim=True).half()
                    scale = ((part.amax(dim=-1, keepdim=True) - low.float()) / (2**width - 1)).clamp_min(1e-8).half()
                    codes = ((part - low.float()) / scale.float()).round().clamp(0, 2**width - 1).to(torch.uint8)
                    parts += [_numpy_bytes(_pack(codes, width)), _numpy_bytes(low), _numpy_bytes(scale)]
            # Sinks and recent window: original FP16 values (keys still rotated), bit-identical.
            exact = torch.cat([tensors[:, :, :start], tensors[:, :, end:]], dim=2)
            parts.append(_numpy_bytes(self._matrix(exact).half()))
        payload = b"".join(parts)
        header = BLOB_HEADER.pack(MAGIC, FORMAT_VERSION, self.fingerprint, position_offset)
        return header + zlib.compress(payload, zlib_level)

    def decompress(self, blob: bytes, dtype=torch.float16, device=None) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Rebuild (keys, values, position_offset); tensors are [layers, kv_heads, tokens, head_dim]."""
        magic, version, fingerprint, position_offset = BLOB_HEADER.unpack_from(blob)
        if magic != MAGIC or version != FORMAT_VERSION:
            raise ValueError("Not a DimmerKV blob (or an unsupported format version).")
        if fingerprint != self.fingerprint:
            raise ValueError("Blob was written with a different calibration file.")
        keys, values = self.decode_payload(zlib.decompress(blob[BLOB_HEADER.size :]), position_offset, dtype)
        if device is not None:
            keys, values = keys.to(device), values.to(device)
        return keys, values, position_offset

    def decode_payload(self, data: bytes, position_offset: int = 0, dtype=torch.float16):
        """Decode the scripts' byte layout (``kvtc_storage.encode_bytes``) to stacked tensors."""
        offset = 0

        def read(dtype_np, count):
            nonlocal offset
            array = np.frombuffer(data, dtype=dtype_np, count=count, offset=offset)
            offset += array.nbytes
            return torch.from_numpy(array.copy()).to(self.device)

        tokens, start, end = read(PAYLOAD_HEADER, 3).tolist()
        middle = end - start
        cos, sin = self.rope.tables(position_offset, tokens)
        out = {}
        for kind in ("key", "value"):
            stream = self.streams[kind]
            used = stream.basis.shape[1]
            rungs = read(np.uint8, middle).long()
            exact_rung = len(stream.rung_bytes)
            decoded = torch.zeros(middle, used, device=self.device)
            exact_rows = None
            for rung in sorted(rungs.unique().tolist()):
                rows = rungs == rung
                count = int(rows.sum())
                if rung == exact_rung:
                    exact_rows = (rows, read(np.float16, count * self.features).float().view(count, self.features))
                    continue
                grouped = torch.zeros(count, used // self.group, self.group, device=self.device)
                bits = stream.rungs[rung]
                for width in BIT_OPTIONS[1:]:
                    chosen = (bits == width).nonzero().flatten()
                    if chosen.numel() == 0:
                        continue
                    n = count * chosen.numel() * self.group
                    codes = _unpack(read(np.uint8, _packed_length(n, width)), width, n)
                    low = read(np.float16, count * chosen.numel()).float().view(count, -1, 1)
                    scale = read(np.float16, count * chosen.numel()).float().view(count, -1, 1)
                    grouped[:, chosen] = codes.float().view(count, -1, self.group) * scale + low
                decoded[rows] = grouped.view(count, used)
            matrix = torch.empty(tokens, self.features, device=self.device)
            if middle:
                matrix[start:end] = self._inverse(decoded, stream)
            if exact_rows is not None:
                middle_rows = matrix[start:end]
                middle_rows[exact_rows[0]] = exact_rows[1]
            rebuilt = self._unmatrix(matrix)
            if kind == "key":
                rebuilt = self.rope.rerotate(rebuilt, cos, sin)
            rebuilt = rebuilt.to(dtype)
            exact = read(np.float16, (tokens - middle) * self.features).view(tokens - middle, self.features)
            exact = self._unmatrix(exact).to(dtype)
            rebuilt[:, :, :start] = exact[:, :, :start]
            rebuilt[:, :, end:] = exact[:, :, start:]
            out[kind] = rebuilt.contiguous()
        if offset != len(data):
            raise ValueError(f"Decoded {offset} of {len(data)} bytes.")
        return out["key"], out["value"]

    def fp16_bytes(self, tokens: int) -> int:
        """Size of these tokens in an uncompressed FP16 K+V cache, for ratios."""
        return tokens * 2 * self.exact_bytes
