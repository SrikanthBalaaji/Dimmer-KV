"""kvtc-style transform coding of the KV cache, plus DimmerKV per-token rungs.

Following kvtc (Staniszewski & Łańcucki, ICLR 2026): for each token, keys
(with RoPE removed) from every layer and KV head are concatenated into one
vector, likewise values; a PCA basis fitted once on calibration data
decorrelates it; coefficients are quantized in groups of consecutive
components with a per-group bit width (0/2/4/8) and a per-token FP16 shift
and scale per group. No entropy coding yet (sizes are "before DEFLATE").

The fitted file holds a *ladder* of bit allocations (rungs), from very
coarse to fine, built greedily on calibration data. ``kvtc`` gives every
token the same rung; ``DimmerKV-TC`` (``allocate_rungs``) gives each token
the rung where its own error drop per byte is best, under one byte budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

GROUP_BYTES_OVERHEAD = 4  # FP16 shift + FP16 scale per (token, group) with bits > 0.
BIT_OPTIONS = (0, 2, 4, 8)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope_tables(model, tokens: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """cos, sin [tokens, head_dim] in float32 for positions 0..tokens-1."""
    positions = torch.arange(tokens, device=device)[None]
    probe = torch.zeros(1, device=device, dtype=torch.float32)
    cos, sin = model.model.rotary_emb(probe, positions)
    return cos[0].float(), sin[0].float()


def derotate(keys: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Undo RoPE on [..., tokens, head_dim] keys (inverse rotation, any scaling)."""
    keys = keys.float()
    return (keys * cos - rotate_half(keys) * sin) / (cos.square() + sin.square())


def rerotate(keys: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return keys * cos + rotate_half(keys) * sin


def token_matrix(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Per-layer [1, heads, tokens, dim] -> [tokens, layers * heads * dim] (float32)."""
    return torch.cat([t[0].float().transpose(0, 1).reshape(t.shape[2], -1) for t in tensors], dim=1)


def split_token_matrix(matrix: torch.Tensor, layers: int, heads: int, head_dim: int) -> list[torch.Tensor]:
    """Inverse of ``token_matrix``: [tokens, features] -> per-layer [1, heads, tokens, dim]."""
    tokens = matrix.shape[0]
    blocks = matrix.view(tokens, layers, heads, head_dim)
    return [blocks[:, layer].transpose(0, 1)[None] for layer in range(layers)]


def quantize_groups(coefficients: torch.Tensor, bits: torch.Tensor, group: int) -> torch.Tensor:
    """Quantize-dequantize [tokens, features] with per-group bit widths.

    Each (token, group) with b > 0 bits uses asymmetric uniform levels between
    its own min and max (stored as FP16 shift and scale); b = 0 drops it.
    """
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


@dataclass
class Stream:
    mean: torch.Tensor  # [features]
    basis: torch.Tensor  # [features, features], columns ordered by variance
    rungs: torch.Tensor  # [rungs, groups] bit widths, coarse -> fine
    rung_bytes: list[int]


class KVTCCodec:
    """Loads a ``fit_kvtc.py`` file and encodes caches at given rungs."""

    def __init__(self, path: str | Path, device, model) -> None:
        saved = torch.load(path, map_location="cpu", weights_only=False)
        self.model_id = saved["model"]
        self.layers, self.heads, self.head_dim = saved["layers"], saved["heads"], saved["head_dim"]
        self.group = saved["group"]
        self.sinks = saved["sinks"]
        self.device = torch.device(device)
        self._model = model
        self.streams = {
            kind: Stream(
                saved[kind]["mean"].to(self.device),
                saved[kind]["basis"].to(self.device),
                saved[kind]["rungs"].to(self.device),
                list(saved[kind]["rung_bytes"]),
            )
            for kind in ("key", "value")
        }

    @property
    def model(self) -> str:
        return self.model_id

    @property
    def exact_bytes(self) -> int:
        return self.layers * self.heads * self.head_dim * 2

    def to_matrix(self, tensors, kind, cos, sin) -> torch.Tensor:
        if kind == "key":
            tensors = [derotate(t, cos, sin) for t in tensors]
        return token_matrix(tensors)

    def from_matrix(self, matrix, kind, cos, sin, dtype) -> list[torch.Tensor]:
        tensors = split_token_matrix(matrix, self.layers, self.heads, self.head_dim)
        if kind == "key":
            tensors = [rerotate(t, cos, sin) for t in tensors]
        return [t.to(dtype) for t in tensors]

    def rung_errors(self, coefficients: torch.Tensor, kind: str) -> torch.Tensor:
        """Squared error [tokens, rungs] of every token at every rung."""
        stream = self.streams[kind]
        return torch.stack(
            [(coefficients - quantize_groups(coefficients, bits, self.group)).square().sum(dim=1)
             for bits in stream.rungs],
            dim=1,
        )

    def compress(self, keys, values, budget: float, exact_recent: int, per_token: bool,
                 fill: bool = False, weights: torch.Tensor | None = None):
        """kvtc (same rung for all) or DimmerKV-TC (per-token rungs) under a budget.

        ``fill`` (uniform only) moves an evenly spaced set of tokens one rung up
        to use the leftover budget, so uniform and per-token runs spend the
        same bytes. ``weights`` [tokens] (per-token only) scales each token's
        errors by its importance (mean-normalized over compressed tokens).
        Returns rebuilt per-layer keys and values, counted bytes and a histogram.
        """
        tokens = keys[0].shape[2]
        start = min(self.sinks, tokens)
        end = max(start, tokens - exact_recent)
        middle = end - start
        cos, sin = rope_tables(self._model, tokens, self.device)
        budget_bytes = int(budget * tokens * 2 * self.exact_bytes)
        remaining = budget_bytes - (tokens - middle) * 2 * self.exact_bytes

        matrices, coefficients, errors = {}, {}, []
        for kind, tensors in (("key", keys), ("value", values)):
            stream = self.streams[kind]
            matrices[kind] = self.to_matrix(tensors, kind, cos, sin)
            coefficients[kind] = (matrices[kind][start:end] - stream.mean) @ stream.basis
            if per_token:
                # Relative error so key and value errors are on one scale; exact rung = 0.
                energy = (matrices[kind][start:end] - stream.mean).square().sum(dim=1).mean()
                rung_err = self.rung_errors(coefficients[kind], kind) / energy
                if weights is not None:
                    importance = weights[start:end].float()
                    rung_err = rung_err * (importance / importance.mean().clamp_min(1e-12))[:, None]
                errors.append(torch.cat([rung_err, torch.zeros_like(rung_err[:, :1])], dim=1))

        level_bytes = {kind: self.streams[kind].rung_bytes + [self.exact_bytes] for kind in ("key", "value")}
        if per_token:
            chosen = allocate_rungs(
                torch.cat(errors), torch.tensor(level_bytes["key"] + level_bytes["value"], device=self.device),
                middle, remaining,
            )
            choice = {"key": chosen[:middle], "value": chosen[middle:]}
        else:
            # Uniform: the finest rung (same for K and V) that fits.
            fits = [r for r in range(len(level_bytes["key"]))
                    if middle * (level_bytes["key"][r] + level_bytes["value"][r]) <= remaining]
            if not fits:
                raise ValueError(f"Budget {budget:.3f} cannot cover the exact tokens.")
            rung = max(fits)
            choice = {kind: torch.full((middle,), rung, device=self.device) for kind in ("key", "value")}
            if fill and rung + 1 < len(level_bytes["key"]):
                base = middle * (level_bytes["key"][rung] + level_bytes["value"][rung])
                step = (level_bytes["key"][rung + 1] - level_bytes["key"][rung]
                        + level_bytes["value"][rung + 1] - level_bytes["value"][rung])
                upgrades = min(middle, (remaining - base) // step)
                if upgrades > 0:
                    picked = torch.linspace(0, middle - 1, upgrades, device=self.device).round().long()
                    for kind in choice:
                        choice[kind][picked] = rung + 1

        used = (tokens - middle) * 2 * self.exact_bytes
        rebuilt, histogram = {}, {}
        for kind, tensors in (("key", keys), ("value", values)):
            stream = self.streams[kind]
            matrix = matrices[kind].clone()
            decoded = torch.zeros_like(coefficients[kind])
            exact_rung = len(stream.rung_bytes)
            for rung in choice[kind].unique().tolist():
                rows = choice[kind] == rung
                used += int(rows.sum()) * level_bytes[kind][rung]
                if rung == exact_rung:
                    decoded[rows] = coefficients[kind][rows]
                else:
                    decoded[rows] = quantize_groups(coefficients[kind][rows], stream.rungs[rung], self.group)
                share = float(rows.float().mean()) / 2
                name = "exact" if rung == exact_rung else f"{level_bytes[kind][rung] / self.exact_bytes:.3f}"
                histogram[name] = histogram.get(name, 0.0) + share
            matrix[start:end] = decoded @ stream.basis.T + stream.mean
            rebuilt[kind] = self.from_matrix(matrix, kind, cos, sin, tensors[0].dtype)
            # Exact tokens keep their original (bit-identical) values.
            for layer, original in enumerate(tensors):
                rebuilt[kind][layer][..., :start, :] = original[..., :start, :]
                rebuilt[kind][layer][..., end:, :] = original[..., end:, :]
        if used > budget_bytes:
            raise AssertionError(f"kvtc used {used} bytes over budget {budget_bytes}.")
        return rebuilt["key"], rebuilt["value"], used, histogram


def allocate_rungs(errors: torch.Tensor, level_bytes: torch.Tensor, middle: int, budget: int) -> torch.Tensor:
    """Greedy per-slot rung choice (DimmerKV flat) under a byte budget.

    ``errors`` is [slots, levels] (first ``middle`` slots are keys, the rest
    values) and ``level_bytes`` holds each stream's level sizes back to back.
    Every slot starts at its coarsest rung; an upgrade's value is error drop
    per extra byte, made non-increasing along levels with ``cummin`` so the
    stable greedy pick always takes each slot's upgrades in order.
    """
    slots, levels = errors.shape
    sizes = level_bytes.view(2, levels).repeat_interleave(middle, dim=0).double()  # [slots, levels]
    base = int(sizes[:, 0].sum())
    if base > budget:
        raise ValueError("Budget cannot cover the coarsest rung for every token.")
    gains = (errors[:, :-1] - errors[:, 1:]).clamp_min(0).double()
    costs = sizes[:, 1:] - sizes[:, :-1]
    value = (gains / costs).cummin(dim=1).values
    order = torch.sort(value.reshape(-1), descending=True, stable=True).indices
    taken = order[costs.reshape(-1)[order].cumsum(0) <= budget - base]
    counts = torch.zeros(slots, dtype=torch.long, device=errors.device)
    counts.index_add_(0, taken // (levels - 1), torch.ones_like(taken))
    return counts
