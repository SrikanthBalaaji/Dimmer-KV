"""PCA codec for per-layer Hugging Face KV-cache tensors.

The saved basis holds all head_dim ordered components; ``components`` picks a
prefix, so a code with k coefficients is the first k entries of a longer code.
Coefficients are stored as FP16 (2 B each) or INT8 (1 B each plus one FP16
scale per vector).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from common import uniform_compress

SCALE_BYTES = 2


@dataclass(frozen=True)
class PCAStream:
    """PCA representation for keys or values in one layer.

    ``mean`` is [groups, head_dim] and ``basis`` is [groups, head_dim, head_dim]
    with rows ordered by explained variance. ``groups`` is 1 (shared across
    KV heads) or the number of KV heads (one basis per head).
    """

    mean: torch.Tensor
    basis: torch.Tensor


def quantize_int8(coefficients: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric INT8 with one FP16 absmax scale per vector."""
    scale = coefficients.abs().amax(dim=-1, keepdim=True).float().clamp_min(1e-8) / 127
    codes = (coefficients.float() / scale).round().clamp(-127, 127).to(torch.int8)
    return codes, scale.half()


def dequantize_int8(codes: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (codes.float() * scale.float()).to(dtype)


def bytes_per_vector(components: int, bits: int) -> int:
    """Stored bytes for one head vector's code."""
    if bits == 16:
        return components * 2
    if bits == 8:
        return components + SCALE_BYTES
    raise ValueError("bits must be 8 or 16")


class PCACodec:
    """Encode and decode cache tensors using a saved per-layer PCA basis."""

    def __init__(self, basis_file: str | Path, device: torch.device | str) -> None:
        saved = torch.load(basis_file, map_location="cpu", weights_only=False)
        layers = saved.get("layers")
        if not isinstance(layers, list) or not layers:
            raise ValueError("PCA basis must contain a non-empty layers list.")
        if "kv_heads" not in saved:
            raise ValueError("Old basis format; rerun fit_pca_basis.py.")

        self.model = saved["model"]
        self.head_dim = int(saved["head_dim"])
        self.kv_heads = int(saved["kv_heads"])
        self.max_components = int(saved["components"])
        self.per_head = bool(saved["per_head"])
        self.skip_prefix = int(saved["skip_prefix"])
        self.device = torch.device(device)
        self._streams = [
            {
                kind: PCAStream(
                    layer[f"{kind}_mean"].to(self.device),
                    layer[f"{kind}_basis"].to(self.device),
                )
                for kind in ("key", "value")
            }
            for layer in layers
        ]

    @property
    def layers(self) -> int:
        return len(self._streams)

    def _stream(self, layer: int, kind: str) -> PCAStream:
        if not 0 <= layer < self.layers:
            raise IndexError(f"layer must be between 0 and {self.layers - 1}")
        if kind not in {"key", "value"}:
            raise ValueError("kind must be 'key' or 'value'")
        return self._streams[layer][kind]

    def _check(self, tensor: torch.Tensor, last_dim: int, name: str) -> None:
        if tensor.dim() < 3:
            raise ValueError(f"{name} must be shaped [..., kv_heads, tokens, {last_dim}].")
        if tensor.shape[-1] != last_dim:
            raise ValueError(f"Expected {name} final dimension {last_dim}, got {tensor.shape[-1]}.")
        if self.per_head and tensor.shape[-3] != self.kv_heads:
            raise ValueError(f"Expected {self.kv_heads} KV heads, got {tensor.shape[-3]}.")

    def encode(
        self, vectors: torch.Tensor, layer: int, kind: str, components: int
    ) -> torch.Tensor:
        """Project [..., kv_heads, tokens, head_dim] to [..., kv_heads, tokens, components]."""
        if not 1 <= components <= self.max_components:
            raise ValueError(f"components must be between 1 and {self.max_components}.")
        self._check(vectors, self.head_dim, "vectors")
        stream = self._stream(layer, kind)
        mean = stream.mean.to(vectors.dtype)[:, None]
        basis = stream.basis[:, :components].to(vectors.dtype)
        return (vectors - mean) @ basis.transpose(-1, -2)

    def decode(self, coefficients: torch.Tensor, layer: int, kind: str) -> torch.Tensor:
        """Reconstruct [..., kv_heads, tokens, head_dim] from a code prefix."""
        components = coefficients.shape[-1]
        if not 1 <= components <= self.max_components:
            raise ValueError(f"Got {components} components; max is {self.max_components}.")
        self._check(coefficients, components, "coefficients")
        stream = self._stream(layer, kind)
        mean = stream.mean.to(coefficients.dtype)[:, None]
        basis = stream.basis[:, :components].to(coefficients.dtype)
        return coefficients @ basis + mean

    def encode_codes(
        self, vectors: torch.Tensor, layer: int, kind: str, components: int, bits: int
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Stored code: (INT8 codes, FP16 scale) or (FP16 coefficients, None)."""
        coefficients = self.encode(vectors, layer, kind, components)
        if bits == 8:
            return quantize_int8(coefficients)
        if bits == 16:
            return coefficients.half(), None
        raise ValueError("bits must be 8 or 16")

    def decode_codes(
        self, codes: tuple[torch.Tensor, torch.Tensor | None], layer: int, kind: str, dtype: torch.dtype
    ) -> torch.Tensor:
        values, scale = codes
        coefficients = values.to(dtype) if scale is None else dequantize_int8(values, scale, dtype)
        return self.decode(coefficients, layer, kind)

    def round_trip(
        self, vectors: torch.Tensor, layer: int, kind: str, components: int, bits: int
    ) -> torch.Tensor:
        codes = self.encode_codes(vectors, layer, kind, components, bits)
        return self.decode_codes(codes, layer, kind, vectors.dtype)

    def compress_cache(
        self,
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
        components: int,
        bits: int,
        exact_prefix: int | None = None,
        exact_recent: int = 0,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], int]:
        """Uniform PCA: every middle token gets the same code length.

        The first ``exact_prefix`` positions (default: the sink positions the
        basis was fit without) and the last ``exact_recent`` positions stay
        exact FP16. Returns reconstructed tensors and the counted bytes.
        """
        if len(keys) != self.layers or len(values) != self.layers:
            raise ValueError("The cache layer count does not match the codec.")
        new_keys, key_bytes = self.compress_stream(keys, "key", components, bits, exact_prefix, exact_recent)
        new_values, value_bytes = self.compress_stream(values, "value", components, bits, exact_prefix, exact_recent)
        return new_keys, new_values, key_bytes + value_bytes

    def compress_stream(
        self,
        tensors: list[torch.Tensor],
        kind: str,
        components: int,
        bits: int,
        exact_prefix: int | None = None,
        exact_recent: int = 0,
    ) -> tuple[list[torch.Tensor], int]:
        """Compress one stream (K or V); returns rebuilt tensors and bytes."""
        return uniform_compress(
            tensors,
            lambda tensor, layer: self.round_trip(tensor, layer, kind, components, bits),
            bytes_per_vector(components, bits),
            self.skip_prefix if exact_prefix is None else exact_prefix,
            exact_recent,
        )
