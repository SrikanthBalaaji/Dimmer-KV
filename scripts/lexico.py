"""Lexico sparse coding: batched orthogonal matching pursuit over a dictionary.

A code is an ordered list of (atom index, coefficient) pairs. OMP picks atoms
greedily, so the first j indices are exactly the OMP choice at sparsity j
(prefix property the allocator relies on); ``refit`` recomputes the
least-squares coefficients for a shorter prefix. Byte accounting follows the plan: 3 B per atom
(12-bit index + 12-bit value) plus 2 B for one FP16 scale per vector.
"""

from __future__ import annotations

from pathlib import Path

import torch

from common import uniform_compress

FIRST_ATOM_EXTRA_BYTES = 2
BYTES_PER_ATOM = 3
VALUE_LEVELS = 2047  # Symmetric 12-bit coefficients.
ENCODE_CHUNK = 16384


def bytes_per_vector(atoms: int) -> int:
    return atoms * BYTES_PER_ATOM + (FIRST_ATOM_EXTRA_BYTES if atoms else 0)


def omp(
    vectors: torch.Tensor, dictionary: torch.Tensor, sparsity: int, ridge: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse-code [batch, dim] vectors over unit-norm atoms [dim, atoms].

    Returns atom indices [batch, sparsity] in selection order and the final
    least-squares coefficients [batch, sparsity].
    """
    batch = vectors.shape[0]
    residual = vectors
    indices = torch.empty(batch, 0, dtype=torch.long, device=vectors.device)
    eye = torch.eye(sparsity, device=vectors.device, dtype=vectors.dtype)
    coefficients = vectors.new_zeros(batch, 0)
    for step in range(sparsity):
        correlation = (residual @ dictionary).abs()
        if step:
            correlation.scatter_(1, indices, -1.0)
        indices = torch.cat([indices, correlation.argmax(dim=1, keepdim=True)], dim=1)
        atoms = dictionary.T[indices]  # [batch, step + 1, dim]
        gram = atoms @ atoms.transpose(1, 2) + ridge * eye[: step + 1, : step + 1]
        target = atoms @ vectors[:, :, None]
        coefficients = torch.linalg.solve(gram, target)[..., 0]
        residual = vectors - (coefficients[:, None] @ atoms)[:, 0]
    return indices, coefficients


def reconstruct(indices: torch.Tensor, coefficients: torch.Tensor, dictionary: torch.Tensor) -> torch.Tensor:
    """Rebuild [batch, dim] vectors from a code prefix of any length."""
    atoms = dictionary.T[indices]
    return (coefficients[:, None] @ atoms)[:, 0]


def refit(vectors: torch.Tensor, indices: torch.Tensor, dictionary: torch.Tensor, ridge: float = 1e-6) -> torch.Tensor:
    """Least-squares coefficients for a fixed atom prefix [batch, j]."""
    atoms = dictionary.T[indices]
    eye = torch.eye(indices.shape[1], device=vectors.device, dtype=vectors.dtype)
    gram = atoms @ atoms.transpose(1, 2) + ridge * eye
    return torch.linalg.solve(gram, atoms @ vectors[:, :, None])[..., 0]


def quantize_12bit(coefficients: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric 12-bit values (held in int16) with one FP16 absmax scale per vector."""
    scale = coefficients.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / VALUE_LEVELS
    codes = (coefficients / scale).round().clamp(-VALUE_LEVELS, VALUE_LEVELS).to(torch.int16)
    return codes, scale.half()


class LexicoCodec:
    """Encode and decode cache tensors with saved per-layer Lexico dictionaries."""

    def __init__(self, dictionary_file: str | Path, device: torch.device | str) -> None:
        saved = torch.load(dictionary_file, map_location="cpu", weights_only=False)
        self.model = saved["model"]
        self.head_dim = int(saved["head_dim"])
        self.atoms = int(saved["atoms"])
        self.skip_prefix = int(saved["skip_prefix"])
        self.device = torch.device(device)
        self._dictionaries = [
            {kind: layer[f"{kind}_dict"].float().to(self.device) for kind in ("key", "value")}
            for layer in saved["layers"]
        ]

    @property
    def layers(self) -> int:
        return len(self._dictionaries)

    def encode_codes(
        self, vectors: torch.Tensor, layer: int, kind: str, sparsity: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Code [..., head_dim] vectors as (atom indices, 12-bit values, FP16 scale)."""
        dictionary = self._dictionaries[layer][kind]
        flat = vectors.reshape(-1, self.head_dim).float()
        index_parts, coefficient_parts = [], []
        for start in range(0, flat.shape[0], ENCODE_CHUNK):
            indices, coefficients = omp(flat[start : start + ENCODE_CHUNK], dictionary, sparsity)
            index_parts.append(indices)
            coefficient_parts.append(coefficients)
        codes, scale = quantize_12bit(torch.cat(coefficient_parts))
        shape = vectors.shape[:-1]
        return (
            torch.cat(index_parts).to(torch.int16).reshape(*shape, sparsity),
            codes.reshape(*shape, sparsity),
            scale.reshape(*shape, 1),
        )

    def decode_codes(
        self, codes: tuple[torch.Tensor, torch.Tensor, torch.Tensor], layer: int, kind: str, dtype: torch.dtype
    ) -> torch.Tensor:
        indices, values, scale = codes
        dictionary = self._dictionaries[layer][kind]
        coefficients = values.float() * scale.float()
        atoms = dictionary.T[indices.long()]  # [..., sparsity, head_dim]
        return (coefficients[..., None, :] @ atoms)[..., 0, :].to(dtype)

    def round_trip(self, vectors: torch.Tensor, layer: int, kind: str, sparsity: int) -> torch.Tensor:
        codes = self.encode_codes(vectors, layer, kind, sparsity)
        return self.decode_codes(codes, layer, kind, vectors.dtype)

    def compress_cache(
        self,
        keys: list[torch.Tensor],
        values: list[torch.Tensor],
        sparsity: int,
        exact_prefix: int | None = None,
        exact_recent: int = 0,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], int]:
        """Uniform Lexico: every middle token gets ``sparsity`` atoms."""
        if len(keys) != self.layers or len(values) != self.layers:
            raise ValueError("The cache layer count does not match the dictionaries.")
        return uniform_compress(
            keys,
            values,
            lambda tensor, layer, kind: self.round_trip(tensor, layer, kind, sparsity),
            bytes_per_vector(sparsity),
            self.skip_prefix if exact_prefix is None else exact_prefix,
            exact_recent,
        )
