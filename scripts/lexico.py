"""Lexico sparse coding: batched orthogonal matching pursuit over a dictionary.

A code is an ordered list of (atom index, coefficient) pairs. OMP picks atoms
greedily, so the first j indices are exactly the OMP choice at sparsity j
(prefix property the allocator relies on); ``refit`` recomputes the
least-squares coefficients for a shorter prefix. Byte accounting follows the plan: 3 B per atom
(12-bit index + 12-bit value) plus 2 B for the first atom's scale.
"""

from __future__ import annotations

import torch

FIRST_ATOM_EXTRA_BYTES = 2
BYTES_PER_ATOM = 3


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
