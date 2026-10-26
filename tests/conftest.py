"""Synthetic calibration shared by the tests: a small "model" with correlated, RoPE-rotated keys."""

from __future__ import annotations

import pytest
import torch

from dimmerkv.calibrate import Accumulator, finalize
from dimmerkv.rope import RoPE

LAYERS, HEADS, HEAD_DIM, GROUP, SINKS = 3, 2, 32, 16, 4
FEATURES = LAYERS * HEADS * HEAD_DIM


class SyntheticModel:
    """Generates caches whose pre-RoPE vectors live near a low-rank subspace, like real KV caches."""

    def __init__(self, seed: int = 0, rank: int = 24) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.mixing = {kind: torch.randn(rank, FEATURES, generator=generator) / rank**0.5 for kind in ("key", "value")}
        self.offset = {kind: torch.randn(FEATURES, generator=generator) for kind in ("key", "value")}
        self.rope = RoPE(1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM)))
        self.rank = rank

    def matrices(self, tokens: int, seed: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(seed)
        out = {}
        for kind in ("key", "value"):
            scales = torch.linspace(3, 0.2, self.rank)
            latent = torch.randn(tokens, self.rank, generator=generator) * scales
            out[kind] = latent @ self.mixing[kind] + self.offset[kind] + 0.05 * torch.randn(tokens, FEATURES, generator=generator)
        return out

    def cache(self, tokens: int, seed: int, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Keys (RoPE applied at positions offset..) and values [layers, heads, tokens, dim], float16."""
        matrices = self.matrices(tokens, seed)
        stacked = {kind: m.view(tokens, LAYERS, HEADS, HEAD_DIM).permute(1, 2, 0, 3) for kind, m in matrices.items()}
        cos, sin = self.rope.tables(offset, tokens)
        keys = self.rope.rerotate(stacked["key"], cos, sin)
        return keys.half().contiguous(), stacked["value"].half().contiguous()


@pytest.fixture(scope="session")
def synthetic():
    return SyntheticModel()


@pytest.fixture(scope="session")
def checkpoint(synthetic, tmp_path_factory):
    streams = {kind: Accumulator(FEATURES, "cpu", sample_every=3) for kind in ("key", "value")}
    for seed in range(8):
        matrices = synthetic.matrices(512, 100 + seed)
        for kind in ("key", "value"):
            streams[kind].add(matrices[kind][SINKS:])
    saved = {"model": "synthetic", "layers": LAYERS, "heads": HEADS, "head_dim": HEAD_DIM, "group": GROUP,
             "sinks": SINKS, "rope": synthetic.rope.state()}
    for kind in ("key", "value"):
        saved[kind] = streams[kind].fit(GROUP, "cpu", log=lambda *_: None)
    finalize(saved)
    path = tmp_path_factory.mktemp("calibration") / "synthetic.pt"
    torch.save(saved, path)
    return path
