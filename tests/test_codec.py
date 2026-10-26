from __future__ import annotations

import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dimmerkv.codec import BLOB_HEADER, DimmerKVCodec

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def relative_error(rebuilt: torch.Tensor, original: torch.Tensor) -> float:
    return float((rebuilt.float() - original.float()).square().sum() / original.float().square().sum())


def fake_hf_model(rope):
    """Just enough of an HF model for scripts/kvtc.py's rope_tables()."""

    def rotary_emb(probe, positions):
        cos, sin = rope.tables(0, positions.shape[1])
        return cos[None], sin[None]

    return SimpleNamespace(model=SimpleNamespace(rotary_emb=rotary_emb))


@pytest.mark.parametrize("budget,floor_share", [(0.15, 1.0), (0.25, 0.0), (0.4, 0.5)])
def test_bytes_match_original_scripts(checkpoint, synthetic, budget, floor_share):
    """The payload is byte-for-byte what scripts/kvtc.py + kvtc_storage.py produce."""
    sys.path.insert(0, str(SCRIPTS))
    try:
        import kvtc
        import kvtc_storage
    finally:
        sys.path.remove(str(SCRIPTS))
    codec = DimmerKVCodec(checkpoint)
    keys, values = synthetic.cache(600, seed=7)
    original = kvtc.KVTCCodec(checkpoint, "cpu", fake_hf_model(codec.rope))
    key_layers, value_layers = [k[None] for k in keys], [v[None] for v in values]
    original.compress(key_layers, value_layers, budget, 32, per_token=True, floor_share=floor_share)
    expected = kvtc_storage.encode_bytes(original, key_layers, value_layers, original.last_choice, 32)

    blob = codec.compress(keys, values, budget=budget, recent=32, floor_share=floor_share)
    assert zlib.decompress(blob[BLOB_HEADER.size :]) == expected

    old_keys, old_values = kvtc_storage.decode_bytes(original, expected)
    new_keys, new_values, offset = codec.decompress(blob)
    assert offset == 0
    assert torch.equal(torch.stack([k[0] for k in old_keys]), new_keys)
    assert torch.equal(torch.stack([v[0] for v in old_values]), new_values)


def test_round_trip_quality_and_size(checkpoint, synthetic):
    codec = DimmerKVCodec(checkpoint)
    keys, values = synthetic.cache(2000, seed=3)
    coarse, fine = (codec.compress(keys, values, budget=b) for b in (0.05, 0.3))
    assert len(coarse) < len(fine) < codec.fp16_bytes(2000) * 0.3
    errors = []
    for blob in (coarse, fine):
        k, v, _ = codec.decompress(blob)
        assert k.shape == keys.shape and k.dtype == torch.float16
        errors.append(relative_error(k, keys) + relative_error(v, values))
    assert errors[1] < errors[0] < 0.2
    # Sinks and the recent window are bit-identical.
    k, v, _ = codec.decompress(fine)
    assert torch.equal(k[:, :, :4], keys[:, :, :4]) and torch.equal(v[:, :, -32:], values[:, :, -32:])


def test_position_offset_segment(checkpoint, synthetic):
    """A segment that starts mid-conversation derotates with its own positions."""
    codec = DimmerKVCodec(checkpoint)
    keys, values = synthetic.cache(512, seed=5, offset=1000)
    good = codec.decompress(codec.compress(keys, values, budget=0.2, position_offset=1000))
    wrong = codec.decompress(codec.compress(keys, values, budget=0.2, position_offset=0))
    assert good[2] == 1000
    assert relative_error(good[0], keys) < 0.5 * relative_error(wrong[0], keys)


def test_truncated_basis(checkpoint, synthetic):
    full = DimmerKVCodec(checkpoint)
    small = DimmerKVCodec(checkpoint, max_rung_fraction=1 / 8)
    assert small.streams["key"].basis.shape[1] <= full.streams["key"].basis.shape[1]
    keys, values = synthetic.cache(800, seed=11)
    k, v, _ = small.decompress(small.compress(keys, values, budget=0.1))
    assert relative_error(k, keys) < 0.2


def test_short_sequences_and_wrong_codec(checkpoint, synthetic):
    codec = DimmerKVCodec(checkpoint)
    for tokens in (1, 20, 40):  # All exact, or a budget below the coarsest rung.
        keys, values = synthetic.cache(tokens, seed=tokens)
        k, v, _ = codec.decompress(codec.compress(keys, values))
        assert k.shape == keys.shape
    blob = bytearray(codec.compress(*synthetic.cache(64, seed=1)))
    blob[5] ^= 1  # Flip a fingerprint byte.
    with pytest.raises(ValueError, match="different calibration"):
        codec.decompress(bytes(blob))
