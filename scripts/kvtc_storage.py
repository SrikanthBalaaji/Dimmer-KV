"""Real byte storage for kvtc / DimmerKV-TC caches.

``encode_bytes`` writes the codes that ``KVTCCodec.compress`` chose into one
byte string: per stream, the rung of every compressed token (1 byte each),
then for each rung its bit-packed 2/4/8-bit codes with FP16 shift and scale
per (token, group), then the exact tokens (sinks and recent window) as FP16.
``decode_bytes`` reads it back into per-layer cache tensors. The byte string
can then be DEFLATE-compressed with ``zlib`` as in kvtc.
"""

from __future__ import annotations

import numpy as np
import torch

from kvtc import BIT_OPTIONS, KVTCCodec, rope_tables, split_token_matrix, token_matrix

HEADER = np.dtype(np.int32)


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


def encode_bytes(codec: KVTCCodec, keys, values, choice: dict, exact_recent: int) -> bytes:
    """Serialize a compressed cache. ``choice`` is ``codec.last_choice``."""
    tokens = keys[0].shape[2]
    start = min(codec.sinks, tokens)
    end = max(start, tokens - exact_recent)
    cos, sin = rope_tables(codec._model, tokens, codec.device)
    parts = [np.array([tokens, start, end], dtype=HEADER).tobytes()]
    for kind, tensors in (("key", keys), ("value", values)):
        stream = codec.streams[kind]
        matrix = codec.to_matrix(tensors, kind, cos, sin)
        coefficients = (matrix[start:end] - stream.mean) @ stream.basis
        rungs = choice[kind]
        parts.append(rungs.to(torch.uint8).cpu().numpy().tobytes())
        exact_rung = len(stream.rung_bytes)
        for rung in sorted(rungs.unique().tolist()):
            rows = rungs == rung
            if rung == exact_rung:
                parts.append(matrix[start:end][rows].half().cpu().numpy().tobytes())
                continue
            grouped = coefficients[rows].view(int(rows.sum()), -1, codec.group)
            bits = stream.rungs[rung]
            for width in BIT_OPTIONS[1:]:
                chosen = (bits == width).nonzero().flatten()
                if chosen.numel() == 0:
                    continue
                part = grouped[:, chosen]
                # Same arithmetic as kvtc.quantize_groups, so decoding matches compress().
                low = part.amin(dim=-1, keepdim=True).half()
                scale = ((part.amax(dim=-1, keepdim=True) - low.float()) / (2**width - 1)).clamp_min(1e-8).half()
                codes = ((part - low.float()) / scale.float()).round().clamp(0, 2**width - 1).to(torch.uint8)
                parts += [_pack(codes, width).cpu().numpy().tobytes(),
                          low.cpu().numpy().tobytes(), scale.cpu().numpy().tobytes()]
        # Sinks and recent window: original FP16 values, bit-identical.
        exact = token_matrix(tensors)
        parts.append(torch.cat([exact[:start], exact[end:]]).half().cpu().numpy().tobytes())
    return b"".join(parts)


def decode_bytes(codec: KVTCCodec, data: bytes, dtype=torch.float16):
    """Rebuild per-layer keys and values [1, heads, tokens, dim] from ``encode_bytes`` output."""
    offset = 0

    def read(dtype_np, count):
        nonlocal offset
        array = np.frombuffer(data, dtype=dtype_np, count=count, offset=offset)
        offset += array.nbytes
        return torch.from_numpy(array.copy()).to(codec.device)

    tokens, start, end = read(HEADER, 3).tolist()
    middle = end - start
    features = codec.layers * codec.heads * codec.head_dim
    cos, sin = rope_tables(codec._model, tokens, codec.device)
    out = {}
    for kind in ("key", "value"):
        stream = codec.streams[kind]
        rungs = read(np.uint8, middle).long()
        exact_rung = len(stream.rung_bytes)
        decoded = torch.zeros(middle, features, device=codec.device)
        exact_rows = None
        for rung in sorted(rungs.unique().tolist()):
            rows = rungs == rung
            count = int(rows.sum())
            if rung == exact_rung:
                exact_rows = (rows, read(np.float16, count * features).float().view(count, features))
                continue
            grouped = torch.zeros(count, features // codec.group, codec.group, device=codec.device)
            bits = stream.rungs[rung]
            for width in BIT_OPTIONS[1:]:
                chosen = (bits == width).nonzero().flatten()
                if chosen.numel() == 0:
                    continue
                values = count * chosen.numel() * codec.group
                codes = _unpack(read(np.uint8, _packed_length(values, width)), width, values)
                low = read(np.float16, count * chosen.numel()).float().view(count, -1, 1)
                scale = read(np.float16, count * chosen.numel()).float().view(count, -1, 1)
                grouped[:, chosen] = codes.float().view(count, -1, codec.group) * scale + low
            decoded[rows] = grouped.view(count, features)
        matrix = torch.empty(tokens, features, device=codec.device)
        matrix[start:end] = decoded @ stream.basis.T + stream.mean
        if exact_rows is not None:
            middle_rows = matrix[start:end]
            middle_rows[exact_rows[0]] = exact_rows[1]
        rebuilt = codec.from_matrix(matrix, kind, cos, sin, dtype)
        exact = read(np.float16, (tokens - middle) * features).view(tokens - middle, features)
        exact_layers = split_token_matrix(exact, codec.layers, codec.heads, codec.head_dim)
        for layer, original in enumerate(exact_layers):
            rebuilt[layer][..., :start, :] = original[..., :start, :].to(dtype)
            rebuilt[layer][..., end:, :] = original[..., start:, :].to(dtype)
        out[kind] = rebuilt
    if offset != len(data):
        raise ValueError(f"Decoded {offset} of {len(data)} bytes.")
    return out["key"], out["value"]
