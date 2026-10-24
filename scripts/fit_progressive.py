"""Build the progressive (group, bit-plane) order on top of a kvtc basis.

Collects kvtc coefficients for WikiText-2 train tokens (sinks excluded),
quantizes them once at 8 bits per group, and orders the pieces greedily by
error removed per byte (``progressive.build_order``). Also reports how much
accuracy the nested code loses against the kvtc ladder at the same sizes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import DEFAULT_MODEL, cache_tensors, load_model, wikitext_token_chunks
from fit_kvtc import TARGETS
from kvtc import KVTCCodec, derotate, quantize_groups, rope_tables, token_matrix
from progressive import build_order, dequantize, quantize8


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--kvtc", default="checkpoints/kvtc.pt")
    parser.add_argument("--chunks", type=int, default=24)
    parser.add_argument("--chunk-tokens", type=int, default=1024)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/progressive.pt"))
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model)
    codec = KVTCCodec(args.kvtc, device, model)
    cos, sin = rope_tables(model, args.chunk_tokens, device)
    # Use chunks from the end of the train split so they differ from kvtc's calibration.
    chunks = wikitext_token_chunks(tokenizer, "train", args.chunk_tokens, 100 + args.chunks)[100:]
    samples = {"key": [], "value": []}
    for chunk in chunks:
        with torch.inference_mode():
            keys, values = cache_tensors(model(input_ids=chunk[None].to(device), use_cache=True).past_key_values)
            length = keys[0].shape[2]
            for kind, matrix in (("key", token_matrix([derotate(k, cos[:length], sin[:length]) for k in keys])),
                                 ("value", token_matrix(values))):
                stream = codec.streams[kind]
                samples[kind].append(((matrix[codec.sinks:] - stream.mean) @ stream.basis)[::4])

    saved: dict[str, object] = {"model": args.model, "kvtc": str(args.kvtc)}
    exact = codec.exact_bytes
    for kind in ("key", "value"):
        coefficients = torch.cat(samples[kind])
        order = build_order(coefficients, codec.group, TARGETS, exact)
        saved[kind] = {"planes": order.planes.cpu(), "prefix_bytes": order.prefix_bytes.cpu(), "rungs": order.rungs}
        codes, low, span = quantize8(coefficients, codec.group)
        energy = coefficients.square().sum()
        ladder = {codec.streams[kind].rung_bytes[i]: bits for i, bits in enumerate(codec.streams[kind].rungs)}
        print(f"{kind}: {coefficients.shape[0]} sample tokens, {len(order.rungs)} rungs, "
              f"full stream {int(order.prefix_bytes[-1])} B ({order.prefix_bytes[-1] / exact:.3f} of FP16)")
        for k in order.rungs:
            size = int(order.prefix_bytes[k])
            error = float((coefficients - dequantize(codes, low, span, order.planes[k][None].to(device))).square().sum() / energy)
            # Nearest kvtc ladder rung at or below this size, for comparison.
            below = [b for b in ladder if b <= size]
            note = ""
            if below:
                bits = ladder[max(below)].to(device)
                ladder_error = float((coefficients - quantize_groups(coefficients, bits, codec.group)).square().sum() / energy)
                note = f" | kvtc ladder at {max(below)} B: {ladder_error:.4f}"
            print(f"  prefix {size:>5} B ({size / exact:.3f}): rel err {error:.4f}{note}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, args.output)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
