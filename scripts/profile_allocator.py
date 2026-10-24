"""Time each stage of the allocator on one needle context.

Stages: OMP encode with per-prefix errors, greedy selection, and rebuild
(refit + quantize + decode), each CUDA-synchronized, plus the end-to-end
``allocate`` call. Also checks that OMP prefix errors match a direct refit.
"""

from __future__ import annotations

import argparse
import time

import torch

import dimmer
import lexico
from bench_codecs import synchronize
from common import DEFAULT_MODEL, load_model, wikitext_token_chunks
from lexico import LexicoCodec
from test_generation import build_prompt, split_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dictionary", default="checkpoints/lexico_dict.pt")
    parser.add_argument("--context-tokens", type=int, default=1500)
    parser.add_argument("--budget", type=float, default=0.15)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model)
    dimmer.install(model)
    codec = LexicoCodec(args.dictionary, device)
    filler = wikitext_token_chunks(tokenizer, "test", args.context_tokens, 1)[0]
    context, qa = build_prompt(tokenizer.decode(filler[1:]), 0)
    context_ids = split_ids(tokenizer, context, qa[0][0])[0]

    def timed(fn):
        synchronize(device)
        started = time.perf_counter()
        result = fn()
        synchronize(device)
        return result, (time.perf_counter() - started) * 1000

    with torch.inference_mode():
        (cache, scores), prefill_ms = timed(lambda: dimmer.prefill_with_scores(model, context_ids[None].to(device)))
        keys = [layer.keys for layer in cache.layers]
        values = [layer.values for layer in cache.layers]
        smax = max(dimmer.DEFAULT_ATOM_LEVELS)
        vectors = keys[0][0, :, 1:-32].float().reshape(-1, keys[0].shape[-1])
        dictionary = codec.dictionary(0, "key")

        # Correctness: the error OMP reports for each prefix must match a refit.
        indices, errors = dimmer._encode_with_errors(vectors, dictionary, smax)
        worst = 0.0
        for sparsity in (1, 4, 8, 16):
            prefix = indices[:, :sparsity]
            rebuilt = lexico.reconstruct(prefix, lexico.refit(vectors, prefix, dictionary), dictionary)
            direct = (vectors - rebuilt).square().sum(dim=1)
            worst = max(worst, float(((direct - errors[:, sparsity]).abs() / vectors.square().sum(dim=1)).max()))
        print(f"tokens: {context_ids.shape[0]}, prefill {prefill_ms:.0f} ms, "
              f"max prefix-error mismatch (relative): {worst:.2e}")

        encode_ms = sum(
            timed(lambda: dimmer._encode_with_errors(vectors, dictionary, smax))[1] for _ in range(args.repeats)
        ) / args.repeats
        streams = len(keys) * 2
        print(f"OMP to {smax} atoms, one layer/stream ({vectors.shape[0]} vectors): {encode_ms:.1f} ms "
              f"-> x{streams} streams = {encode_ms * streams:.0f} ms")
        total_ms = sum(
            timed(lambda: dimmer.allocate(keys, values, [torch.ones_like(s) for s in scores["sum"]],
                                          codec, args.budget))[1]
            for _ in range(args.repeats)
        ) / args.repeats
        print(f"allocate() end to end at {args.budget:.0%}: {total_ms:.0f} ms "
              f"(non-OMP part ~{total_ms - encode_ms * streams:.0f} ms)")


if __name__ == "__main__":
    main()
