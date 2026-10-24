"""Compare the orthonormal (Gram-Schmidt, FP16 search) OMP with the original.

On held-out WikiText cache vectors, for every layer and stream: share of
atoms chosen identically, reconstruction error at several sparsities (after
12-bit quantization, as stored), and encode time.
"""

from __future__ import annotations

import argparse
import time

import torch

import lexico
from bench_codecs import synchronize
from common import DEFAULT_MODEL, collect_vectors, load_model, wikitext_token_chunks

SPARSITIES = (4, 8, 16)


def stored_error(vectors, indices, dictionary, sparsity):
    """Relative error of the stored code: refit prefix, 12-bit values."""
    prefix = indices[:, :sparsity]
    codes, scale = lexico.quantize_12bit(lexico.refit(vectors, prefix, dictionary))
    rebuilt = lexico.reconstruct(prefix, codes.float() * scale.float(), dictionary)
    return float((vectors - rebuilt).square().sum() / vectors.square().sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dictionary", default="checkpoints/lexico_dict.pt")
    parser.add_argument("--chunks", type=int, default=2)
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model)
    keys, values = collect_vectors(model, wikitext_token_chunks(tokenizer, "test", 512, args.chunks), device, 1)
    del model
    codec = lexico.LexicoCodec(args.dictionary, device)
    smax = max(SPARSITIES)

    def timed(fn):
        synchronize(device)
        started = time.perf_counter()
        result = fn()
        synchronize(device)
        return result, time.perf_counter() - started

    totals = {"same": 0.0, "time_old": 0.0, "time_new": 0.0}
    errors = {(name, kind, s): [] for name in ("old", "new") for kind in ("key", "value") for s in SPARSITIES}
    streams = 0
    for layer in range(codec.layers):
        for kind, tensors in (("key", keys), ("value", values)):
            vectors = tensors[layer].flatten(0, 1).to(device)
            dictionary = codec.dictionary(layer, kind)
            lexico.omp(vectors[:64], dictionary, smax)  # Warm-up.
            (old, _), old_time = timed(lambda: lexico.omp_reference(vectors, dictionary, smax))
            (new, _), new_time = timed(lambda: lexico.omp(vectors, dictionary, smax))
            totals["same"] += float((old == new).float().mean())
            totals["time_old"] += old_time
            totals["time_new"] += new_time
            streams += 1
            for sparsity in SPARSITIES:
                errors["old", kind, sparsity].append(stored_error(vectors, old, dictionary, sparsity))
                errors["new", kind, sparsity].append(stored_error(vectors, new, dictionary, sparsity))

    vectors_per_stream = keys[0].shape[0] * keys[0].shape[1]
    print(f"{streams} layer/streams x {vectors_per_stream} held-out vectors, OMP to {smax} atoms")
    print(f"atoms chosen identically (same slot): {totals['same'] / streams:.1%}")
    print(f"encode time, all streams: old {totals['time_old'] * 1000:.0f} ms, new {totals['time_new'] * 1000:.0f} ms "
          f"({totals['time_old'] / totals['time_new']:.2f}x)")
    print(f"{'atoms':>5} | {'key err old':>11} {'key err new':>11} | {'val err old':>11} {'val err new':>11}")
    for sparsity in SPARSITIES:
        mean = lambda name, kind: sum(errors[name, kind, sparsity]) / codec.layers
        print(f"{sparsity:>5} | {mean('old', 'key'):>11.4f} {mean('new', 'key'):>11.4f} | "
              f"{mean('old', 'value'):>11.4f} {mean('new', 'value'):>11.4f}")


if __name__ == "__main__":
    main()
