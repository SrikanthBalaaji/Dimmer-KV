"""Memory, time and accuracy of every method on the same contexts.

For each context length:
- accuracy: needle test (question after compression) and perplexity, via
  ``run_needles`` / ``run_perplexity``;
- memory: counted bytes for every method; for the kvtc family also the real
  serialized size (``kvtc_storage``, bit-packed codes plus per-token rung ids)
  and the size after DEFLATE (zlib level 6), as kvtc does; FP16 with DEFLATE
  is the lossless reference;
- time (CUDA-synced): compression, DEFLATE, and restore (inflate + decode
  into cache tensors), compared with re-running prefill from scratch.
The restored cache is checked against the in-memory compressed one.
"""

from __future__ import annotations

import argparse
import csv
import time
import zlib
from pathlib import Path

import torch

import dimmer
import kvtc_storage
from bench_codecs import synchronize
from common import DEFAULT_MODEL, kv_config, load_model
from eval_perplexity import run_perplexity
from methods import Methods
from test_generation import build_cases, run_needles

FIELDS = ("context", "method", "budget", "counted_ratio", "stored_ratio", "deflate_ratio", "stored_kib_per_1k_tokens",
          "deflate_kib_per_1k_tokens", "compress_ms", "deflate_ms", "restore_ms", "prefill_ms", "needles_correct",
          "needles_n", "answer_p", "ppl", "restore_max_abs_diff")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--basis", default="checkpoints/pca_basis.pt")
    parser.add_argument("--dictionary", default="checkpoints/lexico_dict.pt")
    parser.add_argument("--kvtc", default="checkpoints/kvtc.pt")
    parser.add_argument("--contexts", type=int, nargs="+", default=[1500, 3000])
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--ppl-chunks", type=int, default=4)
    parser.add_argument("--recent", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("results/memory_time_accuracy.csv"))
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model)
    dimmer.install(model)
    config = kv_config(model)
    methods = Methods(args.model, args.basis, args.dictionary, device, args.recent, model=model, kvtc=args.kvtc)
    codec = methods.codec("kvtc")

    def timed(fn):
        synchronize(device)
        started = time.perf_counter()
        result = fn()
        synchronize(device)
        return result, (time.perf_counter() - started) * 1000

    results = []
    for context in args.contexts:
        budgets = [0.05, 0.10] if context < 2500 else [0.03, 0.05, 0.10]
        kvtc_names = [f"{m}@{b:g}" for b in budgets for m in ("kvtc", "kvtcf", "dkvtc")]
        names = ["lex@0.1", "dkv:0.1:flat", "h2o:0.05", "h2o:0.1", "h2o:0.1:snap"] + kvtc_names
        print(f"=== context {context}: accuracy", flush=True)
        needle_rows = {r["method"]: r for r in run_needles(model, tokenizer, methods, names, "after", args.seeds, context)}
        ppl_rows = {r["method"]: r for r in run_perplexity(model, tokenizer, methods, names, context, 512, args.ppl_chunks)}

        print(f"=== context {context}: storage and restore", flush=True)
        storage = {name: {"stored": 0, "deflate": 0, "deflate_ms": 0.0, "restore_ms": 0.0, "diff": 0.0}
                   for name in ["full"] + kvtc_names}
        prefill_ms, total_tokens = 0.0, 0
        cases = build_cases(tokenizer, "after", args.seeds, context)
        for context_ids, _ in cases:
            with torch.inference_mode():
                (cache, scores), elapsed = timed(lambda: dimmer.prefill_with_scores(model, context_ids[None].to(device)))
                prefill_ms += elapsed
                keys = [layer.keys.clone() for layer in cache.layers]
                values = [layer.values.clone() for layer in cache.layers]
                del cache
                total_tokens += context_ids.shape[0]
                raw = b"".join(t.cpu().numpy().tobytes() for t in keys + values)
                packed, deflate_ms = timed(lambda: zlib.compress(raw, 6))
                entry = storage["full"]
                entry["stored"] += len(raw)
                entry["deflate"] += len(packed)
                entry["deflate_ms"] += deflate_ms
                _, restore_ms = timed(lambda: [torch.frombuffer(bytearray(zlib.decompress(packed)), dtype=torch.float16)
                                               .to(device)])
                entry["restore_ms"] += restore_ms
                for name in kvtc_names:
                    compressed = methods.apply(name, keys, values, scores)
                    data = kvtc_storage.encode_bytes(codec, keys, values, codec.last_choice, args.recent)
                    packed, deflate_ms = timed(lambda: zlib.compress(data, 6))
                    (restored_keys, restored_values), restore_ms = timed(
                        lambda: kvtc_storage.decode_bytes(codec, zlib.decompress(packed))
                    )
                    diff = max(float((a.float() - b.float()).abs().max())
                               for a, b in zip(restored_keys + restored_values, compressed.keys + compressed.values))
                    entry = storage[name]
                    entry["stored"] += len(data)
                    entry["deflate"] += len(packed)
                    entry["deflate_ms"] += deflate_ms
                    entry["restore_ms"] += restore_ms
                    entry["diff"] = max(entry["diff"], diff)

        fp16_total = total_tokens * config.fp16_bytes_per_token()
        for name in ["full"] + names:
            needle, ppl = needle_rows[name], ppl_rows[name]
            stored = storage.get(name)
            row = {
                "context": round(total_tokens / len(cases)), "method": name, "budget": needle["budget"],
                "counted_ratio": needle["bytes_ratio"], "needles_correct": needle["correct"],
                "needles_n": needle["n"], "answer_p": needle["answer_p"], "ppl": ppl["ppl"],
                "compress_ms": needle["compress_ms"], "prefill_ms": round(prefill_ms / len(cases), 1),
            }
            if stored:
                row.update({
                    "stored_ratio": round(stored["stored"] / fp16_total, 4),
                    "deflate_ratio": round(stored["deflate"] / fp16_total, 4),
                    "stored_kib_per_1k_tokens": round(stored["stored"] / total_tokens * 1000 / 1024, 1),
                    "deflate_kib_per_1k_tokens": round(stored["deflate"] / total_tokens * 1000 / 1024, 1),
                    "deflate_ms": round(stored["deflate_ms"] / len(cases), 1),
                    "restore_ms": round(stored["restore_ms"] / len(cases), 1),
                    "restore_max_abs_diff": round(stored["diff"], 5),
                })
            results.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n{'ctx':>5} {'method':>14} | {'counted':>7} {'stored':>7} {'deflate':>7} {'x':>6} | "
          f"{'comp ms':>7} {'defl ms':>7} {'restore':>7} {'prefill':>7} | {'needles':>7} {'ans p':>5} {'ppl':>6}")
    for r in results:
        ratio = r.get("deflate_ratio") or r["counted_ratio"]
        print(f"{r['context']:>5} {r['method']:>14} | {r['counted_ratio']:>7.3f} {r.get('stored_ratio', ''):>7} "
              f"{r.get('deflate_ratio', ''):>7} {1 / ratio:>6.1f} | {r['compress_ms']:>7.0f} {r.get('deflate_ms', ''):>7} "
              f"{r.get('restore_ms', ''):>7} {r['prefill_ms']:>7.0f} | {r['needles_correct']:>3}/{r['needles_n']:<3} "
              f"{r['answer_p']:>5.2f} {r['ppl']:>6.2f}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
