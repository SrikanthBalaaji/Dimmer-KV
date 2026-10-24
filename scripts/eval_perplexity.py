"""Perplexity of the continuation after the context cache is compressed.

Each WikiText-2 test chunk is split into a context (prefilled, then
compressed by each method) and a continuation scored teacher-forced on top
of the compressed cache. The first continuation token is not scored (its
prediction comes from the uncompressed prefill). Lower is better; ``full``
is the control.
"""

from __future__ import annotations

import argparse
import math
import time

import torch

import dimmer
from bench_codecs import synchronize
from common import DEFAULT_MODEL, build_cache, kv_config, load_model, wikitext_token_chunks
from methods import Methods, budget_of

DEFAULT_METHODS = (
    "lex@0.10", "h2o:0.10", "h2o:0.10:snap", "dkv:0.10:flat", "dkv:0.10:snap",
    "lex@0.15", "h2o:0.15", "h2o:0.15:snap", "dkv:0.15:flat", "dkv:0.15:snap",
    "lex@0.20", "h2o:0.20", "h2o:0.20:snap", "dkv:0.20:flat", "dkv:0.20:snap",
)


def run_perplexity(model, tokenizer, methods: Methods, names, context_tokens=1536, eval_tokens=512,
                   chunks=4) -> list[dict]:
    """Returns one result row per method (``full`` first)."""
    device = methods.device
    config = kv_config(model)
    names = ["full"] + [name for name in names if name != "full"]
    stats = {name: {"nll": 0.0, "count": 0, "compress": 0.0, "bytes": 0.0} for name in names}
    samples = wikitext_token_chunks(tokenizer, "test", context_tokens + eval_tokens, chunks)
    prefill_time = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for sample in samples:
        context_ids = sample[:context_tokens][None].to(device)
        continuation = sample[context_tokens:][None].to(device)
        positions = torch.arange(context_tokens, context_tokens + continuation.shape[1], device=device)[None]
        full_bytes = context_tokens * config.fp16_bytes_per_token()
        with torch.inference_mode():
            synchronize(device)
            started = time.perf_counter()
            cache, scores = dimmer.prefill_with_scores(model, context_ids)
            synchronize(device)
            prefill_time += time.perf_counter() - started
            keys = [layer.keys.clone() for layer in cache.layers]
            values = [layer.values.clone() for layer in cache.layers]
            del cache
            for name in names:
                compressed = methods.apply(name, keys, values, scores)
                dimmer.set_eviction(compressed.evicted)
                rebuilt = build_cache([k.clone() for k in compressed.keys],
                                      [v.clone() for v in compressed.values], model)
                logits = model(input_ids=continuation, past_key_values=rebuilt, position_ids=positions).logits[0]
                dimmer.set_eviction(None)
                nll = torch.nn.functional.cross_entropy(
                    logits[:-1].float(), continuation[0, 1:], reduction="sum"
                )
                entry = stats[name]
                entry["nll"] += float(nll)
                entry["count"] += continuation.shape[1] - 1
                entry["compress"] += compressed.seconds
                entry["bytes"] += compressed.used_bytes / full_bytes / len(samples)

    peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
    rows = []
    for name, entry in stats.items():
        nll = entry["nll"] / entry["count"]
        rows.append({
            "task": "perplexity", "method": name, "budget": budget_of(name),
            "bytes_ratio": round(entry["bytes"], 4), "n": entry["count"], "correct": "", "accuracy": "",
            "agree": "", "answer_nll": round(nll, 4), "answer_p": "", "ppl": round(math.exp(nll), 3),
            "context_tokens": context_tokens,
            "prefill_ms": round(prefill_time / len(samples) * 1000, 1),
            "compress_ms": round(entry["compress"] / len(samples) * 1000, 1),
            "gen_ms_per_token": "", "peak_gpu_mb": round(peak), "levels": "",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--basis", default="checkpoints/pca_basis.pt")
    parser.add_argument("--dictionary", default="checkpoints/lexico_dict.pt")
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_METHODS), help="Method names (methods.py).")
    parser.add_argument("--context-tokens", type=int, default=1536)
    parser.add_argument("--eval-tokens", type=int, default=512)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--kvtc", default="checkpoints/kvtc.pt", help="fit_kvtc.py output.")
    parser.add_argument("--recent", type=int, default=32)
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model)
    dimmer.install(model)
    methods = Methods(args.model, args.basis, args.dictionary, device, args.recent,
                      model=model, kvtc=args.kvtc)
    rows = run_perplexity(model, tokenizer, methods, args.configs, args.context_tokens, args.eval_tokens,
                          args.chunks)
    print(f"chunks: {args.chunks}, context {args.context_tokens} + eval {args.eval_tokens} tokens, "
          f"recent exact: {args.recent}")
    print(f"{'method':>16} {'bytes':>6} {'ppl':>8} {'compress ms':>11}")
    for row in rows:
        print(f"{row['method']:>16} {row['bytes_ratio']:>6.3f} {row['ppl']:>8.3f} {row['compress_ms']:>11.0f}")


if __name__ == "__main__":
    main()
