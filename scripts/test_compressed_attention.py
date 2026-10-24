"""Answer from the compressed cache directly vs from the rebuilt FP16 cache.

For each needle question (question after compression) and each kvtc-family
method: both paths answer greedily from the same compressed cache.

- "rebuilt" (Level 1): decode everything back to an FP16 cache, then attend.
- "direct" (compressed_attention): keys decoded one layer at a time, values
  never decoded, exact tokens in a small cache.

Reports agreement (same reply, max first-step logit difference), GPU memory
held per conversation (FP16 cache vs coefficients + exact tokens), extra
working memory while answering, and tokens/s over a fixed-length answer.
"""

from __future__ import annotations

import argparse
import time

import torch

import dimmer
from bench_codecs import synchronize
from common import DEFAULT_MODEL, build_cache, load_model
from compressed_attention import CompressedContext, greedy_answer
from methods import Methods
from test_generation import build_cases

MIB = 2**20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--context-tokens", type=int, default=8000)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--methods", nargs="+", default=["kvtcf@0.02", "dkvtc@0.02", "dkvtc@0.05"])
    parser.add_argument("--speed-tokens", type=int, default=32)
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model)
    dimmer.install(model)
    methods = Methods(args.model, "checkpoints/pca_basis.pt", "checkpoints/lexico_dict.pt", device, 32,
                      model=model, kvtc="checkpoints/kvtc.pt")
    codec = methods.codec("kvtc")

    def timed(fn):
        synchronize(device)
        started = time.perf_counter()
        result = fn()
        synchronize(device)
        return result, time.perf_counter() - started

    stats = {name: {"n": 0, "same": 0, "correct_rebuilt": 0, "correct_direct": 0, "max_diff": 0.0,
                    "held_rebuilt": 0.0, "held_direct": 0.0, "basis": 0.0, "peak_rebuilt": 0.0,
                    "peak_direct": 0.0, "tps_rebuilt": [], "tps_direct": []} for name in args.methods}
    cases = build_cases(tokenizer, "after", args.seeds, args.context_tokens)
    for context_ids, questions in cases:
        length = context_ids.shape[0]
        with torch.inference_mode():
            cache, scores = dimmer.prefill_with_scores(model, context_ids[None].to(device))
            keys = [layer.keys.clone() for layer in cache.layers]
            values = [layer.values.clone() for layer in cache.layers]
            del cache
            for name in args.methods:
                compressed = methods.apply(name, keys, values, scores)
                context = CompressedContext(codec, keys, values, model)
                entry = stats[name]
                entry["held_rebuilt"] = sum(t.numel() * 2 for t in compressed.keys + compressed.values) / MIB
                entry["held_direct"] = (context.code_bytes() + context.exact_bytes()) / MIB
                entry["basis"] = context.basis_bytes() / MIB
                for tail_ids, answer, _ in questions:
                    ids = tail_ids[None].to(device)
                    # Rebuilt FP16 cache (current pipeline).
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    base = torch.cuda.memory_allocated()
                    full = build_cache([k.clone() for k in compressed.keys], [v.clone() for v in compressed.values],
                                       model)
                    reply_r, logits_r = greedy_answer(model, tokenizer, None, full, ids, length,
                                                      evicted=compressed.evicted)
                    entry["peak_rebuilt"] = max(entry["peak_rebuilt"], (torch.cuda.max_memory_allocated() - base) / MIB)
                    del full
                    # Direct attention over the compressed tokens.
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    base = torch.cuda.memory_allocated()
                    # A fresh copy of the small exact cache: answering appends the question to it.
                    exact = build_cache([l.keys.clone() for l in context.exact_cache.layers],
                                        [l.values.clone() for l in context.exact_cache.layers], model)
                    reply_d, logits_d = greedy_answer(model, tokenizer, context, exact, ids, length)
                    entry["peak_direct"] = max(entry["peak_direct"], (torch.cuda.max_memory_allocated() - base) / MIB)
                    del exact
                    entry["n"] += 1
                    entry["same"] += reply_r == reply_d
                    entry["correct_rebuilt"] += answer in reply_r
                    entry["correct_direct"] += answer in reply_d
                    entry["max_diff"] = max(entry["max_diff"], float((logits_r - logits_d).abs().max()))
                # Speed: a fixed-length answer for the first question, both paths.
                tail_ids = questions[0][0][None].to(device)
                full = build_cache([k.clone() for k in compressed.keys], [v.clone() for v in compressed.values], model)
                _, seconds = timed(lambda: greedy_answer(model, tokenizer, None, full, tail_ids, length,
                                                         args.speed_tokens, compressed.evicted, stop_at_eos=False))
                entry["tps_rebuilt"].append(args.speed_tokens / seconds)
                del full
                exact = build_cache([l.keys.clone() for l in context.exact_cache.layers],
                                    [l.values.clone() for l in context.exact_cache.layers], model)
                _, seconds = timed(lambda: greedy_answer(model, tokenizer, context, exact, tail_ids, length,
                                                         args.speed_tokens, stop_at_eos=False))
                entry["tps_direct"].append(args.speed_tokens / seconds)
                del exact, context, compressed

    print(f"context ~{round(sum(c.shape[0] for c, _ in cases) / len(cases))} tokens, "
          f"{stats[args.methods[0]]['n']} questions per method (question after compression)")
    print(f"{'method':>12} | {'same reply':>10} {'max logit diff':>14} | {'correct reb/dir':>15} | "
          f"{'held MiB reb':>12} {'held MiB dir':>12} {'basis MiB':>9} | {'peak+ reb':>9} {'peak+ dir':>9} | "
          f"{'tok/s reb':>9} {'tok/s dir':>9}")
    for name, e in stats.items():
        print(f"{name:>12} | {e['same']:>4}/{e['n']:<5} {e['max_diff']:>14.3f} | "
              f"{e['correct_rebuilt']:>6}/{e['correct_direct']:<8} | {e['held_rebuilt']:>12.1f} "
              f"{e['held_direct']:>12.1f} {e['basis']:>9.1f} | {e['peak_rebuilt']:>9.0f} {e['peak_direct']:>9.0f} | "
              f"{sum(e['tps_rebuilt']) / len(e['tps_rebuilt']):>9.1f} {sum(e['tps_direct']) / len(e['tps_direct']):>9.1f}")


if __name__ == "__main__":
    main()
