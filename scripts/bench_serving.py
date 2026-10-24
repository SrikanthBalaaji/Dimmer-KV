"""End-to-end serving benchmark in absolute units (MB, ms, tokens/s).

Scenario: a chat assistant keeps many conversations' KV caches in host RAM
between turns; a follow-up question arrives for one of them. Per method:

- stored MB per conversation and how many conversations fit in ``--ram-gib``;
- time to first token (TTFT): get the cache onto the GPU, then run the
  question and pick the first answer token;
- decode speed (tokens/s) generating ``--gen-tokens`` tokens afterwards;
- peak extra GPU memory while answering (above the static model + codecs);
- accuracy on the same contexts (needles, question after compression).

Ways to get the cache back:
- ``full (GPU-resident)``: cache never left the GPU (best case, no saving);
- ``full (RAM, FP16)``: copy the uncompressed cache from host RAM;
- ``recompute``: store nothing, re-run prefill on the conversation text;
- ``h2o``: copy the kept FP16 tokens plus a keep-mask (transfer timed; the
  Level 1 cache keeps evicted slots masked);
- kvtc family: inflate the DEFLATE bytes and decode them on the GPU.
All answering uses the same eager attention (needed for H2O's eviction
masks), so decode speeds compare like with like; the recompute baseline
prefills in chunks with PyTorch SDPA, its fastest path here. Prefill for
compression is chunked so long contexts fit.
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
from common import DEFAULT_MODEL, build_cache, kv_config, load_model
from methods import Methods
from test_generation import build_cases, run_needles

MIB = 2**20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--basis", default="checkpoints/pca_basis.pt")
    parser.add_argument("--dictionary", default="checkpoints/lexico_dict.pt")
    parser.add_argument("--kvtc", default="checkpoints/kvtc.pt")
    parser.add_argument("--context-tokens", type=int, default=3000)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--gen-tokens", type=int, default=32)
    parser.add_argument("--ram-gib", type=float, default=8.0)
    parser.add_argument("--prefill-chunk", type=int, default=1024, help="Chunk size for the recompute baseline.")
    parser.add_argument("--recent", type=int, default=32)
    parser.add_argument("--methods", nargs="+",
                        default=["h2o:0.05", "h2o:0.1", "kvtcf@0.03", "dkvtc@0.03", "kvtcf@0.05", "dkvtc@0.05"])
    parser.add_argument("--output", type=Path, default=Path("results/serving.csv"))
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

    print("=== accuracy", flush=True)
    accuracy = {r["method"]: r for r in run_needles(model, tokenizer, methods, args.methods, "after", args.seeds,
                                                    args.context_tokens)}

    print("=== serving", flush=True)
    ways = ["full (GPU-resident)", "full (RAM, FP16)", "recompute"] + args.methods
    stats = {w: {"bytes": 0, "ttft": 0.0, "restore": 0.0, "decode": 0.0, "peak": 0.0, "questions": 0} for w in ways}
    cases = build_cases(tokenizer, "after", args.seeds, args.context_tokens)
    tokens_total = 0
    for context_ids, questions in cases:
        length = context_ids.shape[0]
        tokens_total += length
        with torch.inference_mode():
            cache, scores = dimmer.prefill_with_scores(model, context_ids[None].to(device))
            keys = [layer.keys.clone() for layer in cache.layers]
            values = [layer.values.clone() for layer in cache.layers]
            del cache
            stored = {}  # way -> (bytes, restore function returning (keys, values, evicted))
            host = [t.cpu() for t in keys + values]
            try:
                host = [t.pin_memory() for t in host]  # Fast host-to-GPU copies.
            except RuntimeError:
                pass  # Windows limits pinned memory; fall back to pageable RAM (slower copy).
            stored["full (GPU-resident)"] = (sum(t.numel() * 2 for t in host),
                                             lambda: ([k.clone() for k in keys], [v.clone() for v in values], None))
            stored["full (RAM, FP16)"] = (sum(t.numel() * 2 for t in host), lambda: (
                [t.to(device, non_blocking=True) for t in host[: config.layers]],
                [t.to(device, non_blocking=True) for t in host[config.layers :]], None))

            def recompute():
                # Chunked prefill with PyTorch SDPA (the fastest prefill here), as a
                # server would do for long prompts; then back to the shared attention.
                model.set_attn_implementation("sdpa")
                ids_all, out = context_ids[None].to(device), None
                for start in range(0, length, args.prefill_chunk):
                    end = min(length, start + args.prefill_chunk)
                    out = model(input_ids=ids_all[:, start:end], past_key_values=out, use_cache=True,
                                position_ids=torch.arange(start, end, device=device)[None]).past_key_values
                model.set_attn_implementation(dimmer.ATTENTION_NAME)
                return [l.keys for l in out.layers], [l.values for l in out.layers], None
            stored["recompute"] = (length * 4, recompute)  # Only the token ids are kept.
            for name in args.methods:
                compressed = methods.apply(name, keys, values, scores)
                if name.startswith("h2o"):
                    mask_bytes = config.layers * config.kv_heads * length // 8
                    payload = torch.empty(compressed.used_bytes + mask_bytes, dtype=torch.uint8)

                    def restore_h2o(c=compressed, p=payload):
                        p.to(device, non_blocking=True)  # Transfer of kept tokens + mask, timed.
                        return c.keys, c.values, c.evicted
                    stored[name] = (payload.numel(), restore_h2o)
                else:
                    data = zlib.compress(kvtc_storage.encode_bytes(codec, keys, values, codec.last_choice,
                                                                    args.recent), 6)
                    stored[name] = (len(data), lambda d=data: (*kvtc_storage.decode_bytes(codec, zlib.decompress(d)),
                                                              None))

            for way in ways:
                size, restore = stored[way]
                entry = stats[way]
                entry["bytes"] += size
                for tail_ids, _, _ in questions:
                    ids = tail_ids[None].to(device)
                    positions = torch.arange(length, length + ids.shape[1], device=device)[None]
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    static = torch.cuda.memory_allocated()

                    def first_token():
                        restored_keys, restored_values, evicted = restore()
                        dimmer.set_eviction(evicted)
                        rebuilt = build_cache(restored_keys, restored_values, model)
                        logits = model(input_ids=ids, past_key_values=rebuilt, position_ids=positions).logits
                        return rebuilt, logits[0, -1].argmax()

                    (rebuilt, token), ttft = timed(first_token)
                    _, restore_ms = timed(lambda: restore())

                    def decode():
                        current = token.view(1, 1)
                        for step in range(args.gen_tokens):
                            position = torch.tensor([[length + ids.shape[1] + step]], device=device)
                            logits = model(input_ids=current, past_key_values=rebuilt, position_ids=position).logits
                            current = logits[:, -1:].argmax(dim=-1)
                    _, decode_ms = timed(decode)
                    dimmer.set_eviction(None)
                    entry["ttft"] += ttft
                    entry["restore"] += restore_ms
                    entry["decode"] += decode_ms
                    entry["peak"] = max(entry["peak"], (torch.cuda.max_memory_allocated() - static) / MIB)
                    entry["questions"] += 1
                    del rebuilt

    ram = args.ram_gib * 2**30
    rows = []
    for way in ways:
        entry = stats[way]
        per_conv = entry["bytes"] / len(cases)
        acc_key = "full" if way in ("full (GPU-resident)", "full (RAM, FP16)", "recompute") else way
        acc = accuracy[acc_key]
        rows.append({
            "method": way, "context_tokens": round(tokens_total / len(cases)),
            "stored_mib": round(per_conv / MIB, 2), "conversations_in_ram": int(ram // per_conv),
            "restore_ms": round(entry["restore"] / entry["questions"], 1),
            "ttft_ms": round(entry["ttft"] / entry["questions"], 1),
            "decode_tokens_per_s": round(args.gen_tokens * entry["questions"] / (entry["decode"] / 1000), 1),
            "peak_extra_gpu_mib": round(entry["peak"]),
            "needles": f"{acc['correct']}/{acc['n']}", "answer_p": acc["answer_p"],
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\ncontext ~{rows[0]['context_tokens']} tokens, {stats[ways[0]]['questions']} questions, "
          f"RAM budget {args.ram_gib:g} GiB, {args.gen_tokens} generated tokens per answer")
    print(f"{'method':>20} | {'stored MiB':>10} {'convs/RAM':>9} | {'restore ms':>10} {'TTFT ms':>8} "
          f"{'tok/s':>6} {'peak+MiB':>8} | {'needles':>7} {'ans p':>5}")
    for r in rows:
        print(f"{r['method']:>20} | {r['stored_mib']:>10} {r['conversations_in_ram']:>9} | {r['restore_ms']:>10} "
              f"{r['ttft_ms']:>8} {r['decode_tokens_per_s']:>6} {r['peak_extra_gpu_mib']:>8} | "
              f"{r['needles']:>7} {r['answer_p']:>5}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
