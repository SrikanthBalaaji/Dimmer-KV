"""Behavioral test: ask questions after the context cache is compressed.

Needle facts are placed in WikiText filler. Two modes (``--mode``):
- ``after`` (main): the context alone is prefilled and compressed; each
  question is then fed on top of the rebuilt cache, so the compressor cannot
  know what will be asked.
- ``prompt``: the question is at the end of the prefilled text (one prefill
  and compression per question), so scores can see it; only the chat
  closing tokens come after compression.
Every method uses the same prompts and greedy decoding, and the full-cache
run is the control. Method names are documented in ``methods.py``.

Metrics, over all seeds and needles:
- correct: the expected number appears in the greedy reply.
- agree: the greedy reply equals the full-cache reply.
- answer NLL: teacher-forced mean negative log-likelihood per answer token
  (lower is better; smooth, so it separates methods with equal accuracy).
- compress: encode + allocation + decode time for the context (CUDA-synced).
- gen: generation time per new token.
"""

from __future__ import annotations

import argparse
import math
import random
import time

import torch

import dimmer
from bench_codecs import synchronize
from common import DEFAULT_MODEL, build_cache, kv_config, load_model, wikitext_token_chunks
from methods import Methods, budget_of

NEEDLES = (
    ("The access code for the Orion locker is {}.", "What is the access code for the Orion locker?"),
    ("Dr. Patel's lab is in room {}.", "Which room is Dr. Patel's lab in?"),
    ("The blue train departs from platform {}.", "Which platform does the blue train depart from?"),
    ("The museum's founding year was {}.", "In which year was the museum founded?"),
    ("The vault combination ends with the digits {}.", "Which digits does the vault combination end with?"),
    ("Order number {} was shipped to Lisbon.", "Which order number was shipped to Lisbon?"),
)
SPLIT = "<<<SPLIT>>>"
DEFAULT_METHODS = (
    "lex@0.10", "h2o:0.10", "dkv:0.10:flat",
    "lex@0.15", "h2o:0.15", "h2o:0.15:snap", "dkv:0.15:flat", "dkv:0.15:snap",
    "lex@0.20", "h2o:0.20", "dkv:0.20:flat",
)


def build_prompt(filler: str, seed: int) -> tuple[str, list[tuple[str, str]]]:
    rng = random.Random(seed)
    sentences = [s.strip() + "." for s in filler.split(".") if s.strip()]
    qa = []
    for index, (fact, question) in enumerate(NEEDLES):
        answer = str(rng.randint(1000, 9999))
        position = (index + 1) * len(sentences) // (len(NEEDLES) + 1)
        sentences.insert(position + index, fact.format(answer))
        qa.append((question + " Answer with the number only.", answer))
    return " ".join(sentences), qa


def split_ids(tokenizer, context: str, question: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Token ids of the chat prompt, split into context part and question part."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"Read this document.\n\n{context}\n\n{SPLIT}{question}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    head, tail = rendered.split(SPLIT)
    return encode_text(tokenizer, head), encode_text(tokenizer, tail)


def split_ids_in_prompt(tokenizer, context: str, question: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Like ``split_ids`` but the question is part of the prefilled part."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"Read this document.\n\n{context}\n\n{question}{SPLIT}"}],
        tokenize=False,
        add_generation_prompt=True,
    )
    head, tail = rendered.split(SPLIT)
    return encode_text(tokenizer, head), encode_text(tokenizer, tail)


def encode_text(tokenizer, text: str) -> torch.Tensor:
    return tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]


def build_cases(tokenizer, mode: str, seeds: int, context_tokens: int):
    """A case is one prefill + compression with its questions asked on top."""
    cases = []
    for seed, filler_ids in enumerate(wikitext_token_chunks(tokenizer, "test", context_tokens, seeds)):
        context, qa = build_prompt(tokenizer.decode(filler_ids[1:]), seed)
        if mode == "after":
            context_ids = split_ids(tokenizer, context, qa[0][0])[0]
            questions = []
            for question, answer in qa:
                head_ids, tail_ids = split_ids(tokenizer, context, question)
                if not torch.equal(head_ids, context_ids):
                    raise RuntimeError("Context tokenization changed between questions.")
                questions.append((tail_ids, answer, encode_text(tokenizer, answer)))
            cases.append((context_ids, questions))
        else:
            for question, answer in qa:
                head_ids, tail_ids = split_ids_in_prompt(tokenizer, context, question)
                cases.append((head_ids, [(tail_ids, answer, encode_text(tokenizer, answer))]))
    return cases


def run_needles(model, tokenizer, methods: Methods, names, mode="after", seeds=3, context_tokens=1500,
                max_new_tokens=12) -> list[dict]:
    """Run the needle test; returns one result row per method (``full`` first)."""
    device = methods.device
    config = kv_config(model)
    names = ["full"] + [name for name in names if name != "full"]

    def timed(fn):
        synchronize(device)
        started = time.perf_counter()
        result = fn()
        synchronize(device)
        return result, time.perf_counter() - started

    def ask(keys, values, input_ids):
        cache = build_cache([k.clone() for k in keys], [v.clone() for v in values], model)
        return model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            past_key_values=cache,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    def answer_nll(keys, values, tail_ids, answer_ids, context_length):
        cache = build_cache([k.clone() for k in keys], [v.clone() for v in values], model)
        ids = torch.cat([tail_ids, answer_ids])[None].to(device)
        positions = torch.arange(context_length, context_length + ids.shape[1], device=device)
        logits = model(input_ids=ids, past_key_values=cache, position_ids=positions[None]).logits[0]
        # Logits at position i predict token i + 1; score only the answer tokens.
        predicted = logits[tail_ids.shape[0] - 1 : -1].float()
        return float(torch.nn.functional.cross_entropy(predicted, answer_ids.to(device)))

    stats = {name: {"correct": 0, "agree": 0, "nll": 0.0, "compress": 0.0, "gen": 0.0, "new": 0, "bytes": 0.0,
                    "histogram": {}} for name in names}
    cases = build_cases(tokenizer, mode, seeds, context_tokens)
    total = sum(len(questions) for _, questions in cases)
    prefill_time = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for context_ids, questions in cases:
        context_length = context_ids.shape[0]
        full_bytes = context_length * config.fp16_bytes_per_token()
        with torch.inference_mode():
            (cache, scores), elapsed = timed(lambda: dimmer.prefill_with_scores(model, context_ids[None].to(device)))
            prefill_time += elapsed
            keys = [layer.keys.clone() for layer in cache.layers]
            values = [layer.values.clone() for layer in cache.layers]
            del cache

        full_replies = []
        with torch.inference_mode():
            ask(keys, values, torch.cat([context_ids, questions[0][0]])[None].to(device))  # Warm-up.
            # One method at a time, so only one compressed cache is held in memory.
            for name in names:
                compressed = methods.apply(name, keys, values, scores)
                dimmer.set_eviction(compressed.evicted)
                entry = stats[name]
                entry["compress"] += compressed.seconds
                entry["bytes"] += compressed.used_bytes / full_bytes / len(cases)
                for level, share in compressed.histogram.items():
                    entry["histogram"][level] = entry["histogram"].get(level, 0.0) + share / len(cases)
                for index, (tail_ids, answer, answer_ids) in enumerate(questions):
                    input_ids = torch.cat([context_ids, tail_ids])[None].to(device)
                    generated, elapsed = timed(lambda: ask(compressed.keys, compressed.values, input_ids))
                    new_tokens = generated[0, input_ids.shape[1] :]
                    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                    if name == "full":
                        full_replies.append(reply)
                    entry["correct"] += answer in reply
                    entry["agree"] += reply == full_replies[index]
                    entry["gen"] += elapsed
                    entry["new"] += new_tokens.shape[0]
                    entry["nll"] += answer_nll(compressed.keys, compressed.values, tail_ids, answer_ids,
                                               context_length)
            dimmer.set_eviction(None)

    peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
    rows = []
    for name, entry in stats.items():
        nll = entry["nll"] / total
        rows.append({
            "task": f"needle_{mode}", "method": name, "budget": budget_of(name),
            "bytes_ratio": round(entry["bytes"], 4), "n": total, "correct": entry["correct"],
            "accuracy": round(entry["correct"] / total, 4), "agree": entry["agree"],
            "answer_nll": round(nll, 4), "answer_p": round(math.exp(-nll), 4), "ppl": "",
            "context_tokens": round(sum(c.shape[0] for c, _ in cases) / len(cases)),
            "prefill_ms": round(prefill_time / len(cases) * 1000, 1),
            "compress_ms": round(entry["compress"] / len(cases) * 1000, 1),
            "gen_ms_per_token": round(entry["gen"] / max(entry["new"], 1) * 1000, 2),
            "peak_gpu_mb": round(peak),
            "levels": "; ".join(f"{k} {v:.3f}" for k, v in entry["histogram"].items() if v > 0.0005),
        })
    return rows


def print_rows(rows: list[dict]) -> None:
    print(f"{'method':>16} {'bytes':>6} {'correct':>8} {'agree':>6} {'ans NLL':>8} {'ans p':>6} "
          f"{'compress ms':>11} {'gen ms/tok':>10}")
    for row in rows:
        print(
            f"{row['method']:>16} {row['bytes_ratio']:>6.3f} {row['correct']:>4}/{row['n']:<3} {row['agree']:>6} "
            f"{row['answer_nll']:>8.3f} {row['answer_p']:>6.3f} {row['compress_ms']:>11.0f} "
            f"{row['gen_ms_per_token']:>10.1f}"
        )
    for row in rows:
        if row["levels"]:
            print(f"{row['method']:>16} levels: {row['levels']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--basis", default="checkpoints/pca_basis.pt")
    parser.add_argument("--dictionary", default="checkpoints/lexico_dict.pt")
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_METHODS), help="Method names (methods.py).")
    parser.add_argument("--context-tokens", type=int, default=1500)
    parser.add_argument("--seeds", type=int, default=3, help="Contexts (different filler and numbers).")
    parser.add_argument("--kvtc", default="checkpoints/kvtc.pt", help="fit_kvtc.py output.")
    parser.add_argument("--recent", type=int, default=32, help="Last context tokens kept exact.")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--mode", default="after", choices=("after", "prompt"))
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model)
    dimmer.install(model)
    methods = Methods(args.model, args.basis, args.dictionary, device, args.recent,
                      model=model, kvtc=args.kvtc)
    rows = run_needles(model, tokenizer, methods, args.configs, args.mode, args.seeds, args.context_tokens,
                       args.max_new_tokens)
    print(f"mode: {args.mode}, seeds: {args.seeds}, needles: {rows[0]['n']}, "
          f"context ~{rows[0]['context_tokens']} tokens, recent exact: {args.recent}, device: {device}")
    print_rows(rows)


if __name__ == "__main__":
    main()
