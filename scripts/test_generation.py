"""Behavioral test: ask questions after the context cache is compressed.

Needle facts are placed in WikiText filler. The context alone is prefilled,
its cache is compressed, and each question is then fed on top of the rebuilt
cache. Every configuration uses the same prompts and greedy decoding, and the
full-cache run is the control.

A configuration is ``CODEC`` for both streams or ``KEYCODEC/VALUECODEC``,
where a codec is ``pca<k>`` (INT8 PCA, k components), ``lex<s>`` (Lexico,
s atoms) or ``fp16`` (exact). Example: ``pca16/lex8``.

Metrics, over all seeds and needles:
- correct: the expected number appears in the greedy reply.
- agree: the greedy reply equals the full-cache reply.
- answer NLL: teacher-forced mean negative log-likelihood per answer token
  (lower is better; smooth, so it separates configs with equal accuracy).
- compress: encode + decode time for the context (CUDA-synchronized).
- gen: generation time per new token.
"""

from __future__ import annotations

import argparse
import math
import random
import time

import torch

from bench_codecs import synchronize
from common import (
    DEFAULT_MODEL,
    build_cache,
    cache_tensors,
    kv_config,
    load_model,
    wikitext_token_chunks,
)
from lexico import LexicoCodec
from pca_codec import PCACodec

NEEDLES = (
    ("The access code for the Orion locker is {}.", "What is the access code for the Orion locker?"),
    ("Dr. Patel's lab is in room {}.", "Which room is Dr. Patel's lab in?"),
    ("The blue train departs from platform {}.", "Which platform does the blue train depart from?"),
    ("The museum's founding year was {}.", "In which year was the museum founded?"),
    ("The vault combination ends with the digits {}.", "Which digits does the vault combination end with?"),
    ("Order number {} was shipped to Lisbon.", "Which order number was shipped to Lisbon?"),
)
SPLIT = "<<<SPLIT>>>"
DEFAULT_CONFIGS = (
    # ~14% of FP16 KV.
    "pca16", "lex5", "pca16/lex5", "pca8/lex8",
    # ~20%.
    "lex8", "pca24/lex8", "pca16/lex11",
    # ~27%.
    "pca32", "lex11", "pca32/lex11", "pca16/lex16",
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


def encode_text(tokenizer, text: str) -> torch.Tensor:
    return tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]


def parse_codec(spec: str) -> tuple[str, int]:
    if spec == "fp16":
        return "fp16", 0
    for prefix in ("pca", "lex"):
        if spec.startswith(prefix) and spec[len(prefix) :].isdigit():
            return prefix, int(spec[len(prefix) :])
    raise ValueError(f"Unknown codec {spec!r}; use pca<k>, lex<s> or fp16.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--basis", default="checkpoints/pca_basis.pt")
    parser.add_argument("--dictionary", default="checkpoints/lexico_dict.pt")
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--context-tokens", type=int, default=1500)
    parser.add_argument("--seeds", type=int, default=3, help="Contexts (different filler and numbers).")
    parser.add_argument("--recent", type=int, default=0, help="Last context tokens kept exact.")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model)
    config = kv_config(model)
    configs = [(name, [parse_codec(part) for part in (name.split("/") * 2)[:2]]) for name in args.configs]
    codecs = {
        "pca": PCACodec(args.basis, device),
        "lex": LexicoCodec(args.dictionary, device),
    }
    for codec in codecs.values():
        if codec.model != args.model:
            raise RuntimeError(f"Codec was fit for {codec.model}; refit for {args.model}.")

    def timed(fn):
        synchronize(device)
        started = time.perf_counter()
        result = fn()
        synchronize(device)
        return result, time.perf_counter() - started

    def compress(tensors, kind, codec, size):
        if codec == "fp16":
            return tensors, len(tensors) * tensors[0].numel() * 2
        if codec == "pca":
            return codecs["pca"].compress_stream(tensors, kind, size, 8, exact_recent=args.recent)
        return codecs["lex"].compress_stream(tensors, kind, size, exact_recent=args.recent)

    def ask(keys, values, input_ids):
        cache = build_cache([k.clone() for k in keys], [v.clone() for v in values], model)
        return model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            past_key_values=cache,
            max_new_tokens=args.max_new_tokens,
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

    stats = {name: {"correct": 0, "agree": 0, "nll": 0.0, "compress": 0.0, "gen": 0.0, "new": 0, "bytes": 0.0}
             for name in ["full"] + [n for n, _ in configs]}
    contexts = wikitext_token_chunks(tokenizer, "test", args.context_tokens, args.seeds)
    total = 0
    for seed, filler_ids in enumerate(contexts):
        context, qa = build_prompt(tokenizer.decode(filler_ids[1:]), seed)
        context_ids, _ = split_ids(tokenizer, context, qa[0][0])
        context_length = context_ids.shape[0]
        with torch.inference_mode():
            outputs = model(input_ids=context_ids[None].to(device), use_cache=True)
        keys, values = cache_tensors(outputs.past_key_values)
        keys = [k.clone() for k in keys]
        values = [v.clone() for v in values]
        full_bytes = context_length * config.fp16_bytes_per_token()

        variants = [("full", keys, values, 0.0, full_bytes)]
        with torch.inference_mode():
            for name, ((key_codec, key_size), (value_codec, value_size)) in configs:
                (new_keys, key_bytes), key_time = timed(lambda: compress(keys, "key", key_codec, key_size))
                (new_values, value_bytes), value_time = timed(
                    lambda: compress(values, "value", value_codec, value_size)
                )
                variants.append((name, new_keys, new_values, key_time + value_time, key_bytes + value_bytes))

        questions = []
        for question, answer in qa:
            head_ids, tail_ids = split_ids(tokenizer, context, question)
            if not torch.equal(head_ids, context_ids):
                raise RuntimeError("Context tokenization changed between questions.")
            questions.append((tail_ids, answer, encode_text(tokenizer, answer)))
        total += len(questions)

        full_replies = []
        with torch.inference_mode():
            ask(keys, values, torch.cat([context_ids, questions[0][0]])[None].to(device))  # Warm-up.
            for name, cfg_keys, cfg_values, compress_time, used in variants:
                entry = stats[name]
                entry["compress"] += compress_time
                entry["bytes"] += used / full_bytes / len(contexts)
                for index, (tail_ids, answer, answer_ids) in enumerate(questions):
                    input_ids = torch.cat([context_ids, tail_ids])[None].to(device)
                    generated, elapsed = timed(lambda: ask(cfg_keys, cfg_values, input_ids))
                    new_tokens = generated[0, input_ids.shape[1] :]
                    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
                    if name == "full":
                        full_replies.append(reply)
                    entry["correct"] += answer in reply
                    entry["agree"] += reply == full_replies[index]
                    entry["gen"] += elapsed
                    entry["new"] += new_tokens.shape[0]
                    entry["nll"] += answer_nll(cfg_keys, cfg_values, tail_ids, answer_ids, context_length)

    print(f"seeds: {len(contexts)}, needles: {total}, context ~{args.context_tokens} tokens, "
          f"recent exact: {args.recent}, device: {device}")
    print(f"{'config':>12} {'bytes':>6} {'correct':>8} {'agree':>6} {'ans NLL':>8} {'ans p':>6} "
          f"{'compress ms':>11} {'gen ms/tok':>10}")
    for name, entry in stats.items():
        nll = entry["nll"] / total
        print(
            f"{name:>12} {entry['bytes']:>6.3f} {entry['correct']:>4}/{total:<3} {entry['agree']:>6} "
            f"{nll:>8.3f} {math.exp(-nll):>6.3f} {entry['compress'] / len(contexts) * 1000:>11.0f} "
            f"{entry['gen'] / max(entry['new'], 1) * 1000:>10.1f}"
        )


if __name__ == "__main__":
    main()
