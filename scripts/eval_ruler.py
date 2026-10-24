"""RULER-style long-context tasks under KV compression (as used in the kvtc paper).

Tasks:
- ``vt`` (variable tracking, multi-hop): a chain ``VAR A = 12345``, ``VAR B = VAR A``,
  … hidden in repeated filler; the model must list every variable holding the
  value. Score = share of the chain's variables named in the reply.
- ``kv`` (key-value retrieval, "lost in the middle"): a JSON object of random
  UUID pairs; the model must return the value of one key. Score = exact match.

Both run with the question after compression (``after``, the main mode) or in
the prompt (``prompt``). Every method sees the same documents and greedy
decoding; the full cache is the control. Method names: ``methods.py``.
"""

from __future__ import annotations

import argparse
import json
import random
import string
import time
import uuid

import torch

import dimmer
from bench_codecs import synchronize
from common import DEFAULT_MODEL, build_cache, load_model
from methods import Methods, budget_of

SPLIT = "<<<SPLIT>>>"
NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "


def build_vt(tokenizer, rng: random.Random, target_tokens: int, hops: int):
    """Variable-tracking document, question and the expected variable names."""
    names = ["".join(rng.choices(string.ascii_uppercase, k=5)) for _ in range(hops + 1)]
    value = str(rng.randint(10000, 99999))
    lines = [f"VAR {names[0]} = {value}."] + [f"VAR {names[i]} = VAR {names[i - 1]}." for i in range(1, hops + 1)]
    noise_tokens = len(tokenizer(NOISE, add_special_tokens=False)["input_ids"])
    repeats = max(hops + 2, target_tokens // noise_tokens)
    slots = sorted(rng.sample(range(repeats), len(lines)))
    parts = []
    line_index = 0
    for i in range(repeats):
        if line_index < len(lines) and i == slots[line_index]:
            parts.append(lines[line_index] + " ")
            line_index += 1
        parts.append(NOISE)
    document = ("Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n"
                + "".join(parts))
    question = f"Question: Find all variables that are assigned the value {value} in the text above."
    prefix = (f"Answer: According to the chain(s) of variable assignment in the text above, {hops + 1} "
              f"variables are assigned the value {value}, they are:")
    return document, question, prefix, names


def build_kv(tokenizer, rng: random.Random, target_tokens: int):
    """Key-value retrieval document (JSON of UUIDs), question and the expected value."""
    def make_uuid():
        return str(uuid.UUID(int=rng.getrandbits(128)))
    pair_tokens = len(tokenizer(json.dumps({make_uuid(): make_uuid()}), add_special_tokens=False)["input_ids"])
    pairs = {make_uuid(): make_uuid() for _ in range(max(4, target_tokens // pair_tokens))}
    key = rng.choice(list(pairs))
    document = ("Extract the value corresponding to the specified key in the JSON object below.\n\nJSON data:\n"
                + json.dumps(pairs, indent=1))
    question = f'Key: "{key}"\nWhat is the value associated with this key?'
    prefix = f'The value associated with the key "{key}" is "'
    return document, question, prefix, [pairs[key]]


def split_prompt(tokenizer, document: str, question: str, prefix: str, in_prompt: bool):
    """Token ids of the prefilled part and of the part fed after compression."""
    if in_prompt:
        content = f"{document}\n\n{question}{SPLIT}"
    else:
        content = f"{document}\n\n{SPLIT}{question}"
    rendered = tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                             add_generation_prompt=True) + prefix
    head, tail = rendered.split(SPLIT)
    encode = lambda text: tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    return encode(head), encode(tail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--task", choices=("vt", "kv"), default="vt")
    parser.add_argument("--mode", choices=("after", "prompt"), default="after")
    parser.add_argument("--context-tokens", type=int, default=4000)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--hops", type=int, default=4, help="Variable-tracking chain length minus one.")
    parser.add_argument("--configs", nargs="+", default=["full"])
    parser.add_argument("--recent", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model)
    dimmer.install(model)
    methods = Methods(args.model, "checkpoints/pca_basis.pt", "checkpoints/lexico_dict.pt", device, args.recent,
                      model=model, kvtc="checkpoints/kvtc.pt", progressive="checkpoints/progressive.pt")
    names = ["full"] + [n for n in args.configs if n != "full"]
    stats = {n: {"score": 0.0, "bytes": 0.0, "compress": 0.0, "n": 0, "skipped": ""} for n in names}
    rng = random.Random(args.seed)
    tokens_seen = 0
    for _ in range(args.samples):
        if args.task == "vt":
            document, question, prefix, expected = build_vt(tokenizer, rng, args.context_tokens, args.hops)
        else:
            document, question, prefix, expected = build_kv(tokenizer, rng, args.context_tokens)
        head_ids, tail_ids = split_prompt(tokenizer, document, question, prefix, args.mode == "prompt")
        tokens_seen += head_ids.shape[0]
        with torch.inference_mode():
            cache, scores = dimmer.prefill_with_scores(model, head_ids[None].to(device))
            keys = [layer.keys.clone() for layer in cache.layers]
            values = [layer.values.clone() for layer in cache.layers]
            del cache
            input_ids = torch.cat([head_ids, tail_ids])[None].to(device)
            for name in names:
                try:
                    compressed = methods.apply(name, keys, values, scores)
                except ValueError as error:
                    stats[name]["skipped"] = str(error)
                    continue
                q_keys, q_values = compressed.keys, compressed.values
                if compressed.refine is not None:
                    q_keys, q_values, _ = compressed.refine(tail_ids, head_ids.shape[0])
                dimmer.set_eviction(compressed.evicted)
                rebuilt = build_cache([k.clone() for k in q_keys], [v.clone() for v in q_values], model)
                generated = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                           past_key_values=rebuilt, max_new_tokens=args.max_new_tokens,
                                           do_sample=False, pad_token_id=tokenizer.eos_token_id)
                dimmer.set_eviction(None)
                reply = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True)
                entry = stats[name]
                entry["score"] += sum(e in reply for e in expected) / len(expected)
                entry["bytes"] += compressed.used_bytes / (head_ids.shape[0] * methods.codec("kvtc").exact_bytes * 2)
                entry["compress"] += compressed.seconds
                entry["n"] += 1

    print(f"task {args.task} ({'hops ' + str(args.hops) if args.task == 'vt' else 'UUID pairs'}), "
          f"mode {args.mode}, {args.samples} samples, ~{tokens_seen // args.samples} tokens")
    print(f"{'method':>20} {'bytes':>6} {'score':>7} {'compress ms':>11}")
    for name, entry in stats.items():
        if entry["n"] == 0:
            print(f"{name:>20} skipped: {entry['skipped']}")
            continue
        print(f"{name:>20} {entry['bytes'] / entry['n']:>6.3f} {entry['score'] / entry['n']:>7.3f} "
              f"{entry['compress'] / entry['n'] * 1000:>11.0f}")


if __name__ == "__main__":
    main()
