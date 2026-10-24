"""Behavioral test: ask questions after the context cache is compressed.

Needle facts are placed in WikiText filler. The context alone is prefilled,
its cache is compressed with uniform PCA, and each question is then fed on
top of the rebuilt cache. Every configuration uses the same prompts and greedy
decoding, and the full-cache run is the control.
"""

from __future__ import annotations

import argparse
import random

import torch

from common import (
    DEFAULT_MODEL,
    build_cache,
    cache_tensors,
    kv_config,
    load_model,
    wikitext_token_chunks,
)
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


def build_prompt(tokenizer, filler: str, seed: int) -> tuple[str, list[tuple[str, str]]]:
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
    encode = lambda text: tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    return encode(head), encode(tail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--basis", default="checkpoints/pca_basis.pt")
    parser.add_argument("--context-tokens", type=int, default=1500)
    parser.add_argument("--components", type=int, nargs="+", default=[8, 16, 32, 48])
    parser.add_argument("--bits", type=int, default=8, choices=(8, 16))
    parser.add_argument("--recent", type=int, default=0, help="Last context tokens kept exact.")
    parser.add_argument(
        "--streams", default="kv", choices=("kv", "k", "v"), help="Which streams to compress."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model)
    config = kv_config(model)
    codec = PCACodec(args.basis, device)
    if codec.model != args.model:
        raise RuntimeError(f"Basis was fit for {codec.model}; refit for {args.model}.")

    filler_ids = wikitext_token_chunks(tokenizer, "test", args.context_tokens, 1)[0][1:]
    context, qa = build_prompt(tokenizer, tokenizer.decode(filler_ids), args.seed)
    context_ids, _ = split_ids(tokenizer, context, qa[0][0])
    with torch.inference_mode():
        outputs = model(input_ids=context_ids[None].to(device), use_cache=True)
    keys, values = cache_tensors(outputs.past_key_values)
    keys = [k.clone() for k in keys]
    values = [v.clone() for v in values]
    tokens = context_ids.shape[0]
    full_bytes = tokens * config.fp16_bytes_per_token()
    print(f"context tokens: {tokens}, needles: {len(qa)}, recent exact: {args.recent}, bits: {args.bits}, streams: {args.streams}")

    configs: list[tuple[str, list, list, int]] = [("full", keys, values, full_bytes)]
    for components in args.components:
        new_keys, new_values, used = codec.compress_cache(
            keys, values, components, args.bits, exact_recent=args.recent
        )
        # A stream left exact counts its full FP16 bytes in place of its codes.
        if args.streams == "k":
            new_values, used = values, used // 2 + full_bytes // 2
        elif args.streams == "v":
            new_keys, used = keys, used // 2 + full_bytes // 2
        configs.append((f"pca k={components}", new_keys, new_values, used))

    for name, cfg_keys, cfg_values, used in configs:
        correct = 0
        answers = []
        for question, answer in qa:
            head_ids, tail_ids = split_ids(tokenizer, context, question)
            if not torch.equal(head_ids, context_ids):
                raise RuntimeError("Context tokenization changed between questions.")
            input_ids = torch.cat([head_ids, tail_ids])[None].to(device)
            cache = build_cache(
                [k.clone() for k in cfg_keys], [v.clone() for v in cfg_values], model
            )
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    past_key_values=cache,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            reply = tokenizer.decode(generated[0, input_ids.shape[1] :], skip_special_tokens=True).strip()
            correct += answer in reply
            answers.append(f"{answer}->{reply[:12]!r}")
        print(f"{name:>10}  bytes {used / full_bytes:6.3f}  correct {correct}/{len(qa)}  {' '.join(answers)}")


if __name__ == "__main__":
    main()
