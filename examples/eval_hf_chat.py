"""Check the Hugging Face integration end to end: facts asked after the chat was compressed.

Turn 1 gives the model a long document with hidden facts and is stored
compressed. Each later question restores the history from the store (only the
question is prefilled) and is answered greedily; answers and the first answer
token's distribution are compared with the uncompressed cache.

    python examples/eval_hf_chat.py --model Qwen/Qwen2.5-0.5B-Instruct \
        --checkpoint checkpoints/qwen2.5-0.5b.pt --budgets 0.03 0.05 0.1
"""

from __future__ import annotations

import argparse
import random
import tempfile
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dimmerkv import DimmerKV
from dimmerkv.integrations.hf import CompressedChat, build_cache, cache_tensors

FACTS = [("the blue door", "4817"), ("the north tower", "2093"), ("the old library", "7765"),
         ("the river gate", "3358"), ("the glass bridge", "6142"), ("the iron well", "9520")]


def document(tokenizer, tokens: int, seed: int) -> str:
    from datasets import load_dataset

    lines = [l for l in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"] if len(l) > 200]
    random.Random(seed).shuffle(lines)
    paragraphs, total = [], 0
    while total < tokens:
        paragraphs.append(lines.pop().strip())
        total += len(tokenizer(paragraphs[-1])["input_ids"])
    for index, (place, code) in enumerate(FACTS):
        position = (index + 1) * len(paragraphs) // (len(FACTS) + 1)
        paragraphs.insert(position, f"Remember this: the secret code of {place} is {code}.")
    return "\n\n".join(paragraphs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--context-tokens", type=int, default=3000)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.03, 0.05, 0.1])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16 if device != "cpu" else torch.float32)
    model = model.to(device).eval()
    history = [{"role": "user", "content": document(tokenizer, args.context_tokens, args.seed)
                + "\n\nRead this carefully; I will ask about it later."},
               {"role": "assistant", "content": "Understood. Ask me anything about it."}]
    questions = [f"What is the secret code of {place}? Answer with the number only." for place, _ in FACTS]

    history_ids = tokenizer.apply_chat_template(history, tokenize=True, return_dict=True, return_tensors="pt")["input_ids"]
    with torch.inference_mode():
        exact_cache = model(input_ids=history_ids.to(device), use_cache=True).past_key_values
    exact_keys, exact_values = cache_tensors(exact_cache)
    print(f"history: {history_ids.shape[1]} tokens, FP16 cache {exact_keys.numel() * 4 / 2**20:.1f} MiB")

    def answer(chat_or_none, question):
        messages = history + [{"role": "user", "content": question}]
        if chat_or_none is not None:
            text = chat_or_none.reply(messages, max_new_tokens=8, save=False, do_sample=False)
            return text, chat_or_none.last_stats
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt")["input_ids"].to(device)
        cache = build_cache(exact_keys.clone(), exact_values.clone(), model.config)
        out = model.generate(input_ids=prompt, past_key_values=cache, max_new_tokens=8, do_sample=False)
        return tokenizer.decode(out[0, prompt.shape[1]:], skip_special_tokens=True), {}

    def first_token_logprobs(cache_keys, cache_values, question):
        messages = history + [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt")["input_ids"].to(device)
        n = cache_keys.shape[2]
        cache = build_cache(cache_keys.clone(), cache_values.clone(), model.config)
        with torch.inference_mode():
            logits = model(input_ids=prompt[:, n:], past_key_values=cache).logits[0, -1].float()
        return logits.log_softmax(-1)

    rows = [("full cache", None)]
    for budget in args.budgets:
        rows.append((f"DimmerKV {budget:.0%}", budget))
    for name, budget in rows:
        chat, size, kl = None, exact_keys.numel() * 4, []
        if budget is not None:
            kv = DimmerKV.open(args.checkpoint, tempfile.mkdtemp(), device=device, budget=budget)
            started = time.perf_counter()
            kv.save_conversation(history_ids[0].tolist(), exact_keys, exact_values)
            compress_ms = (time.perf_counter() - started) * 1000
            size = kv.store.stats()["stored_bytes"]
            chat = CompressedChat(model, tokenizer, kv)
            match = kv.lookup(history_ids[0].tolist())
            started = time.perf_counter()
            keys, values = kv.load(match, dtype=model.dtype, device=device)
            restore_ms = (time.perf_counter() - started) * 1000
            for question in questions:
                exact_lp = first_token_logprobs(exact_keys[:, :, : match.tokens], exact_values[:, :, : match.tokens], question)
                comp_lp = first_token_logprobs(keys, values, question)
                kl.append(float((exact_lp.exp() * (exact_lp - comp_lp)).sum()))
        correct, answers = 0, []
        for (place, code), question in zip(FACTS, questions):
            text, stats = answer(chat, question)
            correct += code in text
            answers.append(text.strip()[:12])
        extra = "" if budget is None else (f"  ratio {exact_keys.numel() * 4 / size:4.1f}x  compress {compress_ms:5.0f} ms"
                                           f"  restore {restore_ms:4.0f} ms  KL {sum(kl) / len(kl):.4f}"
                                           f"  prefilled {stats['prefilled_tokens']} of {stats['prompt_tokens']}")
        print(f"{name:14s} {size / 2**20:7.2f} MiB  correct {correct}/{len(FACTS)}{extra}  {answers}")


if __name__ == "__main__":
    main()
