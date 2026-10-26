"""Multi-turn chat whose history is kept compressed between turns (Hugging Face transformers).

    dimmerkv calibrate --model Qwen/Qwen2.5-0.5B-Instruct --output checkpoints/qwen2.5-0.5b.pt
    python examples/hf_chat.py --checkpoint checkpoints/qwen2.5-0.5b.pt

Each turn restores the stored conversation, prefills only the new message,
and stores what the turn added. Run it twice: the second run starts from the
conversation stored on disk by the first.
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dimmerkv import DimmerKV
from dimmerkv.integrations.hf import CompressedChat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--store", default="~/.cache/dimmerkv/hf-demo")
    parser.add_argument("--budget", type=float, default=0.05)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16 if device != "cpu" else torch.float32)
    model = model.to(device).eval()
    chat = CompressedChat(model, tokenizer, DimmerKV.open(args.checkpoint, args.store, device=device, budget=args.budget))

    messages = [{"role": "system", "content": "You are a concise assistant."}]
    for question in ["My name is Priya and I'm planning a trip to Kyoto in April. Suggest three things to do.",
                     "Which of those is best on a rainy day?",
                     "What's my name, and where am I going?"]:
        messages.append({"role": "user", "content": question})
        answer = chat.reply(messages, max_new_tokens=120, do_sample=False)
        messages.append({"role": "assistant", "content": answer})
        stats = chat.last_stats
        print(f"> {question}\n{answer}\n  [prompt {stats['prompt_tokens']} tokens: restored "
              f"{stats['restored_tokens']}, prefilled {stats['prefilled_tokens']}]\n")
    print("store:", chat.kv.store.stats())


if __name__ == "__main__":
    main()
