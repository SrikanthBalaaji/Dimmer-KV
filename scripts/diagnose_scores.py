"""Where do the needle answer tokens rank under each token score?

Prefills the needle contexts, then reports the percentile of the answer
tokens (the hidden numbers) among all middle tokens, per score, and the
levels the allocator gives them. Low percentiles mean the score cannot see,
before the question, which tokens will matter.
"""

from __future__ import annotations

import argparse

import torch

import dimmer
from common import DEFAULT_MODEL, load_model, wikitext_token_chunks
from lexico import LexicoCodec
from test_generation import build_prompt, split_ids


def answer_positions(tokenizer, context: str, question: str, answers: list[str]) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": f"Read this document.\n\n{context}\n\n"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    positions = []
    for answer in answers:
        begin = rendered.index(answer)
        end = begin + len(answer)
        positions += [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if a < end and b > begin]
    return positions


def percentile(score: torch.Tensor, positions: list[int], start: int, end: int) -> float:
    """Mean percentile (0 = lowest) of positions among middle tokens."""
    middle = score[start:end]
    ranks = middle.argsort().argsort().float() / (middle.shape[0] - 1)
    return float(ranks[[p - start for p in positions]].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dictionary", default="checkpoints/lexico_dict.pt")
    parser.add_argument("--context-tokens", type=int, default=1500)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--recent", type=int, default=32)
    parser.add_argument("--budget", type=float, default=0.15)
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model)
    dimmer.install(model)
    codec = LexicoCodec(args.dictionary, device)
    for seed, filler_ids in enumerate(wikitext_token_chunks(tokenizer, "test", args.context_tokens, args.seeds)):
        context, qa = build_prompt(tokenizer.decode(filler_ids[1:]), seed)
        context_ids, _ = split_ids(tokenizer, context, qa[0][0])
        positions = answer_positions(tokenizer, context, qa[0][0], [answer for _, answer in qa])
        with torch.inference_mode():
            cache, all_scores = dimmer.prefill_with_scores(model, context_ids[None].to(device))
            scores = all_scores["sum"]
            keys = [layer.keys for layer in cache.layers]
            values = [layer.values for layer in cache.layers]
            tokens = context_ids.shape[0]
            start, end = 1, tokens - args.recent
            for name, chosen in (("sum (H2O)", scores), ("mean", dimmer.mean_scores(scores))):
                # Average over layers and KV heads for one summary number.
                combined = torch.stack(chosen).mean(dim=(0, 1))
                per_head = [percentile(s[h], positions, start, end) for s in chosen for h in range(s.shape[0])]
                print(f"seed {seed} {name:>10}: needle percentile (avg score) {percentile(combined, positions, start, end):.2f}, "
                      f"per head median {sorted(per_head)[len(per_head) // 2]:.2f}")
            allocation = dimmer.allocate(keys, values, scores, codec, args.budget, exact_recent=args.recent)
            levels = torch.zeros(len(positions))
            names = list(allocation.histogram)
            # Recover levels from the rebuilt cache: evicted mask and exactness.
            evicted = torch.stack([e[:, positions] for e in allocation.evicted]).float().mean()
            print(f"seed {seed} needle tokens: {len(positions)}, evicted share {float(evicted):.2f} "
                  f"(all tokens: {allocation.histogram['evict']:.2f})")


if __name__ == "__main__":
    main()
