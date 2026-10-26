"""Hugging Face transformers integration: compressed multi-turn chat on top of ``generate``.

``CompressedChat.reply(messages)`` restores the longest stored prefix of the
prompt (earlier turns, or a shared system prompt / document) from the
DimmerKV store, lets ``generate`` prefill only the rest, and stores the tokens
this turn added (prompt tail + answer) as a new compressed segment. Nothing
is kept on the GPU between turns.

``compress_cache`` / ``restore_cache`` do the same for a single
``DynamicCache`` when you manage storage yourself.
"""

from __future__ import annotations

import torch
from transformers import DynamicCache

from ..codec import DimmerKVCodec
from ..engine import DimmerKV


def cache_tensors(cache: DynamicCache) -> tuple[torch.Tensor, torch.Tensor]:
    """A batch-1 DynamicCache -> keys, values [layers, kv_heads, tokens, head_dim]."""
    layers = cache.layers
    if any("Sliding" in type(layer).__name__ for layer in layers):
        raise NotImplementedError("Sliding-window cache layers are not supported.")
    if layers[0].keys.shape[0] != 1:
        raise ValueError("Only batch size 1 caches can be compressed.")
    return torch.stack([layer.keys[0] for layer in layers]), torch.stack([layer.values[0] for layer in layers])


def build_cache(keys: torch.Tensor, values: torch.Tensor, config) -> DynamicCache:
    """Stacked keys, values -> a DynamicCache ready for ``generate(past_key_values=...)``."""
    return DynamicCache([(k[None], v[None]) for k, v in zip(keys, values)], config=config)


def compress_cache(codec: DimmerKVCodec, cache: DynamicCache, **settings) -> bytes:
    keys, values = cache_tensors(cache)
    return codec.compress(keys, values, **settings)


def restore_cache(codec: DimmerKVCodec, blob: bytes, model) -> DynamicCache:
    keys, values, offset = codec.decompress(blob, dtype=model.dtype, device=model.device)
    if offset:
        raise ValueError("This blob is a mid-conversation segment; restore it with DimmerKV.load().")
    return build_cache(keys, values, model.config)


class CompressedChat:
    """Multi-turn chat whose history lives compressed in a DimmerKV store between turns."""

    def __init__(self, model, tokenizer, kv: DimmerKV, namespace: str | None = None) -> None:
        self.model, self.tokenizer, self.kv, self.namespace = model, tokenizer, kv, namespace
        self.last_stats: dict = {}

    def prompt_ids(self, messages) -> torch.Tensor:
        encoded = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                     return_dict=True, return_tensors="pt")
        return encoded["input_ids"]

    def restore(self, prompt: list[int]):
        """Cache holding the longest stored prefix of ``prompt`` (at least one token left to run)."""
        match = self.kv.lookup(prompt, len(prompt) - 1, self.namespace)
        if match is None:
            return DynamicCache(config=self.model.config), 0
        keys, values = self.kv.load(match, dtype=self.model.dtype, device=self.model.device)
        return build_cache(keys, values, self.model.config), match.tokens

    def reply(self, messages, max_new_tokens: int = 256, save: bool = True, **generate_kwargs) -> str:
        input_ids = self.prompt_ids(messages)
        prompt = input_ids[0].tolist()
        cache, restored = self.restore(prompt)
        output = self.model.generate(input_ids=input_ids.to(self.model.device), past_key_values=cache,
                                     max_new_tokens=max_new_tokens, return_dict_in_generate=True,
                                     **generate_kwargs)
        sequence = output.sequences[0].tolist()
        stored = None
        if save:
            cache = output.past_key_values
            tokens = cache.get_seq_length()  # The last sampled token has no KV yet.
            keys, values = cache_tensors(cache)
            stored = self.kv.save_conversation(sequence[:tokens], keys, values, self.namespace)
        self.last_stats = {"prompt_tokens": len(prompt), "restored_tokens": restored,
                           "prefilled_tokens": len(prompt) - restored, "stored_segment": stored}
        return self.tokenizer.decode(sequence[len(prompt):], skip_special_tokens=True)
