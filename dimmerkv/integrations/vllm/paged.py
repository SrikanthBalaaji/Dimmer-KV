"""Moving one sequence's keys/values between vLLM's paged KV cache and stacked tensors.

No vLLM import here, so this is testable anywhere. Two per-layer layouts:

- standardized ``[num_blocks, kv_heads, block_size, head_size * 2]`` (keys in
  the first half of the last dim), as allocated by current vLLM v1;
- legacy FlashAttention ``[2, num_blocks, block_size, kv_heads, head_size]``.
"""

from __future__ import annotations

import re

import torch


def slot_mapping(block_ids: list[int], start: int, end: int, block_size: int) -> torch.Tensor:
    """Cache slots of token positions ``[start, end)`` of a request with these blocks."""
    positions = torch.arange(start, end)
    blocks = torch.as_tensor(block_ids, dtype=torch.long)[positions // block_size]
    return blocks * block_size + positions % block_size


def layer_index(name: str) -> int:
    numbers = [int(part) for part in re.findall(r"\.(\d+)\.", f".{name}.")]
    if not numbers:
        raise ValueError(f"Cannot find a layer index in {name!r}.")
    return numbers[0]


def ordered_layers(kv_caches: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    return [kv_caches[name] for name in sorted(kv_caches, key=layer_index)]


def geometry(cache: torch.Tensor) -> tuple[int, int]:
    """(kv_heads, head_size) of one layer's paged cache."""
    if cache.dim() == 4:
        return cache.shape[1], cache.shape[3] // 2
    if cache.dim() == 5 and cache.shape[0] == 2:
        return cache.shape[3], cache.shape[4]
    raise ValueError(f"Unsupported KV cache layout {tuple(cache.shape)}.")


def gather(layers: list[torch.Tensor], slots: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy the slots' keys and values out: [layers, kv_heads, tokens, head_size] each."""
    slots = slots.to(layers[0].device)
    blocks, offsets = slots // block_size, slots % block_size
    keys, values = [], []
    for cache in layers:
        if cache.dim() == 4:
            rows = cache[blocks, :, offsets]  # [tokens, heads, 2 * head_size]
            half = rows.shape[-1] // 2
            keys.append(rows[..., :half])
            values.append(rows[..., half:])
        else:
            keys.append(cache[0][blocks, offsets])  # [tokens, heads, head_size]
            values.append(cache[1][blocks, offsets])
    return torch.stack(keys).transpose(1, 2).contiguous(), torch.stack(values).transpose(1, 2).contiguous()


def scatter(layers: list[torch.Tensor], keys: torch.Tensor, values: torch.Tensor, slots: torch.Tensor,
            block_size: int) -> None:
    """Write [layers, kv_heads, tokens, head_size] keys and values into the slots."""
    slots = slots.to(layers[0].device)
    blocks, offsets = slots // block_size, slots % block_size
    for layer, cache in enumerate(layers):
        k = keys[layer].transpose(0, 1).to(device=cache.device, dtype=cache.dtype)  # [tokens, heads, dim]
        v = values[layer].transpose(0, 1).to(device=cache.device, dtype=cache.dtype)
        if cache.dim() == 4:
            cache[blocks, :, offsets] = torch.cat([k, v], dim=-1)
        else:
            cache[0][blocks, offsets] = k
            cache[1][blocks, offsets] = v
