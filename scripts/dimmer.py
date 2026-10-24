"""DimmerKV core: H2O token scores, byte-budget allocator, per-head eviction.

Level 1 implementation: chosen Lexico codes are decoded back into the cache
and evicted tokens are hidden from attention by a per-(layer, KV head) mask.
Evicted and compressed tokens still occupy FP16 slots in the tensor, so
memory is *counted* from the codes, not measured on the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from transformers import AttentionInterface
from transformers.masking_utils import AttentionMaskInterface, eager_mask

import lexico

ATTENTION_NAME = "dimmer"
DEFAULT_ATOM_LEVELS = (1, 2, 3, 4, 6, 8, 11, 16)
OMP_CHUNK = 16384


@dataclass
class _State:
    scoring: bool = False
    scores: dict[int, torch.Tensor] = field(default_factory=dict)
    evicted: dict[int, torch.Tensor] = field(default_factory=dict)


STATE = _State()


def _repeat_kv(tensor: torch.Tensor, groups: int) -> torch.Tensor:
    batch, heads, tokens, dim = tensor.shape
    if groups == 1:
        return tensor
    return tensor[:, :, None].expand(batch, heads, groups, tokens, dim).reshape(batch, heads * groups, tokens, dim)


def dimmer_attention(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """Eager attention plus H2O score recording and per-head eviction masks."""
    groups = module.num_key_value_groups
    key_states = _repeat_kv(key, groups)
    value_states = _repeat_kv(value, groups)
    weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        weights = weights + attention_mask
    evicted = STATE.evicted.get(module.layer_idx)
    if evicted is not None:
        # evicted: [kv_heads, context_tokens]; tokens added later are never evicted.
        hidden = torch.zeros(weights.shape[1], weights.shape[-1], dtype=torch.bool, device=weights.device)
        hidden[:, : evicted.shape[1]] = evicted.repeat_interleave(groups, dim=0)
        weights = weights.masked_fill(hidden[None, :, None, :], torch.finfo(weights.dtype).min)
    probabilities = torch.nn.functional.softmax(weights, dim=-1, dtype=torch.float32)
    if STATE.scoring:
        # Attention each key received, summed over queries and the query heads
        # that share its KV head (GQA): [kv_heads, keys].
        received = probabilities[0].sum(dim=1)
        STATE.scores[module.layer_idx] = received.view(-1, groups, received.shape[-1]).sum(dim=1)
    output = torch.matmul(probabilities.to(query.dtype), value_states)
    return output.transpose(1, 2).contiguous(), None


def install(model) -> None:
    """Route the model's attention through ``dimmer_attention``."""
    AttentionInterface.register(ATTENTION_NAME, dimmer_attention)
    AttentionMaskInterface.register(ATTENTION_NAME, eager_mask)
    model.set_attn_implementation(ATTENTION_NAME)


def set_eviction(evicted: list[torch.Tensor] | None) -> None:
    STATE.evicted = {} if evicted is None else dict(enumerate(evicted))


def prefill_with_scores(model, input_ids: torch.Tensor):
    """Prefill [1, tokens]; return (cache, per-layer H2O scores [kv_heads, tokens]).

    The score is H2O's accumulated attention each token received.
    """
    set_eviction(None)
    STATE.scores = {}
    STATE.scoring = True
    try:
        outputs = model(input_ids=input_ids, use_cache=True)
    finally:
        STATE.scoring = False
    return outputs.past_key_values, [STATE.scores[layer] for layer in range(len(STATE.scores))]


def mean_scores(scores: list[torch.Tensor]) -> list[torch.Tensor]:
    """Divide by the number of queries that could attend to each token,
    removing the causal bias of summed scores toward early tokens."""
    tokens = scores[0].shape[-1]
    counts = torch.arange(tokens, 0, -1, device=scores[0].device, dtype=torch.float32)
    return [score / counts for score in scores]


@dataclass
class Allocation:
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    evicted: list[torch.Tensor]  # Per layer [kv_heads, tokens], True = hidden.
    used_bytes: int
    budget_bytes: int
    histogram: dict[str, float]  # Share of middle (head, token) slots per level.


def _encode_with_errors(vectors: torch.Tensor, dictionary: torch.Tensor, sparsity: int):
    """OMP for [n, dim] vectors; returns indices [n, sparsity], errors [n, sparsity + 1]."""
    index_parts, error_parts = [], []
    for start in range(0, vectors.shape[0], OMP_CHUNK):
        errors: list[torch.Tensor] = []
        indices, _ = lexico.omp(vectors[start : start + OMP_CHUNK], dictionary, sparsity, errors=errors)
        index_parts.append(indices)
        error_parts.append(torch.stack(errors, dim=1))
    return torch.cat(index_parts), torch.cat(error_parts)


def _rebuild(vectors, indices, chosen, atom_levels, dictionary, exact_level):
    """Decode each vector at its chosen level; evicted slots become zero."""
    out = torch.zeros_like(vectors)
    exact = chosen == exact_level
    out[exact] = vectors[exact]
    for level, sparsity in enumerate(atom_levels, start=1):
        rows = chosen == level
        if rows.any():
            prefix = indices[rows, :sparsity]
            coefficients = lexico.refit(vectors[rows], prefix, dictionary)
            codes, scale = lexico.quantize_12bit(coefficients)
            out[rows] = lexico.reconstruct(prefix, codes.float() * scale.float(), dictionary)
    return out


def allocate(
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
    scores: list[torch.Tensor],
    codec: lexico.LexicoCodec | None,
    budget: float,
    atom_levels: tuple[int, ...] = DEFAULT_ATOM_LEVELS,
    exact_prefix: int = 1,
    exact_recent: int = 32,
    floor: int = 0,
    key_weight: float = 1.0,
    scope: str = "global",
) -> Allocation:
    """Greedy byte-budget allocation over per-token code prefixes.

    Each middle (layer, KV head, token) slot gets one level: evicted, Lexico
    with ``atom_levels[i]`` atoms for both K and V, or exact FP16. The value
    of moving a slot up one level is score x drop in relative K+V error /
    extra bytes; ``cummin`` along levels makes these non-increasing, so a
    stable greedy pick of the best pieces always takes a prefix per slot.
    ``atom_levels=()`` gives the H2O baseline (evict or exact). ``floor`` forces
    every slot to at least that level index (1 = never evict). ``budget`` is a
    fraction of the context's FP16 KV bytes and is shared ``global``ly, per
    ``layer`` or per ``head``.
    """
    if atom_levels and codec is None:
        raise ValueError("Atom levels need a Lexico codec.")
    batch, heads, tokens, head_dim = keys[0].shape
    if batch != 1:
        raise ValueError("Allocation supports batch size 1.")
    layers = len(keys)
    start = min(exact_prefix, tokens)
    end = max(start, tokens - exact_recent)
    middle = end - start
    exact_vector = head_dim * 2
    level_bytes = torch.tensor(
        [0] + [2 * lexico.bytes_per_vector(s) for s in atom_levels] + [2 * exact_vector],
        dtype=torch.float64,
        device=keys[0].device,
    )
    levels = level_bytes.shape[0]
    exact_level = levels - 1

    budget_bytes = int(budget * layers * heads * tokens * 2 * exact_vector)
    fixed_bytes = layers * heads * (tokens - middle) * 2 * exact_vector
    remaining = budget_bytes - fixed_bytes
    if floor:
        remaining -= int(level_bytes[floor]) * layers * heads * middle
    if remaining < 0:
        raise ValueError(f"Budget {budget:.3f} cannot cover the exact window and floor.")

    # Relative error of every slot at every level: [layers, heads, middle, levels].
    smax = max(atom_levels, default=0)
    errors = torch.zeros(layers, heads, middle, levels, device=keys[0].device)
    indices: dict[tuple[int, str], torch.Tensor] = {}
    for layer in range(layers):
        for kind, tensors, weight in (("key", keys, key_weight), ("value", values, 1.0)):
            vectors = tensors[layer][0, :, start:end].float().reshape(-1, head_dim)
            energy = vectors.square().sum(dim=1)
            scale = energy.mean().clamp_min(1e-12)
            stream = torch.zeros(vectors.shape[0], levels, device=vectors.device)
            stream[:, 0] = energy
            if atom_levels:
                stream_indices, stream_errors = _encode_with_errors(
                    vectors, codec.dictionary(layer, kind), smax
                )
                indices[layer, kind] = stream_indices
                stream[:, 1:exact_level] = stream_errors[:, list(atom_levels)]
            errors[layer] += weight * (stream / scale).view(heads, middle, levels)

    slot_scores = torch.stack([score[:, start:end] for score in scores]).float()
    gains = (errors[..., :-1] - errors[..., 1:]).clamp_min(0)
    piece_bytes = level_bytes[1:] - level_bytes[:-1]
    value = (slot_scores[..., None] * gains / piece_bytes.float()).cummin(dim=-1).values
    if floor:
        value[..., :floor] = float("inf")

    if scope == "global":
        groups = value.reshape(1, -1, levels - 1)
    elif scope == "layer":
        groups = value.reshape(layers, -1, levels - 1)
    elif scope == "head":
        groups = value.reshape(layers * heads, -1, levels - 1)
    else:
        raise ValueError("scope must be 'global', 'layer' or 'head'")
    group_budget = remaining / groups.shape[0] + (
        int(level_bytes[floor]) * groups.shape[1] if floor else 0
    )
    chosen_parts = []
    for group in groups:
        flat = group.reshape(-1)
        order = torch.sort(flat, descending=True, stable=True).indices
        cost = piece_bytes.repeat(group.shape[0])[order]
        taken = order[cost.cumsum(0) <= group_budget]
        counts = torch.zeros(group.shape[0], dtype=torch.long, device=flat.device)
        counts.index_add_(0, taken // (levels - 1), torch.ones_like(taken))
        chosen_parts.append(counts)
    chosen = torch.stack(chosen_parts).view(layers, heads, middle)

    used = fixed_bytes + int(level_bytes[chosen].sum())
    if used > budget_bytes:
        raise AssertionError(f"Allocator used {used} bytes over budget {budget_bytes}.")

    new_keys, new_values, evicted = [], [], []
    for layer in range(layers):
        hidden = torch.zeros(heads, tokens, dtype=torch.bool, device=keys[0].device)
        hidden[:, start:end] = chosen[layer] == 0
        evicted.append(hidden)
        for kind, tensors, out in (("key", keys, new_keys), ("value", values, new_values)):
            rebuilt = tensors[layer].clone()
            vectors = tensors[layer][0, :, start:end].float().reshape(-1, head_dim)
            decoded = _rebuild(
                vectors,
                indices.get((layer, kind)),
                chosen[layer].reshape(-1),
                atom_levels,
                codec.dictionary(layer, kind) if atom_levels else None,
                exact_level,
            )
            rebuilt[0, :, start:end] = decoded.view(heads, middle, head_dim).to(rebuilt.dtype)
            out.append(rebuilt)

    names = ["evict"] + [f"{s} atoms" for s in atom_levels] + ["exact"]
    counts = torch.bincount(chosen.reshape(-1), minlength=levels).float() / max(chosen.numel(), 1)
    histogram = {name: float(share) for name, share in zip(names, counts)}
    return Allocation(new_keys, new_values, evicted, used, budget_bytes, histogram)
