"""Attention over the compressed cache, without rebuilding the full FP16 cache.

After kvtc / DimmerKV compression, each compressed token is a vector of PCA
coefficients ``c`` (one per stream) with ``x = c @ basis.T + mean`` spanning
every layer and head. Instead of decoding everything back to FP16:

- **Keys (decode per layer):** when attention reaches layer ``l``, only that
  layer's keys are rebuilt (coefficients times that layer's slice of the
  basis, plus mean, then RoPE at their positions), used, and discarded.
- **Values (never decoded):** the attention output is
  ``sum_n p_n v_n = (sum_n p_n c_n) @ B_v + mean_v * sum_n p_n``, so the
  probabilities are applied to the small coefficient vectors and the basis
  is applied once per head.

Only coefficient columns that are non-zero for some token are kept (at a
2–5% budget most PCA groups carry 0 bits). The few exact tokens (attention
sinks, recent window, and new question/answer tokens) live in an ordinary
small DynamicCache. Both parts share one softmax, so the result equals
attention over the fully decoded cache up to FP16 rounding.
"""

from __future__ import annotations

import torch

import dimmer
from common import build_cache
from kvtc import KVTCCodec, rerotate, rope_tables


class CompressedContext:
    """Compressed middle tokens of one conversation, attended to directly."""

    def __init__(self, codec: KVTCCodec, keys: list[torch.Tensor], values: list[torch.Tensor], model,
                 dtype=torch.float16):
        start, end = codec.last_span
        self.codec, self.model, self.dtype = codec, model, dtype
        self.start, self.end = start, end
        self.layers, self.heads, self.head_dim = codec.layers, codec.heads, codec.head_dim
        tokens = keys[0].shape[2]
        cos, sin = rope_tables(model, tokens, codec.device)
        self.cos, self.sin = cos[start:end].to(dtype), sin[start:end].to(dtype)
        # Tokens are grouped by precision step; each block keeps only the components its tokens use,
        # so coarse tokens cost little. Evicted keys are left out entirely (never attended).
        self.mean = {kind: codec.streams[kind].mean.to(dtype).reshape(self.layers, -1) for kind in ("key", "value")}
        self.blocks, self.basis = {}, {}
        key_choice = codec.last_choice["key"]
        middle = end - start
        for kind in ("key", "value"):
            stream, decoded, choice = codec.streams[kind], codec.last_coefficients[kind], codec.last_choice[kind]
            kept = (choice >= 0) & (key_choice >= 0)  # Evicted keys are never attended; evicted values = mean.
            # One basis per stream: the union of components any kept token uses (no duplicated rows).
            union = (decoded[kept] != 0).any(dim=0).nonzero().flatten()
            self.basis[kind] = stream.basis[:, union].T.to(dtype).reshape(len(union), self.layers, -1)
            # Blocks by precision step, small ones merged into the next finer step, so that
            # few matrix multiplies run per layer while coarse tokens stay cheap.
            minimum = max(64, int(0.05 * middle))
            pending, blocks = None, []
            for rung in sorted(choice[kept].unique().tolist()):
                rows = (kept & (choice == rung)).nonzero().flatten()
                pending = rows if pending is None else torch.cat([pending, rows])
                if pending.numel() >= minimum:
                    blocks.append(pending)
                    pending = None
            if pending is not None:
                if blocks:
                    blocks[-1] = torch.cat([blocks[-1], pending])
                else:
                    blocks.append(pending)
            self.blocks[kind] = []
            for rows in blocks:
                local = (decoded[rows][:, union] != 0).any(dim=0).nonzero().flatten()  # Indices into the union.
                self.blocks[kind].append({
                    "rows": rows,
                    "columns": local,
                    "coefficients": decoded[rows][:, union[local]].to(dtype).contiguous(),  # [n, r_block]
                })
        # Token order used for scores: the key blocks, concatenated.
        order = torch.cat([b["rows"] for b in self.blocks["key"]])
        self.order_cos, self.order_sin = self.cos[order], self.sin[order]
        position_in_order = torch.full((end - start,), -1, dtype=torch.long, device=codec.device)
        position_in_order[order] = torch.arange(order.numel(), device=codec.device)
        for block in self.blocks["value"]:
            block["index"] = position_in_order[block["rows"]]  # Where its tokens sit in the score order.
        # Exact tokens in an ordinary cache: sinks then the recent window, in sequence order.
        exact_keys = [torch.cat([k[..., :start, :], k[..., end:, :]], dim=2) for k in keys]
        exact_values = [torch.cat([v[..., :start, :], v[..., end:, :]], dim=2) for v in values]
        self.exact_cache = build_cache(exact_keys, exact_values, model)
        self.context_tokens = tokens

    # ----- memory accounting -----
    def code_bytes(self) -> int:
        """Bytes held per conversation for the compressed tokens (FP16 coefficients)."""
        return sum(b["coefficients"].numel() * 2 for blocks in self.blocks.values() for b in blocks)

    def basis_bytes(self) -> int:
        """Bytes of the kept basis rows (per model; shared by all conversations)."""
        return sum(t.numel() * t.element_size() for t in self.basis.values())

    def exact_bytes(self) -> int:
        layers = self.exact_cache.layers
        return sum((l.keys.numel() + l.values.numel()) * l.keys.element_size() for l in layers)

    # ----- attention -----
    def _keys(self, layer: int) -> torch.Tensor:
        """Layer ``layer``'s compressed keys in score order, rebuilt now: [kv_heads, n, head_dim]."""
        basis = self.basis["key"][:, layer]
        flat = torch.cat([b["coefficients"] @ basis[b["columns"]] for b in self.blocks["key"]])
        flat = flat + self.mean["key"][layer]
        heads_first = flat.view(-1, self.heads, self.head_dim).transpose(0, 1)
        return rerotate(heads_first, self.order_cos, self.order_sin)

    def attend(self, module, query, key, value, attention_mask, scaling):
        """Same contract as an HF attention function: returns (output [B, T, H, D], None)."""
        layer, groups = module.layer_idx, module.num_key_value_groups
        batch, q_heads, q_len, dim = query.shape
        # Compressed part: keys decoded for this layer only.
        compressed_keys = self._keys(layer)
        grouped_query = query.view(batch, self.heads, groups, q_len, dim)
        scores_c = torch.einsum("bhgqd,hnd->bhgqn", grouped_query, compressed_keys) * scaling
        # Exact part: sinks, recent window and new tokens, with the usual causal mask.
        scores_e = torch.einsum("bhgqd,bhnd->bhgqn", grouped_query, key) * scaling
        if attention_mask is not None:
            scores_e = scores_e + attention_mask[:, :, None]
        middle = scores_c.shape[-1]
        probabilities = torch.softmax(torch.cat([scores_c, scores_e], dim=-1), dim=-1, dtype=torch.float32)
        p_c = probabilities[..., :middle].to(query.dtype)
        p_e = probabilities[..., middle:].to(query.dtype)
        # Values: weight each block's coefficients, then one basis multiply per head (never decoded).
        out_c = p_c.sum(dim=-1, keepdim=True) * self.mean["value"][layer].view(1, self.heads, 1, 1, dim)
        layer_basis = self.basis["value"][:, layer].view(-1, self.heads, self.head_dim)  # [r, kv_heads, dim]
        for block in self.blocks["value"]:
            weighted = torch.einsum("bhgqn,nr->bhgqr", p_c[..., block["index"]], block["coefficients"])
            out_c = out_c + torch.einsum("bhgqr,rhd->bhgqd", weighted, layer_basis[block["columns"]])
        out_e = torch.einsum("bhgqn,bhnd->bhgqd", p_e, value)
        output = (out_c + out_e).reshape(batch, q_heads, q_len, dim)
        return output.transpose(1, 2).contiguous(), None


def greedy_answer(model, tokenizer, context: CompressedContext | None, cache, input_ids: torch.Tensor,
                  start_position: int, max_new_tokens: int = 12, evicted=None, stop_at_eos: bool = True):
    """Greedy decoding on top of a prefilled cache.

    With ``context`` set, attention runs over the compressed tokens directly and
    ``cache`` must be ``context.exact_cache``; otherwise ``cache`` is a full cache
    (optionally with per-layer ``evicted`` masks). Returns (text, first-step logits).
    """
    device = input_ids.device
    dimmer.STATE.compressed = context
    dimmer.set_eviction(evicted)
    generated, first_logits = [], None
    try:
        ids = input_ids
        position = start_position
        for _ in range(max_new_tokens):
            positions = torch.arange(position, position + ids.shape[1], device=device)[None]
            logits = model(input_ids=ids, past_key_values=cache, position_ids=positions, use_cache=True).logits
            if first_logits is None:
                first_logits = logits[0, -1].float()
            position += ids.shape[1]
            token = int(logits[0, -1].argmax())
            if stop_at_eos and (token == tokenizer.eos_token_id or token in tokenizer.all_special_ids):
                break
            generated.append(token)
            ids = torch.tensor([[token]], device=device)
    finally:
        dimmer.STATE.compressed = None
        dimmer.set_eviction(None)
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), first_logits
