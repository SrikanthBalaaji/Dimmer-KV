"""``DimmerKV``: codec + conversation store behind three calls, for engine integrations.

- ``lookup(token_ids)``: longest stored prefix of a prompt.
- ``load(match)``: decode it to keys/values ``[layers, kv_heads, tokens, head_dim]``.
- ``save(token_ids, keys, values, start, parent)``: compress the tokens a turn
  added and chain them to the stored prefix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from .codec import DimmerKVCodec
from .store import ConversationStore, Match


@dataclass
class CompressionSettings:
    """Defaults follow the repository's recommended ``dkvtc@0.03:floor1`` (question not known)."""

    budget: float = 0.03
    recent: int = 32
    floor_share: float = 1.0
    per_token: bool = True
    key_weight: float = 1.0
    zlib_level: int = 6
    # Exact recent tokens may take at most this share of a segment's budget, so short chat turns are not
    # stored mostly exact (long segments keep all ``recent`` tokens, as in the paper's runs).
    recent_share: float = 0.5

    def kwargs(self, tokens: int) -> dict:
        recent = min(self.recent, int(self.recent_share * self.budget * tokens))
        return {"budget": self.budget, "recent": recent, "floor_share": self.floor_share,
                "per_token": self.per_token, "key_weight": self.key_weight, "zlib_level": self.zlib_level}


@dataclass
class DimmerKV:
    codec: DimmerKVCodec
    store: ConversationStore
    settings: CompressionSettings = field(default_factory=CompressionSettings)

    @classmethod
    def open(cls, checkpoint: str | Path, store_path: str | Path, device="cpu", block_size: int = 16,
             ram_bytes: int = 256 << 20, max_disk_bytes: int | None = None, max_rung_fraction: float | None = None,
             **settings) -> "DimmerKV":
        codec = DimmerKVCodec(checkpoint, device=device, max_rung_fraction=max_rung_fraction)
        store = ConversationStore(store_path, codec.fingerprint, block_size, ram_bytes, max_disk_bytes)
        return cls(codec, store, CompressionSettings(**settings))

    @property
    def block_size(self) -> int:
        return self.store.block_size

    def lookup(self, token_ids, max_tokens: int | None = None, namespace: str | None = None) -> Match | None:
        return self.store.longest_prefix(token_ids, max_tokens, self.store.namespaced(namespace))

    def load(self, match: Match, dtype=torch.float16, device=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Keys and values of the first ``match.tokens`` tokens."""
        keys, values = [], []
        for blob, start, stop in self.store.pieces(match):
            k, v, offset = self.codec.decompress(blob, dtype=dtype, device=device)
            if offset != start:
                raise ValueError(f"Segment starts at {offset}, index says {start}.")
            keys.append(k[:, :, : stop - start])
            values.append(v[:, :, : stop - start])
        if len(keys) == 1:
            return keys[0], values[0]
        return torch.cat(keys, dim=2), torch.cat(values, dim=2)

    def save(self, token_ids, keys: torch.Tensor, values: torch.Tensor, start: int = 0,
             parent: str | None = None, namespace: str | None = None) -> str | None:
        """Store tokens ``[start, len(token_ids))`` whose keys/values are given ([..., tokens, dim]).

        ``token_ids`` is the whole conversation so far. Its end is cut to a
        block boundary (the cut tail is cheap to recompute). Returns the new
        segment id, or None when nothing new remains.
        """
        end = len(token_ids) // self.block_size * self.block_size
        if end <= start:
            return None
        keys, values = keys[:, :, : end - start], values[:, :, : end - start]
        blob = self.codec.compress(keys, values, position_offset=start, **self.settings.kwargs(end - start))
        return self.store.put(list(token_ids[:end]), start, blob, parent, self.store.namespaced(namespace))

    def save_conversation(self, token_ids, keys: torch.Tensor, values: torch.Tensor,
                          namespace: str | None = None) -> str | None:
        """Store a whole conversation's cache, compressing only what the store does not have yet."""
        end = len(token_ids) // self.block_size * self.block_size
        match = self.lookup(token_ids, end, namespace)
        start = match.tokens if match else 0
        return self.save(token_ids, keys[:, :, start:], values[:, :, start:], start,
                         match.segment if match else None, namespace)
