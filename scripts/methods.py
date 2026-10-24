"""Compression methods by name, shared by every evaluation script.

A method name is one of:
- ``full``: the uncompressed cache (control).
- ``CODEC`` for both streams or ``KEYCODEC/VALUECODEC`` (uniform codes), where
  a codec is ``pca<k>`` (INT8 PCA, k components), ``lex<s>`` (Lexico, s atoms)
  or ``fp16``. Example: ``pca16/lex8``.
- ``lex@<budget>``: uniform Lexico with the most atoms that fit the budget.
- ``kvtc@<budget>``: kvtc transform coding (cross-layer PCA, RoPE removed,
  grouped bit widths), every token on the same rung; ``kvtcf@<budget>``:
  the same, with evenly spaced tokens moved one rung up to fill the budget;
  ``dkvtc@<budget>``: the same transform with DimmerKV flat per-token rungs
  (``kvtc.py``); ``dkvtc@<budget>:snap`` weights each token's error by its
  question-window attention (summed over layers and heads); ``:ev`` adds a
  zero-byte eviction level so budgets can go below the coarsest rung;
  ``:floor`` (or ``:floor0.3``) first gives every token the rung uniform kvtc
  could afford with half (or 30%) of the budget, then allocates the rest.
  These keep the first 4 tokens exact (kvtc's sinks) plus the recent window.
- ``prog@<budget>``: progressive bit-plane kvtc (``progressive.py``); the
  budget is the hot tier, DimmerKV picks each token's prefix (flat).
  Options: ``:snap`` (question-window weights), ``:r<frac>`` (at question
  time, fetch the full stream for the top ``frac`` of tokens by the
  question's attention over the hot cache). Example: ``prog@0.02:r0.02``.
- ``h2o:<budget>`` (keep the highest-scoring tokens exact, evict the rest) or
  ``dkv:<budget>`` (DimmerKV allocator over Lexico prefixes), budget as a
  fraction of FP16 KV bytes, with optional ``:`` options ``floor<n>``
  (minimum level, 1 = never evict), ``layer`` / ``head`` budget scope,
  ``mean`` scores, ``snap`` (last-32-query window scores; with h2o this is
  SnapKV-style eviction), ``flat`` (ignore scores) or ``pow<x>`` (score ** x),
  ``kw<x>`` key error weight. Example: ``dkv:0.15:flat``.
The first token and the last ``recent`` tokens stay exact in every method.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

import dimmer
import lexico
from bench_codecs import synchronize
from kvtc import KVTCCodec
from lexico import LexicoCodec
from progressive import ProgressiveCodec, question_importance
from pca_codec import PCACodec


@dataclass
class Compressed:
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    evicted: list[torch.Tensor] | None
    used_bytes: int
    seconds: float
    histogram: dict[str, float] = field(default_factory=dict)
    # Optional question-time refinement: refine(tail_ids, context_length) -> (keys, values, fetched bytes).
    refine: object = None
    # Stored bytes per token (keys + values), for the demo heat map.
    token_bytes: torch.Tensor | None = None


def parse_allocator(spec: str) -> dict:
    method, budget, *options = spec.split(":")
    plan = {"budget": float(budget), "atom_levels": () if method == "h2o" else dimmer.DEFAULT_ATOM_LEVELS,
            "floor": 0, "scope": "global", "mean": False, "key_weight": 1.0, "power": 1.0,
            "snap": False}
    for option in options:
        if option.startswith("floor"):
            plan["floor"] = int(option[5:])
        elif option in {"global", "layer", "head"}:
            plan["scope"] = option
        elif option == "mean":
            plan["mean"] = True
        elif option == "snap":
            plan["snap"] = True
        elif option == "flat":
            plan["power"] = 0.0
        elif option.startswith("pow"):
            plan["power"] = float(option[3:])
        elif option.startswith("kw"):
            plan["key_weight"] = float(option[2:])
        else:
            raise ValueError(f"Unknown option {option!r} in {spec!r}.")
    return plan


def parse_codec(spec: str) -> tuple[str, int]:
    if spec == "fp16":
        return "fp16", 0
    for prefix in ("pca", "lex"):
        if spec.startswith(prefix) and spec[len(prefix) :].isdigit():
            return prefix, int(spec[len(prefix) :])
    raise ValueError(f"Unknown codec {spec!r}; use pca<k>, lex<s> or fp16.")


def budget_of(name: str) -> float:
    """Nominal memory budget of a method name (1.0 for full, NaN if implicit)."""
    if name == "full":
        return 1.0
    if "@" in name:
        return float(name.split("@")[1].split(":")[0])
    if name.startswith(("h2o:", "dkv:")):
        return float(name.split(":")[1])
    return float("nan")


class Methods:
    """Applies named compression methods to a prefilled cache."""

    def __init__(self, model_id: str, basis: str | Path, dictionary: str | Path, device, recent: int = 32,
                 model=None, kvtc: str | Path = "checkpoints/kvtc.pt",
                 progressive: str | Path = "checkpoints/progressive.pt"):
        self.model_id = model_id
        self.device = device
        self.recent = recent
        self._model = model
        self._paths = {"pca": Path(basis), "lex": Path(dictionary), "kvtc": Path(kvtc),
                       "prog": Path(progressive)}
        self._codecs: dict[str, object] = {}

    def codec(self, kind: str):
        """Load a codec on first use and check it was fit for this model."""
        if kind not in self._codecs:
            path = self._paths[kind]
            if kind in ("kvtc", "prog"):
                if self._model is None:
                    raise ValueError("kvtc methods need Methods(..., model=model) for RoPE tables.")
                codec = (KVTCCodec(path, self.device, self._model) if kind == "kvtc"
                         else ProgressiveCodec(self._paths["kvtc"], path, self.device, self._model))
            else:
                codec = PCACodec(path, self.device) if kind == "pca" else LexicoCodec(path, self.device)
            if codec.model != self.model_id:
                raise RuntimeError(f"{path} was fit for {codec.model}; refit for {self.model_id}.")
            self._codecs[kind] = codec
        return self._codecs[kind]

    def _timed(self, fn):
        synchronize(self.device)
        started = time.perf_counter()
        result = fn()
        synchronize(self.device)
        return result, time.perf_counter() - started

    def _uniform(self, tensors, kind, codec, size):
        if codec == "fp16":
            return tensors, len(tensors) * tensors[0].numel() * 2
        if codec == "pca":
            return self.codec("pca").compress_stream(tensors, kind, size, 8, exact_recent=self.recent)
        return self.codec("lex").compress_stream(tensors, kind, size, exact_recent=self.recent)

    def _atoms_for_budget(self, budget: float, tokens: int, head_dim: int) -> int:
        """Most Lexico atoms per vector whose uniform code fits the budget."""
        middle = max(0, tokens - 1 - self.recent)
        best = 1
        for atoms in range(1, 33):
            ratio = ((tokens - middle) * head_dim * 2 + middle * lexico.bytes_per_vector(atoms)) / (
                tokens * head_dim * 2
            )
            if ratio <= budget:
                best = atoms
        return best

    def apply(self, name: str, keys: list[torch.Tensor], values: list[torch.Tensor], scores: dict) -> Compressed:
        """Compress per-layer [1, kv_heads, tokens, head_dim] keys and values."""
        if name == "full":
            used = 2 * len(keys) * keys[0].numel() * 2
            per_token = torch.full((keys[0].shape[2],), used / keys[0].shape[2], device=self.device)
            return Compressed(keys, values, None, used, 0.0, token_bytes=per_token)
        if name.startswith(("h2o:", "dkv:")):
            plan = parse_allocator(name)
            chosen = scores["snap" if plan["snap"] else "sum"]
            if plan["mean"]:
                chosen = dimmer.mean_scores(chosen)
            # power 0 = flat (allocate by error per byte only); 0 < power < 1 softens the scores.
            chosen = [score.clamp_min(0) ** plan["power"] for score in chosen]
            allocation, seconds = self._timed(lambda: dimmer.allocate(
                keys, values, chosen, self.codec("lex") if plan["atom_levels"] else None,
                plan["budget"], plan["atom_levels"], exact_recent=self.recent, floor=plan["floor"],
                key_weight=plan["key_weight"], scope=plan["scope"],
            ))
            # Per token: share of (layer, head) slots kept x the token's FP16 size (eviction view).
            kept = 1 - torch.stack(allocation.evicted).float().mean(dim=(0, 1))
            full = 2 * len(keys) * keys[0].shape[1] * keys[0].shape[3] * 2
            return Compressed(allocation.keys, allocation.values, allocation.evicted,
                              allocation.used_bytes, seconds, allocation.histogram, token_bytes=kept * full)
        if name.startswith("prog@"):
            return self._progressive(name, keys, values, scores)
        if name.startswith(("kvtc@", "kvtcf@", "dkvtc@")):
            per_token = name.startswith("dkvtc@")
            budget, *options = name.split("@")[1].split(":")
            weights, evict, floor_share = None, False, 0.0
            for option in options:
                if option == "snap":
                    weights = torch.stack(scores["snap"]).sum(dim=(0, 1))
                elif option == "ev" and per_token:
                    evict = True
                elif option.startswith("floor") and per_token:
                    floor_share = float(option[5:] or 0.5)
                else:
                    raise ValueError(f"Unknown option {option!r} in {name!r} (dkvtc supports :snap, :ev, :floor).")
            codec = self.codec("kvtc")
            (new_keys, new_values, used, histogram), seconds = self._timed(lambda: codec.compress(
                keys, values, float(budget), self.recent, per_token, fill=name.startswith("kvtcf@"),
                weights=weights, evict=evict, floor_share=floor_share,
            ))
            return Compressed(new_keys, new_values, codec.last_evicted, used, seconds, histogram,
                              token_bytes=self._kvtc_token_bytes(codec, keys[0].shape[2]))
        if name.startswith("lex@"):
            atoms = self._atoms_for_budget(float(name[4:]), keys[0].shape[-2], keys[0].shape[-1])
            specs = [("lex", atoms), ("lex", atoms)]
        else:
            specs = [parse_codec(part) for part in (name.split("/") * 2)[:2]]
        (key_codec, key_size), (value_codec, value_size) = specs
        (new_keys, key_bytes), key_time = self._timed(lambda: self._uniform(keys, "key", key_codec, key_size))
        (new_values, value_bytes), value_time = self._timed(
            lambda: self._uniform(values, "value", value_codec, value_size)
        )
        histogram = {f"{key_size} atoms": 1.0} if name.startswith("lex@") else {}
        return Compressed(new_keys, new_values, None, key_bytes + value_bytes, key_time + value_time, histogram)

    def _progressive(self, name: str, keys, values, scores) -> Compressed:
        budget, *options = name.split("@")[1].split(":")
        weights, fraction = None, 0.0
        for option in options:
            if option == "snap":
                weights = torch.stack(scores["snap"]).sum(dim=(0, 1))
            elif option.startswith("r"):
                fraction = float(option[1:])
            else:
                raise ValueError(f"Unknown option {option!r} in {name!r}.")
        codec = self.codec("prog")
        stored, seconds = self._timed(lambda: codec.encode(keys, values, float(budget), self.recent, weights))
        hot_keys, hot_values = stored.hot()
        refine = None
        if fraction > 0:
            middle = stored.end - stored.start

            def refine(tail_ids, context_length):
                importance = question_importance(self._model, hot_keys, hot_values, tail_ids, context_length)
                count = max(1, int(round(fraction * middle)))
                top = importance[stored.start : stored.end].topk(count).indices + stored.start
                return stored.refine(hot_keys, hot_values, top)
        per_token = torch.full((keys[0].shape[2],), 2.0 * codec.base.exact_bytes, device=self.device)
        per_token[stored.start : stored.end] = sum(
            codec.orders[kind].prefix_bytes[store.hot_pieces].float() for kind, store in stored.stores.items()
        )
        return Compressed(hot_keys, hot_values, None, stored.hot_bytes, seconds, refine=refine,
                          token_bytes=per_token)

    def _kvtc_token_bytes(self, codec, tokens: int) -> torch.Tensor:
        """Bytes kept per token (keys + values) from the codec's last rung choice."""
        per_token = torch.full((tokens,), 2.0 * codec.exact_bytes, device=self.device)
        start = min(codec.sinks, tokens)
        middle = codec.last_choice["key"].shape[0]
        total = torch.zeros(middle, device=self.device)
        for kind, choice in codec.last_choice.items():
            sizes = torch.tensor(codec.streams[kind].rung_bytes + [codec.exact_bytes], device=self.device,
                                 dtype=torch.float32)
            total += torch.where(choice < 0, torch.zeros_like(total), sizes[choice.clamp_min(0)])
        per_token[start : start + middle] = total
        return per_token
