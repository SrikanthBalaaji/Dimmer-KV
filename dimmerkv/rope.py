"""RoPE tables from saved inverse frequencies, so the codec needs no model object.

kvtc removes RoPE from keys before the cross-layer PCA and puts it back after
decoding. The original scripts asked the Hugging Face model for cos/sin; an
inference engine's worker has no such object, so the calibration file stores
the rotary ``inv_freq`` and ``attention_scaling`` and the tables are rebuilt
here (same arithmetic as Hugging Face: positions times frequencies in
float32). Only NeoX-style ``rotate_half`` RoPE is supported (Llama, Qwen,
Mistral, Gemma); partial rotary (``rotary_dim < head_dim``) is handled.
"""

from __future__ import annotations

import importlib

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class RoPE:
    def __init__(self, inv_freq: torch.Tensor, attention_scaling: float = 1.0) -> None:
        self.inv_freq = inv_freq.detach().float().flatten()
        self.attention_scaling = float(attention_scaling)

    @property
    def rotary_dim(self) -> int:
        return 2 * self.inv_freq.numel()

    def to(self, device) -> "RoPE":
        return RoPE(self.inv_freq.to(device), self.attention_scaling)

    def state(self) -> dict:
        return {"inv_freq": self.inv_freq.cpu(), "attention_scaling": self.attention_scaling}

    def tables(self, offset: int, tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        """cos, sin [tokens, rotary_dim] in float32 for positions offset..offset+tokens-1."""
        positions = torch.arange(offset, offset + tokens, device=self.inv_freq.device, dtype=torch.float32)
        freqs = positions[:, None] * self.inv_freq[None]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos() * self.attention_scaling, emb.sin() * self.attention_scaling

    def derotate(self, keys: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Undo RoPE on [..., tokens, head_dim] keys (inverse rotation, any scaling)."""
        keys = keys.float()
        if self.rotary_dim == keys.shape[-1]:
            return (keys * cos - rotate_half(keys) * sin) / (cos.square() + sin.square())
        rot, rest = keys[..., : self.rotary_dim], keys[..., self.rotary_dim :]
        rot = (rot * cos - rotate_half(rot) * sin) / (cos.square() + sin.square())
        return torch.cat((rot, rest), dim=-1)

    def rerotate(self, keys: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        if self.rotary_dim == keys.shape[-1]:
            return keys * cos + rotate_half(keys) * sin
        rot, rest = keys[..., : self.rotary_dim], keys[..., self.rotary_dim :]
        return torch.cat((rot * cos + rotate_half(rot) * sin, rest), dim=-1)

    @classmethod
    def from_hf_model(cls, model) -> "RoPE":
        """Read the rotary embedding of a loaded Hugging Face causal LM."""
        rotary = model.model.rotary_emb
        return cls(rotary.inv_freq, getattr(rotary, "attention_scaling", 1.0))

    @classmethod
    def from_hf_config(cls, config) -> "RoPE":
        """Build the model's own rotary embedding class from its config (no weights)."""
        module = importlib.import_module(f"transformers.models.{config.model_type}.modeling_{config.model_type}")
        candidates = [getattr(module, name) for name in dir(module) if name.endswith("RotaryEmbedding")]
        if not candidates:
            raise ValueError(f"No RotaryEmbedding class in {module.__name__}.")
        rotary = candidates[0](config=config)
        return cls(rotary.inv_freq, getattr(rotary, "attention_scaling", 1.0))
