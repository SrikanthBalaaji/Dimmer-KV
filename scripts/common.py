"""Shared helpers: device choice, model loading, cache access, calibration text."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


@dataclass(frozen=True)
class KVConfig:
    """Cache geometry derived from the model configuration."""

    layers: int
    kv_heads: int
    head_dim: int

    def fp16_bytes_per_token(self) -> int:
        """Bytes one token occupies in an uncompressed FP16 K+V cache."""
        return self.layers * self.kv_heads * self.head_dim * 2 * 2


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    model_id: str = DEFAULT_MODEL,
) -> tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    """Load a causal LM with eager attention so attention weights are available."""
    device = choose_device()
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    return tokenizer, model, device


def kv_config(model: AutoModelForCausalLM) -> KVConfig:
    config = model.config
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    return KVConfig(
        layers=config.num_hidden_layers,
        kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
        head_dim=head_dim,
    )


def cache_tensors(cache: DynamicCache) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Return per-layer keys and values shaped [batch, kv_heads, tokens, head_dim]."""
    layers = getattr(cache, "layers", None)
    if not layers:
        raise RuntimeError("The model did not return a DynamicCache with .layers.")
    return [layer.keys for layer in layers], [layer.values for layer in layers]


def build_cache(
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
    model: AutoModelForCausalLM,
) -> DynamicCache:
    """Build a fresh DynamicCache from per-layer tensors."""
    return DynamicCache(list(zip(keys, values)), config=model.config)


def wikitext_token_chunks(
    tokenizer: AutoTokenizer,
    split: str,
    chunk_tokens: int,
    max_chunks: int,
) -> list[torch.Tensor]:
    """Concatenate WikiText-2 and cut it into chunks, each starting with BOS.

    Every chunk begins with the BOS token so the cache layout matches real
    prompts; callers decide whether to skip that position.
    """
    from datasets import load_dataset

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n".join(line for line in dataset["text"] if line.strip())
    body = chunk_tokens - 1
    ids = tokenizer(
        text[: body * max_chunks * 8],  # ~4+ characters per token; trimmed below.
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]
    bos = torch.tensor([tokenizer.bos_token_id])
    chunks = [
        torch.cat([bos, ids[start : start + body]])
        for start in range(0, ids.shape[0] - body + 1, body)
    ][:max_chunks]
    if not chunks:
        raise RuntimeError("WikiText produced no chunks; lower chunk_tokens.")
    return chunks


def collect_vectors(
    model: AutoModelForCausalLM,
    chunks: list[torch.Tensor],
    device: torch.device,
    skip_prefix: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Prefill each chunk and gather cache vectors per layer.

    Returns per-layer keys and values on the CPU in float32, shaped
    [kv_heads, samples, head_dim]. The first ``skip_prefix`` positions of each
    chunk (the BOS attention sink) are left out because they are kept exact.
    """
    key_samples: list[list[torch.Tensor]] = []
    value_samples: list[list[torch.Tensor]] = []
    for chunk in chunks:
        with torch.inference_mode():
            outputs = model(input_ids=chunk[None].to(device), use_cache=True)
        keys, values = cache_tensors(outputs.past_key_values)
        if not key_samples:
            key_samples = [[] for _ in keys]
            value_samples = [[] for _ in values]
        for index, (key, value) in enumerate(zip(keys, values)):
            # [1, kv_heads, tokens, head_dim] -> [kv_heads, tokens, head_dim].
            key_samples[index].append(key[0, :, skip_prefix:].float().cpu())
            value_samples[index].append(value[0, :, skip_prefix:].float().cpu())
    return (
        [torch.cat(samples, dim=1) for samples in key_samples],
        [torch.cat(samples, dim=1) for samples in value_samples],
    )



def uniform_compress(
    tensors: list[torch.Tensor],
    round_trip,
    vector_bytes: int,
    exact_prefix: int,
    exact_recent: int,
) -> tuple[list[torch.Tensor], int]:
    """Give every middle token of one stream (K or V) the same code.

    ``tensors`` are per-layer [batch, kv_heads, tokens, head_dim];
    ``round_trip(tensor, layer)`` encodes and decodes one layer's slice. Sink
    and recent tokens stay exact. Returns rebuilt tensors and counted bytes
    (codes for middle tokens, FP16 for exact tokens).
    """
    batch, heads, tokens, head_dim = tensors[0].shape
    start = min(exact_prefix, tokens)
    end = max(start, tokens - exact_recent)
    rebuilt = []
    for layer, tensor in enumerate(tensors):
        out = tensor.clone()
        if end > start:
            out[..., start:end, :] = round_trip(tensor[..., start:end, :], layer)
        rebuilt.append(out)
    used = len(tensors) * batch * heads * (
        (end - start) * vector_bytes + (tokens - (end - start)) * head_dim * 2
    )
    return rebuilt, used
