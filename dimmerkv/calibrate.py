"""Calibrate the codec for a model: cross-layer PCA and a ladder of bit rungs.

Port of ``scripts/fit_kvtc.py`` that works for any Hugging Face causal LM with
NeoX-style RoPE and also saves what an inference engine needs without the HF
model: the rotary tables (``rope``) and a ``fingerprint`` that ties stored
conversations to this exact calibration.

Per token, keys (RoPE removed) from all layers and KV heads form one vector,
likewise values. The covariance is accumulated over calibration chunks (the
first ``sinks`` tokens of each chunk excluded) and eigendecomposed. Then, on a
held sample, bits (0 -> 2 -> 4 -> 8) are added greedily to the group of
components that cuts the most error per byte; snapshots along this path at
the target sizes become the rungs.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from .codec import BIT_OPTIONS, checkpoint_fingerprint, quantize_groups, rung_bytes
from .rope import RoPE

TARGETS = (1 / 64, 1 / 48, 1 / 32, 1 / 24, 1 / 16, 1 / 12, 1 / 8, 1 / 6, 1 / 4, 1 / 3, 1 / 2)
FORMAT = "dimmerkv-calibration/1"


def build_ladder(sample: torch.Tensor, group: int, exact_bytes: int) -> tuple[torch.Tensor, list[int]]:
    """Greedy bit allocation path over groups; returns rungs [targets, groups] and their bytes."""
    groups = sample.shape[1] // group
    error = torch.zeros(groups, len(BIT_OPTIONS), dtype=torch.float64)
    for column, width in enumerate(BIT_OPTIONS):
        rebuilt = quantize_groups(sample, torch.full((groups,), width, device=sample.device), group)
        error[:, column] = (sample - rebuilt).square().view(sample.shape[0], groups, group).sum(dim=(0, 2)).double().cpu()
    option_bytes = torch.tensor([0] + [group * w / 8 + 4 for w in BIT_OPTIONS[1:]], dtype=torch.float64)

    level = torch.zeros(groups, dtype=torch.long)
    path = [(level.clone(), 0.0)]
    while (level < len(BIT_OPTIONS) - 1).any():
        movable = level < len(BIT_OPTIONS) - 1
        nxt = (level + 1).clamp_max(len(BIT_OPTIONS) - 1)
        gain = error[torch.arange(groups), level] - error[torch.arange(groups), nxt]
        cost = option_bytes[nxt] - option_bytes[level]
        score = torch.where(movable, gain / cost.clamp_min(1e-9), torch.tensor(-1.0, dtype=torch.float64))
        best = int(score.argmax())
        level[best] += 1
        path.append((level.clone(), float(option_bytes[level].sum())))

    rungs, sizes = [], []
    for target in TARGETS:
        fitting = [state for state, size in path if 0 < size <= target * exact_bytes]
        if fitting:
            bits = torch.tensor([BIT_OPTIONS[i] for i in fitting[-1]])
            size = rung_bytes(bits, group)
            if not sizes or size > sizes[-1]:
                rungs.append(bits)
                sizes.append(size)
    return torch.stack(rungs), sizes


class Accumulator:
    """Running mean/covariance and a held sample of token vectors for one stream."""

    def __init__(self, features: int, device, sample_every: int) -> None:
        self.sums = torch.zeros(features, dtype=torch.float64, device=device)
        self.products = torch.zeros(features, features, dtype=torch.float64, device=device)
        self.samples: list[torch.Tensor] = []
        self.count = 0
        self.sample_every = sample_every

    def add(self, matrix: torch.Tensor) -> None:
        self.sums += matrix.sum(dim=0).double()
        self.products += (matrix.T @ matrix).double()
        self.samples.append(matrix[:: self.sample_every].cpu())
        self.count += matrix.shape[0]

    def fit(self, group: int, device, log=print) -> dict:
        mean = self.sums / self.count
        covariance = self.products / self.count - torch.outer(mean, mean)
        self.products = None
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        except (torch.OutOfMemoryError, RuntimeError):
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance.cpu())
        del covariance
        eigenvalues, eigenvectors = eigenvalues.cpu(), eigenvectors.cpu()
        order = eigenvalues.argsort(descending=True)
        eigenvalues, basis = eigenvalues[order], eigenvectors[:, order].float()
        explained = (eigenvalues.clamp_min(0).cumsum(0) / eigenvalues.clamp_min(0).sum()).float()
        sample = torch.cat(self.samples).to(device)
        coefficients = (sample - mean.float().to(device)) @ basis.to(device)
        exact_bytes = sample.shape[1] * 2
        rungs, sizes = build_ladder(coefficients, group, exact_bytes)
        for bits, size in zip(rungs.to(device), sizes):
            error = (coefficients - quantize_groups(coefficients, bits, group)).square().sum()
            relative = float(error / (sample - mean.float().to(device)).square().sum())
            log(f"  rung {size / exact_bytes:6.3f} of FP16 ({exact_bytes / size:5.1f}x): in-sample rel err {relative:.4f}")
        return {"mean": mean.float().cpu(), "basis": basis, "rungs": rungs, "rung_bytes": sizes,
                "explained": explained}


def finalize(saved: dict) -> dict:
    """Stamp format and fingerprint on a calibration dict."""
    saved["format"] = FORMAT
    saved.pop("fingerprint", None)
    saved["fingerprint"] = checkpoint_fingerprint(saved)
    return saved


def calibration_chunks(tokenizer, chunk_tokens: int, max_chunks: int, text_file: str | None = None):
    """Token chunks starting with BOS, from a text file or WikiText-2 train."""
    if text_file:
        text = Path(text_file).read_text()
    else:
        from datasets import load_dataset

        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        text = "\n".join(line for line in dataset["text"] if line.strip())
    body = chunk_tokens - 1
    ids = tokenizer(text[: body * max_chunks * 8], add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    chunks = [torch.cat([torch.tensor([bos]), ids[start : start + body]])
              for start in range(0, ids.shape[0] - body + 1, body)][:max_chunks]
    if not chunks:
        raise RuntimeError("Calibration text produced no chunks; lower --chunk-tokens or add text.")
    return chunks


def stacked_cache(past_key_values) -> tuple[torch.Tensor, torch.Tensor]:
    """HF DynamicCache (batch 1) -> keys, values [layers, kv_heads, tokens, head_dim]."""
    layers = past_key_values.layers
    return (torch.stack([layer.keys[0] for layer in layers]), torch.stack([layer.values[0] for layer in layers]))


def calibrate(model_id: str, output: str | Path, *, chunk_tokens: int = 1024, chunks: int = 100, sinks: int = 4,
              group: int = 64, sample_every: int = 12, text_file: str | None = None, device: str | None = None,
              log=print) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float32 if device == "cpu" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    config = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    layers = config.num_hidden_layers
    heads = getattr(config, "num_key_value_heads", None) or config.num_attention_heads
    features = layers * heads * head_dim
    if features % group:
        raise ValueError(f"Features per token ({features}) must be a multiple of the group size ({group}).")
    rope = RoPE.from_hf_model(model).to(device)
    # float64 covariance (features^2 * 8 bytes per stream): on the GPU when it fits, else the CPU (MPS has no float64).
    accumulate_on = "cuda" if device == "cuda" and features * features * 8 * 2 < 0.5 * torch.cuda.mem_get_info()[0] else "cpu"
    streams = {kind: Accumulator(features, accumulate_on, sample_every) for kind in ("key", "value")}
    texts = calibration_chunks(tokenizer, chunk_tokens, chunks, text_file)
    cos, sin = rope.tables(0, chunk_tokens)
    for chunk in texts:
        with torch.inference_mode():
            outputs = model(input_ids=chunk[None].to(device), use_cache=True)
        keys, values = stacked_cache(outputs.past_key_values)
        length = keys.shape[2]
        keys = rope.derotate(keys, cos[:length], sin[:length])
        for kind, tensors in (("key", keys), ("value", values)):
            matrix = tensors.float().permute(2, 0, 1, 3).reshape(length, features)[sinks:]
            streams[kind].add(matrix.to(accumulate_on))
    log(f"calibration tokens: {streams['key'].count} ({len(texts)} chunks), features per stream: {features}")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    saved: dict = {"model": model_id, "layers": layers, "heads": heads, "head_dim": head_dim, "group": group,
                   "sinks": sinks, "calibration_tokens": streams["key"].count, "rope": rope.state()}
    for kind in ("key", "value"):
        log(f"{kind}:")
        saved[kind] = streams.pop(kind).fit(group, accumulate_on, log)
    finalize(saved)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, output)
    log(f"saved: {output} in {(time.time() - started) / 60:.1f} min")
    return saved


def convert_legacy(path: str | Path, output: str | Path, model_id: str | None = None) -> dict:
    """Add RoPE tables and a fingerprint to a ``scripts/fit_kvtc.py`` checkpoint."""
    from transformers import AutoConfig

    saved = torch.load(path, map_location="cpu", weights_only=False)
    config = AutoConfig.from_pretrained(model_id or saved["model"])
    saved["rope"] = RoPE.from_hf_config(config).state()
    finalize(saved)
    torch.save(saved, output)
    return saved

