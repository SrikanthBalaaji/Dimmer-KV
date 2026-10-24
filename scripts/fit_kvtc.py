"""Calibrate the kvtc transform: cross-layer PCA and a ladder of bit rungs.

Per token, keys (RoPE removed) from all layers and KV heads form one vector,
likewise values. The covariance is accumulated over WikiText-2 train chunks
(the first ``--sinks`` tokens of each chunk excluded) and eigendecomposed.
Then, on a held sample, bits (0 -> 2 -> 4 -> 8) are added greedily to the
group of components that cuts the most error per byte; snapshots along this
path at the target sizes become the rungs.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from common import DEFAULT_MODEL, cache_tensors, kv_config, load_model, wikitext_token_chunks
from kvtc import BIT_OPTIONS, derotate, quantize_groups, rope_tables, rung_bytes, token_matrix

TARGETS = (1 / 64, 1 / 48, 1 / 32, 1 / 24, 1 / 16, 1 / 12, 1 / 8, 1 / 6, 1 / 4, 1 / 3, 1 / 2)


def build_ladder(sample: torch.Tensor, group: int, exact_bytes: int) -> tuple[torch.Tensor, list[int]]:
    """Greedy bit allocation path over groups; returns rungs [targets, groups] and their bytes."""
    groups = sample.shape[1] // group
    # Error of each group at each bit option, summed over the sample.
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chunk-tokens", type=int, default=1024)
    parser.add_argument("--chunks", type=int, default=100)
    parser.add_argument("--sinks", type=int, default=4)
    parser.add_argument("--group", type=int, default=64)
    parser.add_argument("--sample-every", type=int, default=12, help="Every n-th token goes to the ladder sample.")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/kvtc.pt"))
    args = parser.parse_args()

    started = time.time()
    tokenizer, model, device = load_model(args.model)
    config = kv_config(model)
    features = config.layers * config.kv_heads * config.head_dim
    if features % args.group:
        raise ValueError("Feature count must be a multiple of the group size.")
    sums = {kind: torch.zeros(features, dtype=torch.float64, device=device) for kind in ("key", "value")}
    products = {kind: torch.zeros(features, features, dtype=torch.float64, device=device) for kind in ("key", "value")}
    samples = {kind: [] for kind in ("key", "value")}
    count = 0
    chunks = wikitext_token_chunks(tokenizer, "train", args.chunk_tokens, args.chunks)
    cos, sin = rope_tables(model, args.chunk_tokens, device)
    for chunk in chunks:
        with torch.inference_mode():
            outputs = model(input_ids=chunk[None].to(device), use_cache=True)
            keys, values = cache_tensors(outputs.past_key_values)
            length = keys[0].shape[2]
            matrices = {
                "key": token_matrix([derotate(k, cos[:length], sin[:length]) for k in keys])[args.sinks:],
                "value": token_matrix(values)[args.sinks:],
            }
            for kind, matrix in matrices.items():
                sums[kind] += matrix.sum(dim=0).double()
                products[kind] += (matrix.T @ matrix).double()
                samples[kind].append(matrix[:: args.sample_every].cpu())
        count += matrices["key"].shape[0]
    print(f"calibration tokens: {count} ({len(chunks)} chunks), features per stream: {features}", flush=True)
    del model
    torch.cuda.empty_cache()

    saved: dict[str, object] = {
        "model": args.model, "layers": config.layers, "heads": config.kv_heads, "head_dim": config.head_dim,
        "group": args.group, "sinks": args.sinks, "calibration_tokens": count,
    }
    exact_bytes = features * 2
    for kind in ("key", "value"):
        mean = sums[kind] / count
        covariance = products[kind] / count - torch.outer(mean, mean)
        del products[kind]
        # GPU eigendecomposition when it fits (large models have 28-32K features per token).
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        except torch.OutOfMemoryError:
            covariance = covariance.cpu()
            torch.cuda.empty_cache()
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        eigenvalues, eigenvectors = eigenvalues.cpu(), eigenvectors.cpu()
        del covariance
        order = eigenvalues.argsort(descending=True)
        eigenvalues, basis = eigenvalues[order], eigenvectors[:, order].float()
        explained = (eigenvalues.clamp_min(0).cumsum(0) / eigenvalues.clamp_min(0).sum()).float()
        sample = torch.cat(samples[kind]).to(device)
        coefficients = (sample - mean.float().to(device)) @ basis.to(device)
        rungs, sizes = build_ladder(coefficients, args.group, exact_bytes)
        relative = []
        for bits in rungs.to(device):
            rebuilt = quantize_groups(coefficients, bits, args.group)
            relative.append(float((coefficients - rebuilt).square().sum() / (sample - mean.float().to(device)).square().sum()))
        saved[kind] = {"mean": mean.float().cpu(), "basis": basis, "rungs": rungs, "rung_bytes": sizes,
                       "explained": explained}
        print(f"{kind}: explained@64/256/1024 = {explained[63]:.3f}/{explained[255]:.3f}/{explained[1023]:.3f}")
        for size, error in zip(sizes, relative):
            print(f"  rung {size / exact_bytes:6.3f} of FP16 ({exact_bytes / size:5.1f}x): in-sample rel err {error:.4f}")
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, args.output)
    print(f"saved: {args.output} in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
