"""Fit per-layer PCA bases for the model's cached keys and values.

The full ordered basis (all head_dim components) is saved so the allocator
and codec can take any prefix of it without refitting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from common import (
    DEFAULT_MODEL,
    collect_vectors,
    kv_config,
    load_model,
    wikitext_token_chunks,
)

REPORT_COMPONENTS = (8, 16, 32, 48)


def fit_basis(samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit a centered PCA basis to [groups, samples, head_dim] vectors.

    Returns the basis [groups, head_dim, head_dim] (rows ordered by variance),
    the mean [groups, head_dim] and the cumulative explained variance
    [groups, head_dim].
    """
    mean = samples.mean(dim=1)
    centered = samples - mean[:, None]
    _, singular_values, right_singular_vectors = torch.linalg.svd(
        centered, full_matrices=False
    )
    energy = singular_values.square()
    explained = energy.cumsum(dim=-1) / energy.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return right_singular_vectors, mean, explained


def relative_mse(samples: torch.Tensor, basis: torch.Tensor, mean: torch.Tensor, k: int) -> float:
    centered = samples - mean[:, None]
    top = basis[:, :k]
    reconstructed = (centered @ top.transpose(-1, -2)) @ top + mean[:, None]
    return float((samples - reconstructed).square().mean() / samples.square().mean().clamp_min(1e-12))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--chunks", type=int, default=32)
    parser.add_argument(
        "--skip-prefix",
        type=int,
        default=1,
        help="Leading positions per chunk left out of the fit (BOS sink, kept exact).",
    )
    parser.add_argument(
        "--per-head",
        action="store_true",
        help="Fit one basis per (layer, KV head) instead of one per layer.",
    )
    parser.add_argument("--output", type=Path, default=Path("checkpoints/pca_basis.pt"))
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model)
    config = kv_config(model)
    chunks = wikitext_token_chunks(tokenizer, "train", args.chunk_tokens, args.chunks)
    keys, values = collect_vectors(model, chunks, device, args.skip_prefix)

    basis: dict[str, object] = {
        "model": args.model,
        "components": config.head_dim,
        "head_dim": config.head_dim,
        "kv_heads": config.kv_heads,
        "per_head": args.per_head,
        "skip_prefix": args.skip_prefix,
        "calibration": f"wikitext-2-raw-v1 train, {len(chunks)} x {args.chunk_tokens} tokens",
        "layers": [],
    }
    errors = {k: [] for k in REPORT_COMPONENTS}
    for layer_index, (layer_keys, layer_values) in enumerate(zip(keys, values)):
        entry: dict[str, object] = {"layer": layer_index}
        line = [f"layer {layer_index:02d}:"]
        for kind, samples in (("key", layer_keys), ("value", layer_values)):
            if not args.per_head:
                # [kv_heads, samples, head_dim] -> [1, kv_heads * samples, head_dim].
                samples = samples.flatten(0, 1)[None]
            stream_basis, stream_mean, explained = fit_basis(samples)
            entry[f"{kind}_basis"] = stream_basis
            entry[f"{kind}_mean"] = stream_mean
            entry[f"{kind}_explained"] = explained
            for k in REPORT_COMPONENTS:
                errors[k].append(relative_mse(samples, stream_basis, stream_mean, k))
            line.append(f"{kind} explained@16={float(explained[:, 15].mean()):.3f}")
        basis["layers"].append(entry)
        print(" ".join(line))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(basis, args.output)
    print(f"saved: {args.output}")
    print(f"per head: {args.per_head}")
    print(f"vectors per layer/stream: {keys[0].shape[0] * keys[0].shape[1]}")
    for k in REPORT_COMPONENTS:
        print(f"in-sample mean relative MSE @ {k:2d} components: {sum(errors[k]) / len(errors[k]):.4f}")


if __name__ == "__main__":
    main()
