"""Train per-layer Lexico dictionaries for cached keys and values.

Alternates batched OMP sparse coding with an Adam step on the dictionary,
renormalizing atoms to unit length. Saves after every layer so a partial
run is still usable.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from common import DEFAULT_MODEL, collect_vectors, kv_config, load_model, wikitext_token_chunks
from lexico import bytes_per_vector, omp, reconstruct

REPORT_SPARSITY = (4, 8, 16, 32)


def relative_mse(vectors: torch.Tensor, dictionary: torch.Tensor, sparsity: int) -> float:
    indices, coefficients = omp(vectors, dictionary, sparsity)
    error = (vectors - reconstruct(indices, coefficients, dictionary)).square().mean()
    return float(error / vectors.square().mean().clamp_min(1e-12))


def train_dictionary(
    train: torch.Tensor, atoms: int, sparsity: int, steps: int, batch: int, lr: float, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    start = train[torch.randperm(train.shape[0], generator=generator)[:atoms]]
    dictionary = torch.nn.functional.normalize(start, dim=1).T.contiguous().requires_grad_(True)
    optimizer = torch.optim.Adam([dictionary], lr=lr)
    scale = train.square().mean()
    for _ in range(steps):
        sample = train[torch.randint(train.shape[0], (batch,), generator=generator)]
        with torch.no_grad():
            indices, coefficients = omp(sample, dictionary, sparsity)
        loss = (sample - reconstruct(indices, coefficients, dictionary)).square().mean() / scale
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            dictionary.copy_(torch.nn.functional.normalize(dictionary, dim=0))
    return dictionary.detach()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--chunks", type=int, default=32)
    parser.add_argument("--skip-prefix", type=int, default=1)
    parser.add_argument("--atoms", type=int, default=4096)
    parser.add_argument("--sparsity", type=int, default=16, help="Atoms per vector while training.")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/lexico_dict.pt"))
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model)
    config = kv_config(model)
    train_keys, train_values = collect_vectors(
        model, wikitext_token_chunks(tokenizer, "train", args.chunk_tokens, args.chunks), device, args.skip_prefix
    )
    test_keys, test_values = collect_vectors(
        model, wikitext_token_chunks(tokenizer, "test", args.chunk_tokens, 2), device, args.skip_prefix
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    saved: dict[str, object] = {
        "model": args.model,
        "head_dim": config.head_dim,
        "kv_heads": config.kv_heads,
        "atoms": args.atoms,
        "train_sparsity": args.sparsity,
        "skip_prefix": args.skip_prefix,
        "layers": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"bytes/vector at sparsity {REPORT_SPARSITY}: {[bytes_per_vector(s) for s in REPORT_SPARSITY]}"
          f" (FP16 vector: {config.head_dim * 2})", flush=True)
    for layer in range(config.layers):
        entry: dict[str, object] = {"layer": layer}
        line = [f"layer {layer:02d}:"]
        for kind, train, test in (
            ("key", train_keys[layer], test_keys[layer]),
            ("value", train_values[layer], test_values[layer]),
        ):
            started = time.time()
            # Pool all KV heads: one dictionary per layer and stream.
            dictionary = train_dictionary(
                train.flatten(0, 1).to(device), args.atoms, args.sparsity,
                args.steps, args.batch, args.lr, args.seed + layer,
            )
            entry[f"{kind}_dict"] = dictionary.cpu()
            held_out = test.flatten(0, 1).to(device)
            errors = [relative_mse(held_out, dictionary, s) for s in REPORT_SPARSITY]
            line.append(
                f"{kind} held-out relMSE " + "/".join(f"{e:.3f}" for e in errors)
                + f" ({time.time() - started:.0f}s)"
            )
        saved["layers"].append(entry)
        torch.save(saved, args.output)
        print(" ".join(line), flush=True)
    print(f"saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
