"""Round-trip test for the PCA codec on held-out WikiText cache vectors."""

from __future__ import annotations

import argparse

import torch

from common import DEFAULT_MODEL, cache_tensors, kv_config, load_model, wikitext_token_chunks
from pca_codec import PCACodec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--basis", default="checkpoints/pca_basis.pt")
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--components", type=int, nargs="+", default=[16, 32, 48])
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model)
    config = kv_config(model)
    codec = PCACodec(args.basis, device)
    if codec.model != args.model or codec.head_dim != config.head_dim or codec.layers != config.layers:
        raise RuntimeError(f"Basis was fit for {codec.model}; refit for {args.model}.")

    # The test split is never seen by fit_pca_basis.py.
    chunks = wikitext_token_chunks(tokenizer, "test", args.chunk_tokens, args.chunks)
    caches = []
    for chunk in chunks:
        with torch.inference_mode():
            outputs = model(input_ids=chunk[None].to(device), use_cache=True)
        caches.append(cache_tensors(outputs.past_key_values))

    tokens = sum(chunk.shape[0] for chunk in chunks)
    print(f"basis: {args.basis} (per head: {codec.per_head}, skip prefix: {codec.skip_prefix})")
    print(f"held-out tokens: {tokens} in {len(chunks)} chunks")
    print(f"{'k':>3} {'bits':>4} {'ratio':>6} | {'key relMSE':>10} {'cos p5':>7} {'cos min':>7} | "
          f"{'val relMSE':>10} {'cos p5':>7} {'cos min':>7}")
    for components in args.components:
        for bits in (16, 8):
            stats = {"key": ([], []), "value": ([], [])}
            compressed_bytes = 0
            for keys, values in caches:
                new_keys, new_values, used = codec.compress_cache(keys, values, components, bits)
                compressed_bytes += used
                for kind, originals, rebuilt in (("key", keys, new_keys), ("value", values, new_values)):
                    for original, reconstructed in zip(originals, rebuilt):
                        if original.shape != reconstructed.shape:
                            raise RuntimeError("Round trip changed the cache shape.")
                        if not torch.isfinite(reconstructed).all():
                            raise RuntimeError("Round trip produced non-finite values.")
                        # Only the compressed middle tokens; the sink stays exact.
                        a = original[..., codec.skip_prefix :, :].float()
                        b = reconstructed[..., codec.skip_prefix :, :].float()
                        errors, cosines = stats[kind]
                        errors.append(float((a - b).square().mean() / a.square().mean().clamp_min(1e-12)))
                        cosines.append(torch.nn.functional.cosine_similarity(a, b, dim=-1).flatten())
            ratio = compressed_bytes / (tokens * config.fp16_bytes_per_token())
            row = [f"{components:>3} {bits:>4} {ratio:>6.3f}"]
            for kind in ("key", "value"):
                errors, cosines = stats[kind]
                cos = torch.cat(cosines)
                row.append(
                    f"{sum(errors) / len(errors):>10.4f} {float(cos.quantile(0.05)):>7.3f} "
                    f"{float(cos.min()):>7.3f}"
                )
            print(" | ".join(row))


if __name__ == "__main__":
    main()
