"""Compare PCA and Lexico codecs on held-out cache vectors: error, bytes, time.

Encode and decode are timed separately (CUDA-synchronized, after a warm-up)
over every layer and both streams, and reported per 1K cached tokens.
"""

from __future__ import annotations

import argparse
import time

import torch

import lexico
import pca_codec
from common import DEFAULT_MODEL, cache_tensors, kv_config, load_model, wikitext_token_chunks


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--basis", default="checkpoints/pca_basis.pt")
    parser.add_argument("--dictionary", default="checkpoints/lexico_dict.pt")
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--chunks", type=int, default=4)
    parser.add_argument("--pca", type=int, nargs="*", default=[8, 16, 32, 48])
    parser.add_argument("--lexico", type=int, nargs="*", default=[4, 5, 8, 11, 16])
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model)
    config = kv_config(model)
    pca = pca_codec.PCACodec(args.basis, device)
    lex = lexico.LexicoCodec(args.dictionary, device)
    for codec in (pca, lex):
        if codec.model != args.model:
            raise RuntimeError(f"Codec was fit for {codec.model}; refit for {args.model}.")

    caches = []
    for chunk in wikitext_token_chunks(tokenizer, "test", args.chunk_tokens, args.chunks):
        with torch.inference_mode():
            outputs = model(input_ids=chunk[None].to(device), use_cache=True)
        keys, values = cache_tensors(outputs.past_key_values)
        # Middle tokens only; the sink is kept exact by both codecs.
        caches.append(([k[..., 1:, :] for k in keys], [v[..., 1:, :] for v in values]))
    tokens = sum(keys[0].shape[-2] for keys, _ in caches)

    configs = [
        (f"pca k={k} int8", pca_codec.bytes_per_vector(k, 8),
         lambda x, layer, kind, k=k: pca.encode_codes(x, layer, kind, k, 8), pca.decode_codes)
        for k in args.pca
    ] + [
        (f"lexico s={s}", lexico.bytes_per_vector(s),
         lambda x, layer, kind, s=s: lex.encode_codes(x, layer, kind, s), lex.decode_codes)
        for s in args.lexico
    ]

    print(f"device: {device}, held-out tokens: {tokens} (sink excluded), "
          f"{config.layers} layers x {config.kv_heads} KV heads x K+V")
    print(f"{'codec':>16} {'B/vec':>5} {'ratio':>6} | {'key err':>7} {'val err':>7} {'val cos p5':>10} | "
          f"{'enc ms/1K tok':>13} {'dec ms/1K tok':>13}")
    for name, vector_bytes, encode, decode in configs:
        with torch.inference_mode():
            encode(caches[0][0][0], 0, "key")  # Warm-up.
            encode_time = decode_time = 0.0
            errors = {"key": [], "value": []}
            cosines = []
            for keys, values in caches:
                for kind, tensors in (("key", keys), ("value", values)):
                    for layer, tensor in enumerate(tensors):
                        synchronize(device)
                        started = time.perf_counter()
                        codes = encode(tensor, layer, kind)
                        synchronize(device)
                        encode_time += time.perf_counter() - started
                        started = time.perf_counter()
                        rebuilt = decode(codes, layer, kind, tensor.dtype)
                        synchronize(device)
                        decode_time += time.perf_counter() - started
                        a, b = tensor.float(), rebuilt.float()
                        errors[kind].append(float((a - b).square().mean() / a.square().mean()))
                        if kind == "value":
                            cosines.append(torch.nn.functional.cosine_similarity(a, b, dim=-1).flatten())
        print(
            f"{name:>16} {vector_bytes:>5} {vector_bytes / (config.head_dim * 2):>6.3f} | "
            f"{sum(errors['key']) / len(errors['key']):>7.4f} "
            f"{sum(errors['value']) / len(errors['value']):>7.4f} "
            f"{float(torch.cat(cosines).quantile(0.05)):>10.3f} | "
            f"{encode_time / tokens * 1000 * 1000:>13.1f} {decode_time / tokens * 1000 * 1000:>13.1f}"
        )


if __name__ == "__main__":
    main()
