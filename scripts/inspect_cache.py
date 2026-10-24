"""Load the model, prefill a prompt, and inspect its DynamicCache layout."""

from __future__ import annotations

import argparse

import torch

from common import DEFAULT_MODEL, kv_config, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model identifier or local model path.",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "KV-cache compression stores past attention keys and values more "
            "compactly while preserving the most useful context."
        ),
        help="Text used for the prefill pass.",
    )
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model)

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(
            **inputs,
            use_cache=True,
            return_dict=True,
        )

    cache = outputs.past_key_values
    print(f"model: {args.model}")
    print(f"device: {device}")
    print(f"dtype: {model.dtype}")
    print(f"config: {kv_config(model)}")
    print(f"tokens: {inputs['input_ids'].shape[1]}")
    print(f"cache type: {type(cache).__name__}")

    layers = getattr(cache, "layers", None)
    if layers is None:
        raise RuntimeError("The model did not return a cache with a .layers attribute.")
    if not layers:
        raise RuntimeError("The returned cache has no layers.")

    first_key = layers[0].keys
    first_value = layers[0].values
    print(f"layers: {len(layers)}")
    print(f"key shape: {tuple(first_key.shape)}")
    print(f"value shape: {tuple(first_value.shape)}")
    print(f"key dtype: {first_key.dtype}")
    print(f"value dtype: {first_value.dtype}")

    for index, layer in enumerate(layers):
        if layer.keys.shape != first_key.shape or layer.values.shape != first_value.shape:
            raise RuntimeError(f"Layer {index} has a different cache shape.")


if __name__ == "__main__":
    main()
