# DimmerKV plugin: compressed chat history for vLLM and Hugging Face

The `dimmerkv` package turns the research code in `scripts/` into something an inference engine can
load. It keeps every conversation's KV cache **compressed (about 3–5% of FP16) between turns and on
disk**. When the user comes back, the next turn, or a conversation from last week, restores the
history from the store instead of prefilling it again.

```
turn ends ──► copy the KV of the tokens this turn added ──► DimmerKV codec (per-token precision)
                                                            └─► segment appended to the conversation's chain on disk
next prompt ──► longest stored prefix (hash index) ──► decode into the KV cache ──► prefill only the new tokens
```

## Install and calibrate (once per model)

```bash
pip install -e ".[hf,calibrate]"          # add ",vllm" on a CUDA machine
dimmerkv calibrate --model meta-llama/Llama-3.2-1B-Instruct --output checkpoints/llama-3.2-1b.pt
```

Calibration fits the cross-layer PCA basis and the rung ladder on WikiText-2. It takes about 1.5 minutes
for Qwen2.5-0.5B on an M4 Mac; pass `--text-file` to use your own text. To convert a file you already
made with `scripts/fit_kvtc.py`, run `dimmerkv convert checkpoints/kvtc.pt --output checkpoints/llama-3.2-1b.pt`.

## vLLM

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct --gpu-memory-utilization 0.7 \
  --kv-transfer-config '{"kv_connector": "DimmerKVConnector", "kv_role": "kv_both",
    "kv_connector_extra_config": {"checkpoint": "checkpoints/llama-3.2-1b.pt",
                                  "store_path": "/var/lib/dimmerkv", "budget": 0.03}}'
```

The package registers `DimmerKVConnector` through vLLM's `vllm.general_plugins` entry point. You can
also add `"kv_connector_module_path": "dimmerkv.integrations.vllm.connector"` to the config. The
connector:

- **saves** when a request finishes. vLLM keeps the request's blocks until the worker has copied out
  the tokens it computed (prompt and answer). Compression then runs on a background CUDA stream, and
  only tokens the store does not already have are compressed.
- **loads** when a new prompt starts with a stored conversation that vLLM's GPU prefix cache no longer
  holds. The longest stored prefix is decoded straight into the request's paged blocks, and only the
  rest is prefilled. Blocks vLLM still has exact are left alone. If a load fails, its blocks are
  reported to vLLM, which recomputes them.

`kv_connector_extra_config`:

| key | default | meaning |
|---|---|---|
| `checkpoint` | required | file from `dimmerkv calibrate` |
| `store_path` | `~/.cache/dimmerkv` | store directory, shared by the scheduler and worker processes |
| `budget` | `0.03` | stored size as a fraction of the FP16 KV of each segment |
| `recent` / `floor_share` | `32` / `1.0` | exact tokens at each segment's end (at most half the budget), and the uniform floor |
| `min_tokens` | `256` | shorter conversations are not stored |
| `ram_cache_mb` / `max_disk_gb` | `512` / no limit | worker RAM cache of blobs / disk limit (least recently used conversations go first) |
| `max_rung_fraction` | none | drop rungs finer than this fraction to shrink the basis held on the GPU |
| `codec_device` | the KV cache's device | where compression runs (`cpu` is possible but slow) |
| `save` / `load` | from `kv_role` | `kv_producer` saves, `kv_consumer` loads, `kv_both` does both |

See `examples/vllm_chat.py` for an offline two-turn run that clears the GPU prefix cache between
turns.

## Hugging Face transformers

```python
from dimmerkv import DimmerKV
from dimmerkv.integrations.hf import CompressedChat

chat = CompressedChat(model, tokenizer, DimmerKV.open("checkpoints/qwen2.5-0.5b.pt", "~/.cache/dimmerkv",
                                                      device="cuda", budget=0.05))
answer = chat.reply(messages, max_new_tokens=256)   # restores history, prefills only the new message, stores the turn
```

For a single cache you manage yourself, use `compress_cache(codec, cache)` / `restore_cache(codec, blob, model)`.
With any other engine, call `DimmerKV.lookup / load / save` on stacked `[layers, kv_heads, tokens, head_dim]` tensors.

## How storage works

- **Codec** (`dimmerkv/codec.py`): a port of `scripts/kvtc.py` and `scripts/kvtc_storage.py`. Its payload
  is byte-identical to the scripts' output (tested). Changes: RoPE comes from the calibration file, so no
  HF model is needed; the basis keeps only the columns some rung uses; a segment can start mid-conversation.
- **Segments** (`dimmerkv/store.py`): each turn stores only the tokens it added, chained to the previous
  segment. Old turns are never re-encoded, so quality does not drift as a chat grows, and compression
  costs only the new tokens.
- **Prefix index**: a chained hash every 16 tokens, kept in SQLite, so an edited earlier message (a branch)
  still reuses the shared part. LoRA adapters get their own namespace, and each calibration file has its
  own fingerprint.

## What has been checked

| check | where | result |
|---|---|---|
| Payload identical to `scripts/kvtc.py` + `kvtc_storage.py`, decode identical | `tests/test_codec.py` | 3 budget/floor settings, byte-equal |
| Store: prefix lookup, branches, segment chains, eviction, two processes | `tests/test_store.py` | pass |
| vLLM connector logic: save on finish, block release, load into paged blocks, partial local hits, load failures, both cache layouts | `tests/test_vllm_connector.py` (stub vLLM modules, CPU tensors) | pass |
| HF chat, Qwen2.5-0.5B-Instruct, 3.2K-token history, 6 facts asked after compression (M4 Mac, MPS) | `examples/eval_hf_chat.py` | full 6/6; **5%: 6/6 at 22× smaller**, restore 91 ms; 3%: 4/6 at 36×; 10%: 5/6 |
| HF chat, 3 short turns | `examples/hf_chat.py` | 256 tokens stored in 180 KB (5.7% of FP16); name recalled in turn 3 |

**Not yet checked: running inside a real vLLM server.** The connector was written against vLLM's
source (main, 2026-10-26, commit `d26d5db`), but nothing here has a CUDA GPU. Before relying on it,
run `examples/vllm_chat.py` on a GPU machine. The connector API is still marked experimental in vLLM,
so a different vLLM version may need small changes.

## Limits

- vLLM: tensor and pipeline parallel size 1 (the PCA basis spans every layer and KV head), one
  full-attention KV cache group (no sliding-window or hybrid models), no FP8 KV cache, text-only
  requests.
- Restored KV is lossy, and once loaded it is also served from vLLM's GPU prefix cache to other
  requests with the same prefix.
- The basis sits in GPU memory. Llama-3.2-1B needs 2 × 8192² floats (≈0.5 GB in FP32); 8B models need
  several GB unless `max_rung_fraction` trims it. Leave headroom with `--gpu-memory-utilization`.
- The connector compresses without the question (`dkvtc@b:floor1`). Question scores, question-time
  refinement and attention over the codes are still only in `scripts/`.

## Next steps

- **SGLang**: a HiCache storage backend. The codec is per token, so it can compress SGLang's pages directly.
- **LMCache**: a serializer, which would cover vLLM and SGLang through LMCache.
- **Tensor parallel**: fit one basis per TP rank over that rank's KV heads.
- **Hot/cold tiers and question-time refinement** from `scripts/progressive.py`, on top of the segment store.
