# DimmerKV plugin guide

This guide covers the `dimmerkv` package: what it adds to the research code in `scripts/`, how to
install and run it, how it works, how far it has been tested, and what to build next.

- [1. What this is](#1-what-this-is)
- [2. What was built](#2-what-was-built)
- [3. Quickstart](#3-quickstart)
- [4. Calibrating a model](#4-calibrating-a-model)
- [5. Running with Hugging Face transformers](#5-running-with-hugging-face-transformers)
- [6. Running with vLLM](#6-running-with-vllm)
- [7. Python API](#7-python-api)
- [8. How it works](#8-how-it-works)
- [9. Testing and results](#9-testing-and-results)
- [10. Troubleshooting](#10-troubleshooting)
- [11. Limits](#11-limits)
- [12. Future improvements](#12-future-improvements)

---

## 1. What this is

DimmerKV (see the [README](README.md) and [FINDINGS.md](FINDINGS.md)) compresses an LLM's KV cache
to 3–5% of its FP16 size. Every token keeps some precision; nothing is deleted. Until now it existed
as research scripts that work only with a Hugging Face model in one process.

The `dimmerkv` package makes it usable in an inference engine, for two jobs:

1. **Chat.** Between turns, a conversation's KV cache is kept compressed instead of taking GPU
   memory or being thrown away. When the next message arrives, the history is restored from the
   compressed copy and only the new message is prefilled.
2. **Old conversations.** Compressed conversations are stored on disk and indexed by their tokens.
   A user returning hours or days later, a shared system prompt, or a long document asked about
   again all restore from the store instead of re-reading the text.

```
turn ends    ──► copy the KV of the tokens this turn added ──► DimmerKV codec (precision per token)
                                                               └─► segment appended to the conversation on disk
next prompt  ──► longest stored prefix (hash index) ──► decode into the KV cache ──► prefill only the new tokens
```

Supported today: **vLLM** (as a KV connector), **Hugging Face transformers** (a `CompressedChat`
wrapper), and **any other engine** that can hand over a sequence's keys and values as tensors.

---

## 2. What was built

### 2.1 Package layout

| file | what it does |
|---|---|
| `dimmerkv/codec.py` | `DimmerKVCodec`: compresses one sequence's keys/values to bytes and back. Port of `scripts/kvtc.py` (`compress`) and `scripts/kvtc_storage.py` (`encode_bytes`/`decode_bytes`) |
| `dimmerkv/rope.py` | `RoPE`: removes and re-applies rotary position embeddings using tables saved in the calibration file, so no HF model object is needed |
| `dimmerkv/store.py` | `ConversationStore`: blobs on disk, a SQLite prefix index, segment chains, a RAM cache, size-based eviction |
| `dimmerkv/engine.py` | `DimmerKV`: codec + store behind `lookup` / `load` / `save`; `CompressionSettings` |
| `dimmerkv/calibrate.py` | fits the PCA basis and rung ladder for any HF causal LM; converts old `fit_kvtc.py` files |
| `dimmerkv/cli.py` | the `dimmerkv` command (`calibrate`, `convert`, `stats`) |
| `dimmerkv/integrations/hf.py` | `CompressedChat` for transformers, plus `compress_cache` / `restore_cache` |
| `dimmerkv/integrations/vllm/connector.py` | `DimmerKVConnector`, a vLLM v1 KV connector |
| `dimmerkv/integrations/vllm/paged.py` | copies one sequence's KV in and out of vLLM's paged cache blocks (no vLLM import, so it is testable anywhere) |
| `dimmerkv/integrations/vllm/__init__.py` | vLLM plugin entry point that registers the connector name |
| `tests/` | codec parity with the scripts, store behaviour, vLLM connector lifecycle |
| `examples/` | `hf_chat.py` (multi-turn demo), `eval_hf_chat.py` (accuracy check), `vllm_chat.py` (vLLM two-turn run) |
| `pyproject.toml` | makes `pip install -e .` work, adds the `dimmerkv` command and the vLLM plugin entry point |

`scripts/` is unchanged; the research code and its results stay as they were.

### 2.2 What changed compared with the scripts

| scripts | package | why |
|---|---|---|
| needs the HF model for RoPE (`model.model.rotary_emb`) | RoPE tables saved in the calibration file | a vLLM worker has no HF model |
| per-layer lists `[1, heads, tokens, dim]` | stacked `[layers, heads, tokens, dim]` | one format any engine can produce |
| full PCA basis kept | only the columns some rung uses | less memory; same bytes |
| rung errors over all tokens at once | computed in chunks of 2,048 tokens | bounded temporary memory on long contexts |
| always starts at position 0 with 4 sinks | `position_offset` for segments that start mid-conversation | append-only storage of chat turns |
| budget below the coarsest rung raises an error | falls back to the coarsest rung | short sequences still store |
| raw byte payload | header (magic, version, calibration fingerprint, position offset) + DEFLATE of the same payload | blobs are self-describing and cannot be decoded with the wrong calibration |

The compressed payload is **byte-for-byte identical** to the scripts' output for the same input and
settings (`tests/test_codec.py`). The package therefore inherits the scripts' measured quality.

### 2.3 Default setting

The package defaults to the repository's recommended setting without question scores:
`dkvtc@0.03:floor1`, meaning a 3% budget and per-token precision on top of kvtc's uniform floor.
The first 4 tokens and the last 32 tokens of each stored segment stay exact. The exact recent window
is capped at half of the segment's budget, so short chat turns are not stored mostly in FP16
(see [8.3](#83-segments-why-chat-turns-are-stored-append-only)).

---

## 3. Quickstart

### On a Mac or any machine without an NVIDIA GPU (Hugging Face path)

```bash
git clone -b plugin https://github.com/SrikanthBalaaji/Dimmer-KV.git
cd Dimmer-KV
python -m venv .venv && source .venv/bin/activate
pip install -e ".[hf,calibrate,test]"

dimmerkv calibrate --model Qwen/Qwen2.5-0.5B-Instruct --output checkpoints/qwen2.5-0.5b.pt   # ~2 min, once
python examples/hf_chat.py --checkpoint checkpoints/qwen2.5-0.5b.pt                           # 3-turn chat
python -m pytest -q tests                                                                     # unit tests
```

### On an NVIDIA GPU machine (vLLM)

```bash
pip install -e ".[hf,calibrate,vllm]"
hf auth login                                     # Llama models are gated
dimmerkv calibrate --model meta-llama/Llama-3.2-1B-Instruct --output checkpoints/llama-3.2-1b.pt
python examples/vllm_chat.py --checkpoint checkpoints/llama-3.2-1b.pt
```

Calibration files are not in git (`checkpoints/` is ignored). Make one per model on the machine
that will use it, or copy the `.pt` file across.

---

## 4. Calibrating a model

The codec needs a one-time calibration per model: a PCA basis that decorrelates all layers' keys
(and values) of a token, and a ladder of bit allocations (rungs) from coarse to fine.

```bash
dimmerkv calibrate --model <HF model id or path> --output checkpoints/<name>.pt
```

| option | default | meaning |
|---|---|---|
| `--chunks` / `--chunk-tokens` | 100 / 1024 | calibration text: number of chunks and tokens per chunk |
| `--text-file` | WikiText-2 | calibrate on your own text (also avoids the `datasets` dependency) |
| `--group` | 64 | PCA components per quantization group; layers × KV heads × head_dim must be a multiple |
| `--sinks` | 4 | leading tokens kept exact (and left out of calibration) |
| `--device` | best available | `cuda`, `mps` or `cpu` |

The command prints each rung's size and error. Qwen2.5-0.5B (3,072 features per token) took 1.6 minutes
on an M4 Mac, and its file is 75 MB. Memory grows with the square of the feature count: Llama-3.2-1B has
8,192 features per token, Llama-3.1-8B has 32,768. Large models need a GPU and plenty of RAM for the
covariance (8 bytes × features² per stream).

To reuse files made by `scripts/fit_kvtc.py`, add the RoPE tables and fingerprint:

```bash
dimmerkv convert checkpoints/kvtc.pt --output checkpoints/llama-3.2-1b.pt
```

**Which models work:** decoder-only models with NeoX-style RoPE and full attention, such as Llama,
Qwen2/2.5/3 and Mistral (without sliding window). See [Limits](#11-limits).

Calibrating on text close to your traffic (for example, chat transcripts) may compress better than
WikiText; this has not been measured yet.

---

## 5. Running with Hugging Face transformers

### 5.1 Multi-turn chat

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dimmerkv import DimmerKV
from dimmerkv.integrations.hf import CompressedChat

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to("cuda")   # or "mps"

kv = DimmerKV.open("checkpoints/qwen2.5-0.5b.pt", "~/.cache/dimmerkv", device="cuda", budget=0.05)
chat = CompressedChat(model, tokenizer, kv)

messages = [{"role": "user", "content": "My name is Priya. Plan a day in Kyoto."}]
answer = chat.reply(messages, max_new_tokens=200, do_sample=False)
messages += [{"role": "assistant", "content": answer}, {"role": "user", "content": "What's my name?"}]
answer = chat.reply(messages, max_new_tokens=50)
print(chat.last_stats)   # {'prompt_tokens': ..., 'restored_tokens': ..., 'prefilled_tokens': ..., 'stored_segment': ...}
```

`reply` does three things:

1. It looks up the longest stored prefix of the prompt and decodes it into a `DynamicCache`.
2. It calls `model.generate`, which prefills only the uncached tokens. Extra keyword arguments go
   to `generate`.
3. It compresses the tokens this turn added (the rest of the prompt plus the answer) and stores
   them as a new segment. Pass `save=False` to skip this.

Nothing stays on the GPU between turns, and the store survives restarts.

### 5.2 One cache, storage managed by you

```python
from dimmerkv import DimmerKVCodec
from dimmerkv.integrations.hf import compress_cache, restore_cache

codec = DimmerKVCodec("checkpoints/qwen2.5-0.5b.pt", device="cuda")
blob = compress_cache(codec, outputs.past_key_values, budget=0.03)   # bytes: write anywhere
cache = restore_cache(codec, blob, model)                            # DynamicCache, ready for generate()
```

### 5.3 Examples

```bash
python examples/hf_chat.py --checkpoint checkpoints/qwen2.5-0.5b.pt [--budget 0.05] [--store DIR]
python examples/eval_hf_chat.py --checkpoint checkpoints/qwen2.5-0.5b.pt --budgets 0.03 0.05 0.1 [--context-tokens 3000]
```

`hf_chat.py` runs three turns and prints, for each one, how many prompt tokens were restored and how
many were prefilled. Run it twice: the second run restores the conversation stored by the first.

`eval_hf_chat.py` hides six facts in a long document. It stores that turn compressed, then asks for
each fact in a new turn. For each budget it prints stored size, correct answers, compress and restore
time, and the KL divergence of the first answer token against the uncompressed cache.

---

## 6. Running with vLLM

### 6.1 How it plugs in

vLLM v1 has a **KV connector** API for moving KV cache in and out of its paged GPU memory (LMCache and
the NIXL disaggregation connector use it). `DimmerKVConnector` implements it:

- **On the scheduler side**, it decides what to load for each new request and what to save for each
  finished one, using the store's prefix index.
- **On the worker side**, it copies KV out of the request's blocks (saves) and decodes stored KV into
  newly allocated blocks (loads).

Installing the package registers the connector name through vLLM's `vllm.general_plugins` entry point.
`"kv_connector_module_path": "dimmerkv.integrations.vllm.connector"` also works, without the entry point.

### 6.2 Server

```bash
vllm serve meta-llama/Llama-3.2-1B-Instruct --gpu-memory-utilization 0.7 \
  --kv-transfer-config '{
    "kv_connector": "DimmerKVConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "checkpoint": "checkpoints/llama-3.2-1b.pt",
      "store_path": "/var/lib/dimmerkv",
      "budget": 0.03
    }}'
```

Clients need no changes. Any OpenAI-compatible chat client works; send the full message history
each turn as usual:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
history = [{"role": "user", "content": open("report.txt").read() + "\n\nSummarize this."}]
reply = client.chat.completions.create(model="meta-llama/Llama-3.2-1B-Instruct", messages=history)
history += [{"role": "assistant", "content": reply.choices[0].message.content},
            {"role": "user", "content": "What did it say about Q3 revenue?"}]
reply = client.chat.completions.create(model="meta-llama/Llama-3.2-1B-Instruct", messages=history)
```

The second request's prompt starts with the first conversation. If vLLM's GPU prefix cache still has
it, vLLM uses that, exactly and for free. If it was evicted (busy server, restart, user away), the
connector restores it from the store.

### 6.3 Offline (Python)

```python
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

llm = LLM(model="meta-llama/Llama-3.2-1B-Instruct", gpu_memory_utilization=0.7,
          kv_transfer_config=KVTransferConfig(
              kv_connector="DimmerKVConnector", kv_role="kv_both",
              kv_connector_module_path="dimmerkv.integrations.vllm.connector",
              kv_connector_extra_config={"checkpoint": "checkpoints/llama-3.2-1b.pt",
                                         "store_path": "/tmp/dimmerkv"}))
```

`examples/vllm_chat.py` runs two turns and calls `llm.reset_prefix_cache()` between them. Turn 2 must
then restore from the store; check that it still answers the secret code (6142). Add `--no-dimmerkv`
for a baseline run without the connector.

### 6.4 Configuration (`kv_connector_extra_config`)

| key | default | meaning |
|---|---|---|
| `checkpoint` | **required** | calibration file from `dimmerkv calibrate` for the served model |
| `store_path` | `~/.cache/dimmerkv` | store directory, shared by the scheduler and worker processes; must be on local disk (SQLite) |
| `budget` | `0.03` | stored size as a fraction of the FP16 KV of each segment (0.03 = 33× smaller) |
| `recent` | `32` | exact tokens at the end of each segment (capped at half the budget) |
| `floor_share` | `1.0` | share of the budget first spent on a uniform floor (the `floor1` setting) |
| `min_tokens` | `256` | conversations shorter than this are not stored |
| `ram_cache_mb` | `512` | worker-side cache of recently used compressed blobs |
| `max_disk_gb` | no limit | when exceeded, least recently used conversation tails are deleted |
| `max_rung_fraction` | none | drop rungs finer than this fraction of FP16, so a smaller basis sits on the GPU |
| `codec_device` | KV cache device | where compression and decoding run; `cpu` saves GPU memory but is slow |
| `save` / `load` | from `kv_role` | `kv_producer` only saves, `kv_consumer` only loads, `kv_both` does both |

### 6.5 GPU memory

The codec keeps the PCA basis on the GPU (Llama-3.2-1B: about 0.5 GB in FP32). While a finished turn
is being compressed it also needs temporary memory, roughly a few times the FP16 KV of that turn.
vLLM pre-allocates most of the GPU for its own cache, so lower `--gpu-memory-utilization` (0.7 is a
safe start). For large models, use `max_rung_fraction` (for example `0.125`) to shrink the basis.

### 6.6 What happens in one request (step by step)

1. **Scheduling (scheduler process).** `get_num_new_matched_tokens` hashes the prompt and asks the
   store for the longest stored prefix. The prefix is cut to vLLM's block size and must leave at
   least one prompt token to compute. If the store has more tokens than vLLM's own GPU prefix cache,
   the connector claims the difference.
2. **Allocation.** `update_state_after_alloc` records which positions to load: from where vLLM's own
   cache stops to where the stored prefix ends. `build_connector_meta` turns them into cache slots.
3. **Load (worker, before the forward pass).** `start_load_kv` decodes the segment chain on the GPU
   and writes those positions into the paged cache. If anything fails, the blocks are reported
   through `get_block_ids_with_load_errors` and vLLM recomputes them.
4. **Finish (scheduler).** `request_finished` works out which computed tokens the store does not
   have yet. If the conversation is long enough, it returns `True`, so vLLM keeps the blocks.
5. **Save (worker, next step).** `wait_for_save` copies those tokens out of the blocks and queues
   them for compression on a background CUDA stream. `get_finished` tells vLLM the blocks can be
   freed. The compressed segment is written to disk and indexed.

### 6.7 Inspecting the store

```bash
dimmerkv stats /var/lib/dimmerkv --checkpoint checkpoints/llama-3.2-1b.pt
# {"conversations": 1, "segments": 3, "stored_bytes": 179946, "tokens": 256, "ram_cached_bytes": 0}
```

Deleting the store directory removes all stored conversations; nothing else depends on it.

---

## 7. Python API

For engines without a ready-made integration, everything goes through `DimmerKV` with stacked tensors
`[layers, kv_heads, tokens, head_dim]`. Keys are passed **with RoPE applied**, as engines cache them.

```python
from dimmerkv import DimmerKV

kv = DimmerKV.open("checkpoints/model.pt", "/path/to/store", device="cuda", budget=0.03)

# After a turn: store what the store does not have yet (compresses only new tokens).
kv.save_conversation(token_ids, keys, values, namespace=None)

# Before the next turn: find and decode the longest stored prefix.
match = kv.lookup(prompt_ids, max_tokens=len(prompt_ids) - 1)       # Match(segment, tokens) or None
if match:
    keys, values = kv.load(match, dtype=torch.float16, device="cuda")  # first match.tokens tokens
```

| call | purpose |
|---|---|
| `DimmerKV.open(checkpoint, store_path, device, block_size=16, ram_bytes, max_disk_bytes, max_rung_fraction, **settings)` | open the codec and store; `settings` are `CompressionSettings` fields (`budget`, `recent`, `floor_share`, `per_token`, `key_weight`, `zlib_level`, `recent_share`) |
| `lookup(token_ids, max_tokens=None, namespace=None)` | longest stored prefix, at 16-token granularity |
| `load(match, dtype, device)` | decode the segment chain, cut to `match.tokens` |
| `save(token_ids, keys, values, start, parent, namespace)` | store tokens `[start, len)` chained to `parent` (low level) |
| `save_conversation(token_ids, keys, values, namespace)` | look up the stored prefix, then save only the rest |
| `DimmerKVCodec.compress(keys, values, budget, recent, position_offset, ...)` / `.decompress(blob)` | the codec alone, for your own storage |
| `ConversationStore.stats()`, `.evict(max_bytes)`, `.delete(segment)` | store maintenance |

Use `namespace` for anything that changes the KV values of the same tokens, such as a LoRA adapter.
The vLLM connector does this automatically (`lora:<name>`). A namespace also keeps tenants'
conversations apart in the index.

---

## 8. How it works

### 8.1 Codec

This is the method from the README, unchanged:

1. **Transform.** Remove RoPE from the keys. Join all layers and KV heads of a token into one vector
   (one for keys, one for values). Project it onto the calibrated PCA basis.
2. **Allocate.** For every token, measure its error at every rung. Give every token the finest rung
   uniform kvtc could afford (the floor). Then spend the remaining bytes greedily on the upgrades
   with the most error removed per byte.
3. **Encode.** Quantize the coefficients in groups of 64 with 2, 4 or 8 bits plus an FP16 offset and
   scale per group. Bit-pack them, add the exact tokens in FP16, and DEFLATE the result.

Decoding runs these steps in reverse and is fast: Qwen2.5-0.5B with 3.2K tokens restores in 66–101 ms
on an M4 Mac. It multiplies only the PCA components that any token actually uses.

### 8.2 Store

```
<store_path>/
  index.sqlite          segments(id, parent, start, end, bytes, created, last_used)
                        prefixes(hash -> segment, tokens)
  blobs/<segment>.dkv   one compressed segment (header + DEFLATE payload)
```

- **Prefix index.** After every 16 tokens, a chained hash of all tokens so far (BLAKE2b, salted with
  the calibration fingerprint and namespace) points to the segment holding that position. A lookup
  hashes the new prompt and takes the longest hit. As with vLLM's own prefix caching, a match means
  the whole prefix matches.
- **Concurrency.** SQLite in WAL mode lets vLLM's scheduler (reads) and worker (writes) share one
  directory. Blob files are written to a temporary name and renamed into place.
- **Eviction.** With `max_disk_gb`, the least recently used segments that have nothing chained after
  them are deleted first. Deleting a segment also deletes everything chained after it.

### 8.3 Segments: why chat turns are stored append-only

When turn 5 ends, turns 1–4 are already stored. Re-compressing the whole conversation would have
three costs:

- it would redo work on every turn,
- it would compress already-compressed KV again, so quality would drift as the chat grows,
- it would store the history several times over.

Instead each turn stores only its own tokens as a **segment** with a parent pointer. Restoring decodes
the chain and concatenates it. Segments that start mid-conversation remove RoPE using their real
positions (`position_offset`) and keep no sinks. Each segment keeps its last `recent` tokens exact,
capped at half its budget. Without the cap, a 100-token turn would spend 32% of its tokens in FP16,
and a 3-turn chat measured 43% of FP16 instead of about 5%. With the cap it measured 5.7%. Long
segments (the paper's 3K–16K contexts) keep all 32 tokens exact, as in the paper's runs.

### 8.4 Correctness safeguards

- Every blob carries the calibration fingerprint, and decoding with a different calibration raises
  an error. The fingerprint is also part of the hash salt, so a re-calibrated model never matches
  old entries.
- Requests with images or audio, or with prompt embeddings, are neither saved nor loaded.
- vLLM's own exact GPU prefix-cache blocks are never overwritten. Only positions past them are loaded.
- Blocks of a finished request are freed only after their KV has been copied out.

---

## 9. Testing and results

### 9.1 Automated tests

```bash
python -m pytest -q tests          # 15 tests, under a second, CPU only
```

| file | what it checks |
|---|---|
| `tests/test_codec.py` | Payload is **byte-identical** to `scripts/kvtc.py` + `scripts/kvtc_storage.py` at three budget/floor settings, and decoded tensors are identical. Also: more budget means smaller error, sinks and recent window are exact, mid-conversation segments use the right positions, trimmed basis, short sequences, fingerprint mismatch |
| `tests/test_store.py` | Prefix lookup, edited-conversation branches, namespaces, segment chains, cascading delete, LRU eviction, two store instances sharing one directory, append-only turns through `DimmerKV` |
| `tests/test_vllm_connector.py` | The connector over two chat turns with stand-in vLLM modules and CPU paged caches, in both the current and the legacy cache layout: save on finish and block release, restore into new blocks, only new tokens saved on turn 2, partial vLLM prefix-cache hits, load failures reported as invalid blocks, tensor parallel rejected |

The tests use a small synthetic calibration, so they need no model download.

### 9.2 End-to-end results (Hugging Face, Qwen2.5-0.5B-Instruct, M4 Mac, MPS)

From `examples/eval_hf_chat.py`: a 3,205-token history with six hidden facts, compressed after
turn 1. Each fact was then asked in a new turn, which restored the history and prefilled only 29
tokens.

| setting | stored | smaller by | correct | first-token KL vs full | compress | restore |
|---|---|---|---|---|---|---|
| full cache | 37.6 MiB | 1× | 6/6 | — | — | — |
| DimmerKV 3% | 1.04 MiB | 36× | 4/6 | 0.84 | 399 ms | 66 ms |
| DimmerKV 5% | 1.70 MiB | 22× | 6/6 | 0.11 | 416 ms | 91 ms |
| DimmerKV 10% | 3.38 MiB | 11× | 5/6 | 0.04 | 441 ms | 101 ms |
| DimmerKV 20% | 6.89 MiB | 5.5× | 6/6 | 0.006 | 571 ms | 145 ms |

From `examples/hf_chat.py`: three short turns (256 tokens) were stored in 180 KB, 5.7% of FP16
(3.1 MB). Turn 3 recalled the user's name from the restored history.

With six questions, a difference of one answer is noise. The KL column falls steadily as the budget
grows. Use 5% as a starting point for small models; the paper's 3% results are on Llama-3.2-1B/3B.

### 9.3 Not yet tested

- **The vLLM connector inside a real vLLM server.** It was written against vLLM's source (main,
  2026-10-26, commit `d26d5db`) and tested only with the stand-ins above, because no CUDA GPU was
  available. The first job on a GPU machine is `examples/vllm_chat.py`. vLLM marks the connector API
  as experimental, so other vLLM versions may need small changes.
- Throughput and latency under concurrent load, and models larger than 0.5B through the package.

---

## 10. Troubleshooting

| symptom | cause and fix |
|---|---|
| `Calibration file has no RoPE tables` | made by `scripts/fit_kvtc.py`: run `dimmerkv convert` |
| `Blob was written with a different calibration file` | the store was filled with another calibration; use a separate `store_path` per calibration file |
| `Calibration ... has N layers x H KV heads ...` (vLLM) | the checkpoint is for a different model than the one served |
| `needs tensor_parallel_size = ... = 1` | the connector supports a single GPU only (see [Limits](#11-limits)) |
| Nothing is restored in vLLM | expected while vLLM's GPU prefix cache still holds the conversation. Also check: conversations under `min_tokens` (256) are not stored; requests with images are skipped; different LoRA adapters do not share entries; look for `DimmerKV` lines in the log |
| CUDA out of memory with the connector | lower `--gpu-memory-utilization`, set `max_rung_fraction` (e.g. 0.125), or `codec_device: "cpu"` |
| Calibration fails importing `datasets` | `pip install datasets`, or use `--text-file` |
| Answers worse than expected | raise `budget` (0.05–0.1); calibrate with more `--chunks` or on text like your traffic |
| Store directory keeps growing | set `max_disk_gb`, or delete the directory |

---

## 11. Limits

- **vLLM: single GPU.** Tensor and pipeline parallel size must be 1, because one PCA basis covers
  every layer and KV head.
- **Full attention only.** Sliding-window and hybrid models (Gemma 2/3, Mistral with sliding window,
  Mamba hybrids) are not supported, nor are models without NeoX-style RoPE.
- **FP16/BF16 KV cache only**; no FP8 KV cache.
- **Text-only requests.** Multimodal and prompt-embedding requests are skipped.
- **Lossy.** Restored KV is approximate. In vLLM, once loaded it also serves other requests that share
  that prefix through the GPU prefix cache.
- **Question not used.** The serving path compresses before the next question is known. The
  question-aware modes, question-time refinement and attention over the codes from `scripts/` are not
  in the package yet. They give the paper's best results.
- **Local disk only.** The store is one directory with SQLite, so it cannot be shared between
  machines.
- **Privacy.** Stored KV is derived from users' conversations and should be treated as sensitive as
  the text. Protect `store_path` (permissions, disk encryption) and delete entries when users delete
  conversations. There is no per-user delete API yet.

---

## 12. Future improvements

Roughly in priority order.

### Validate and measure

1. **Run inside vLLM on a GPU.** Run `examples/vllm_chat.py`, then a multi-turn benchmark: time to
   first token versus recomputing and versus vLLM's CPU offloading, conversations per GiB, accuracy
   with `scripts/test_generation.py` cases.
2. **Larger models and real benchmarks.** Llama-3.1-8B, Qwen3, LongBench and multi-turn chat datasets
   through the package path, not just `scripts/`.
3. **Metrics.** Hits, restored tokens, bytes and compression time through vLLM's connector stats and
   Prometheus hooks (`build_kv_connector_stats`, `build_prom_metrics`).

### Engine coverage

4. **SGLang.** A HiCache storage backend. The codec works per token, so it can compress SGLang's pages
   directly, with the budget applied per page.
5. **LMCache.** A DimmerKV serializer would reach vLLM and SGLang through LMCache's storage tiers
   (CPU, disk, Redis, S3).
6. **llama.cpp / GGUF.** A C/C++ decoder (PCA multiply plus bit unpacking) and the basis stored
   next to the GGUF file, hooked into llama.cpp's session save and load.
7. **TensorRT-LLM.** Through its KV cache offloading hooks.

### Wider model and setup support

8. **Tensor parallel.** Fit one basis per TP rank over that rank's KV heads. Each rank then
   compresses its own shard and stores it under a rank suffix.
9. **Sliding-window and hybrid models.** Compress only the full-attention layers, or store the
   window separately.
10. **FP8 KV caches.** Dequantize on save, and quantize to FP8 on load.
11. **Multimodal requests.** Include image hashes in the prefix hash, as vLLM already does for its own
    prefix cache.

### Quality

12. **Question-aware restore.** When a stored conversation is restored, the new prompt is known. Score
    the coarse cache with it and fetch more precision for the tokens it needs: question-time refinement
    from `scripts/progressive.py`, which measured close to full-cache accuracy at 2% in FINDINGS.md.
13. **Hot/cold tiers.** Store progressive bit-planes: keep the first planes in RAM and the rest on
    disk, and demote old conversations by dropping planes without re-encoding.
14. **Calibration on chat data** and per-model choice of the default budget.

### Performance

15. **Attention over the codes in vLLM.** A custom attention backend that reads compressed tokens
    directly (`scripts/compressed_attention.py`). It would cut GPU memory while answering by 8–14×,
    not only the storage.
16. **Asynchronous loads.** Return `load_kv_async=True` and decode on a side stream while other
    requests run, so restores do not stall the batch.
17. **Fused kernels.** Triton kernels for bit unpacking and the PCA multiply. Use zstd instead of
    zlib, on the GPU, for faster compression.
18. **Compress during long generations.** Offload the compressed prefix of a running request when
    memory is tight, not only when it finishes.
19. **Segment compaction.** Merge many short turn segments of an old conversation into one, re-coded
    once with a better per-token allocation.

### Production

20. **Shared and remote storage.** A pluggable blob backend (Redis, S3, NFS) with the index in a
    server database, so several vLLM instances share conversations.
21. **Per-user controls.** Store and delete by user or conversation ID, apply retention policies,
    encrypt blobs at rest.
22. **Packaging.** Publish to PyPI, pin tested vLLM versions in CI, and run a GPU smoke test.
