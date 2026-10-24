# DimmerKV — Hashcode 14.0 (Team Deadoop)

Build plan page (team-facing): https://claude.ai/artifact/MFC8w1cwiTh7zMDqrvLte4

## Event and rules
- Hashcode 14.0, Track 01 LLM Inference (Akamai), PES University. Build window: **Sat 24 Oct 2026 3:00 PM → Sun 25 Oct 3:00 PM IST**.
- No project code before 3 PM Saturday (Rule 3: working on the solution before the start = disqualification). Setup, installs and model downloads are fine.
- Must use at least one track paper: we use **H2O** and **Lexico** (plus capstone papers: Mooncake, EVICPRESS, kvtc, MiKV, Loki, ChunkAttention, Minima, Unweight, ZipServ).
- AI coding tools are allowed (confirmed by organisers).
- Write all code fresh. Do **not** reuse `Capstone/HashCode/prototype/` (pre-event).
- Team: Srikanth Balaaji = cache + allocator; Sushanth B = models + baselines; T S Sathya Kiran = eval + demo.

## What we build
Per-token KV-cache compression for Hugging Face models. After prefill:
1. Score each token with H2O accumulated attention (per layer, per KV head; sum query heads sharing a KV head under GQA). Score layer by layer with hooks so it fits small GPUs.
2. Encode older tokens: **PCA first** (per-layer basis from WikiText, 1 B/coef, ordered), **Lexico second** (OMP, ≤32 atoms from 4,096, 3 B/atom + 2 B for the first).
3. Greedy allocator: value of piece j of token i = score(i) × error drop / bytes; `cummin` along j so codes stay prefixes; fill a byte budget; 0 pieces = evicted. Recent window stays exact. Assert bytes ≤ budget.
4. Level 1 (build first): rebuild vectors from codes, write back to the cache, mask evicted tokens, `generate` the next question. Memory is *counted* from codes — label it that way. Level 2 (stretch): attention on codes, (q·basis)·code.
- Main claim to test: at ≤15% of FP16 KV, DimmerKV beats H2O and uniform PCA/Lexico at equal bytes on questions asked **after** compression, while staying near H2O on perplexity. Goal = capacity; speed is stretch, report honestly.
- HF notes: `attn_implementation="eager"` for attention weights; cache is `DynamicCache` with `.layers[l].keys/.values`; cached keys are post-RoPE. Pin library versions for the whole team.

## Phases (one at a time; each ends with a test, then wait for "next")
1. Setup (3–5 PM): env check, model load, one generate; collect K/V, fit PCA; start Lexico dictionary training in background.
2. Core (5–11 PM): scores, PCA codec, allocator, compressed cache. Exit: budget never exceeded; pass-key correct at 20%.
3. Baselines (7 PM–1 AM): full, H2O, uniform PCA, uniform Lexico, MiKV-style (H2O + INT2), KIVI-2 if time — equal bytes, same prompts/seeds.
4. Harness (5 PM–1 AM): needle (question after compression = main; question in prompt), perplexity, LongBench subset (~30/task, 2–3 tasks), GSM8K (~50); CSV + charts.
5. Main runs (1–7 AM) on the biggest GPU available.
6. Tuning (7–11 AM): one axis at a time; then extensions.
7. Demo + wrap (9 AM–3 PM): Gradio (memory slider, per-token heat-map, H2O vs DimmerKV side by side, offline results tab), README, final table, backup video by 1 PM.
- Cut order if late: KIVI-2 → Lexico → LongBench → extensions → Level 2. Never cut: needle test, H2O + uniform baselines, demo.

## Experiment matrix
- Budgets 10/15/20/30% of FP16 KV.
- Storage: PCA vs Lexico separately for K and V; PCA keys + Lexico values.
- Score: H2O sum (default) vs mean vs last 32/64-query window (SnapKV-style); with/without min-pieces floor.
- Recent window: 32 vs half-budget. Budget scope: per head / per layer / global.
- Code lengths: free vs fixed levels 0/8/16/32/64 (dense GPU work). Keys pre vs post RoPE if time.
- Record: accuracy/recall/ppl, compressed bytes, peak GPU memory, encode time, tokens/s, allocator histogram.

## Extensions (priority order, after core works)
1. One K+V coefficient pool (allocator picks the split). 2. Choose count and precision (INT4 vs FP8) per token.
3. Refine on demand: dim codes on GPU, exact KV in CPU; codes as search index → fetch top-k exact tokens when a question hits dim ones (headline fix for "scores judged before the question").
4. Re-score with real questions / pool across users on shared prefixes. 5. Shared basis across fine-tunes and layer groups.
6. Entropy-code codes off-GPU. 7. Lossless exponent coding for exact tokens. Stretch: Level 2, self-speculative draft on bright tokens, joint weight+KV budget with GPTQ.

## Machines
- **LOQ** (RTX 5060 Laptop 8 GB, Windows) — **main machine**, runs Claude Code from 3 PM: dev, unit tests, demo; Llama-3.2-1B fp16, 3B 4-bit; 2–4K-token docs.
- **AWS g6e.xlarge** (L40S 48 GB, us-east-1, credits) — quota request pending; main runs on 3B/8B fp16, 4–8K docs. Add the budget auto-stop action after launch; terminate on Sunday.
- **RunPod** 4090/L40S — instant paid fallback (~$10–20/day) if no LOQ and no AWS.
- **Kaggle** 2× T4 — free fallback, batch runs via `kaggle kernels push`.
- **Sushanth's and Sathya Kiran's Macs** (Apple Silicon): baselines/eval/demo development; small runs on `mps` (1B fp16; no bitsandbytes/4-bit); venv at `~/.venvs/hashcode`; set `PYTORCH_ENABLE_MPS_FALLBACK=1`. Code must pick the device automatically (cuda → mps → cpu) and use POSIX-safe paths (pathlib), since the team mixes Windows and macOS.
- **Srikanth's laptop** (Windows, no GPU): CPU venv at `%USERPROFILE%\.venvs\hashcode` (Python 3.11, torch CPU) — logic tests with tiny models only. Do not use the Anaconda Python here (numpy is broken).
- Fallback models: Qwen2.5-1.5B/3B-Instruct if Llama access is slow. Final results table comes from one machine.
- Don't put the venv inside this OneDrive folder.

## Progress log (2026-10-24, 4:39 PM IST)

### Environment and model smoke test

- Python environment: `%USERPROFILE%\.venvs\hashcode` on Windows.
- Hugging Face download completed successfully with:
  `$hf = "$env:USERPROFILE\.venvs\hashcode\Scripts\hf.exe"`
  `& $hf download meta-llama/Llama-3.2-1B-Instruct`
- `meta-llama/Llama-3.2-1B-Instruct` loads and generates successfully.
- The model is loaded with `attn_implementation="eager"` so attention weights can
  later be used by H2O scoring.
- Device selection used so far: CUDA first, then MPS, then CPU. This machine used
  CUDA with FP16.
- Model configuration confirmed:
  - 16 transformer layers
  - 32 query heads
  - 8 KV heads
  - head dimension 64
  - hidden dimension 2048
  - GQA group size is 4 query heads per KV head

### Cache inspection

Added [scripts/inspect_cache.py](scripts/inspect_cache.py). It runs a prefill,
reads the returned `DynamicCache`, and validates that every layer has the same
shape. On the test machine the observed cache was:

```text
cache type: DynamicCache
layers: 16
key shape: (1, 8, 20, 64)
value shape: (1, 8, 20, 64)
dtype: torch.float16
```

The cache layout is `[batch, kv_heads, sequence_length, head_dim]`. Keys are the
post-RoPE keys returned by the model.

### PCA calibration

Added [scripts/fit_pca_basis.py](scripts/fit_pca_basis.py). It uses eight
calibration prompts, runs model prefill, collects per-layer key/value vectors,
and fits a separate centered PCA basis for every layer and stream. Each vector
has 64 dimensions; the current basis uses 16 components. The collected
calibration vectors are flattened from `[batch, kv_heads, tokens, head_dim]` to
`[kv_heads * tokens, head_dim]`.

The saved artifact is `checkpoints\pca_basis.pt` (ignored by `.gitignore`). It
contains, for each of 16 layers:

- key mean and key basis
- value mean and value basis

Calibration run results:

```text
samples per layer/stream: 936
mean relative MSE: 0.3184
mean explained variance: 0.6695
```

Keys retained approximately 0.82-0.87 explained variance; values retained
approximately 0.46-0.56. These are in-sample calibration metrics, not a final
quality claim.

### PCA codec

Added:

- [scripts/pca_codec.py](scripts/pca_codec.py)
- [scripts/test_pca_codec.py](scripts/test_pca_codec.py)

`PCACodec` loads the saved basis, encodes and decodes individual vectors, and
encodes/decodes all cache layers. The current codec uses FP16 coefficients to
isolate PCA loss from future INT4/FP8 quantization loss:

```text
64-value FP16 vector -> 16-value FP16 coefficient vector -> 64-value reconstruction
```

For both keys and values, the nominal per-token cache cost is 25% of the
uncompressed FP16 KV cost, excluding the shared PCA basis:

```text
full:       64 values * 2 bytes * 2 streams
compressed: 16 coefficients * 2 bytes * 2 streams
ratio:      0.25
```

The held-out round-trip test uses a prompt not included in the eight calibration
prompts. It verifies shape preservation, finite reconstructed values, error, and
cosine similarity. Observed result:

```text
tokens: 24
encoded key shape: (1, 8, 24, 16)
full FP16 KV bytes: 786432
PCA coefficient bytes: 196608
nominal byte ratio: 0.250
mean relative MSE: 0.3857
minimum cosine similarity: 0.614569
```

The held-out error is higher than the in-sample error, so the current 16
component basis is only a codec baseline. Do not describe it as preserving
generation quality until generation is tested with the reconstructed cache.

The scripts can be run from the project root with:

```powershell
$py = "$env:USERPROFILE\.venvs\hashcode\Scripts\python.exe"
& $py scripts\inspect_cache.py
& $py scripts\fit_pca_basis.py
& $py scripts\test_pca_codec.py
```

The 4:39 PM results above are superseded by the update below (old basis:
8 prompts, BOS included, 16 components only).

## Progress log (2026-10-24, ~5:20 PM IST): codec rework + first behavioral test

### Changes
- `scripts/common.py`: shared `choose_device`, `load_model` (uses `dtype=`),
  `kv_config` (layers / KV heads / head dim read from the model config),
  `cache_tensors`, `build_cache`, WikiText chunking, vector collection.
- `fit_pca_basis.py`: WikiText-2 train (`Salesforce/wikitext`), 32 x 512 tokens,
  BOS position left out (`--skip-prefix 1`, kept exact in the cache), saves the
  **full ordered 64x64 basis** (take any prefix k), stores `kv_heads`,
  `per_head`, `skip_prefix`. `--per-head` fits one basis per (layer, KV head).
- `pca_codec.py`: `encode(..., components)` slices the basis; INT8 coefficients
  with one FP16 absmax scale per vector (`bytes_per_vector`: k + 2);
  `compress_cache` = uniform PCA baseline with exact sink + optional exact
  recent window, returns counted bytes. Old basis files are rejected.
- `test_pca_codec.py`: held-out WikiText **test** split, sweeps k and 8/16 bit,
  reports per-token cosine (p5, min).
- `test_generation.py`: 6 needle facts in ~1.6K tokens of WikiText; prefill the
  context only, compress, then ask each question on top of the rebuilt cache
  (question after compression). Full cache is the control. `--streams k|v|kv`.
- `lexico.py` (batched OMP, `refit`, byte accounting 3 B/atom + 2 B) and
  `train_lexico.py` (OMP + Adam dictionary updates, 4096 atoms, per layer, all
  KV heads pooled, saves after every layer to `checkpoints/lexico_dict.pt`).

### Results (Llama-3.2-1B, LOQ, held-out)
Per-head basis beats the shared basis everywhere and is now the default
(`checkpoints/pca_basis.pt` = per head; `pca_basis_shared.pt` kept for A/B).
INT8 coefficients cost nothing measurable vs FP16.

| k | bytes (INT8) | key relMSE | value relMSE |
|---|---|---|---|
| 8 | 0.080 | 0.264 | 0.608 |
| 16 | 0.142 | 0.171 | 0.460 |
| 32 | 0.267 | 0.065 | 0.256 |
| 48 | 0.392 | 0.019 | 0.109 |

Needle recall after compression (uniform PCA, INT8, 6 needles):
full 6/6; k=8 0/6; k=16 0/6; k=32 0/6; k=48 4/6 (5/6 with 32 recent exact).
Keys-only or values-only compression both fail at k=16 and give 4-5/6 at k=32,
so both streams matter. **Uniform PCA alone is not viable below ~40% bytes**,
which is the motivation for the allocator (bright tokens exact, dim tokens
short codes). Lexico smoke test (20 steps): values relMSE ~0.1 at 8 atoms
(26 B) vs 0.46 for PCA k=16 (18 B), so expect PCA keys + Lexico values or
Lexico for both. Lexico numbers do not yet include coefficient quantization.

Open note: per-head PCA has a few vectors with very negative cosine (min about
-0.8) while p5 is fine; likely tiny-norm vectors. Check before trusting
per-token error as an allocator signal.

### Run
```powershell
$py = "$env:USERPROFILE\.venvs\hashcode\Scripts\python.exe"
& $py scripts\fit_pca_basis.py --per-head      # writes checkpoints\pca_basis.pt
& $py scripts\test_pca_codec.py
& $py scripts\test_generation.py --components 16 32 48 [--recent 32] [--streams k]
& $py scripts\train_lexico.py --steps 1500     # ~45 min on the 5060
```

### Next
1. When Lexico training finishes: Lexico codec (12-bit coefficients), add it to
   `test_generation.py` as uniform Lexico at equal bytes.
2. H2O scores (per layer, per KV head, GQA summed) via hooks.
3. Greedy allocator over PCA/Lexico prefixes with byte budget assert; exit test
   = pass-key correct at 20%.

## Progress log (2026-10-24, ~5:45 PM IST): Lexico codec + timing

Working rule from the team: the phase plan is a guide, adapt as results
demand, and **every benchmark reports time** next to error and bytes.

- `train_lexico.py` finished: 4096 atoms per layer/stream, 1500 steps, ~19 min.
- `lexico.py`: `LexicoCodec` (OMP, 12-bit coefficients + FP16 scale per vector,
  3 B/atom + 2 B), shared `uniform_compress` in `common.py`.
- `bench_codecs.py`: held-out error, bytes, encode/decode ms per 1K tokens.
- `test_generation.py`: `--pca` / `--lexico` lists, prefill/compress/generate times.

Held-out (2,044 tokens, all layers, K+V, RTX 5060 laptop):

| codec | B/vec | ratio | key err | val err | enc ms/1K tok | dec ms/1K tok |
|---|---|---|---|---|---|---|
| PCA k=16 int8 | 18 | 0.141 | 0.171 | 0.460 | 18 | 11 |
| PCA k=32 int8 | 34 | 0.266 | 0.065 | 0.256 | 14 | 9 |
| PCA k=48 int8 | 50 | 0.391 | 0.019 | 0.109 | 26 | 15 |
| Lexico s=4 | 14 | 0.109 | 0.110 | 0.263 | 287 | 9 |
| Lexico s=5 | 17 | 0.133 | 0.088 | 0.209 | 359 | 10 |
| Lexico s=8 | 26 | 0.203 | 0.047 | 0.104 | 581 | 12 |
| Lexico s=11 | 35 | 0.273 | 0.025 | 0.050 | 838 | 15 |

Needle (6 facts, 1,588-token context, question after compression, uniform):
PCA k=16/32/48 -> 0/0/4 of 6; Lexico s=4/5/8/11 -> 0/0/3/5 of 6.
Compress time for the whole context: PCA ~10 ms, Lexico 0.6-1.4 s
(prefill itself ~0.5 s). Lexico wins quality at equal bytes by a wide margin
but encode is ~40-60x slower than PCA; decode cost is the same.
Per-question generate times are not comparable yet (first call includes
warm-up; wrong answers stop early).

## Progress log (2026-10-24, ~6:15 PM IST): K/V codec mix + better needle metrics

`test_generation.py` now takes `--configs` specs (`pca16`, `lex8`,
`pca16/lex8` = key/value) and `--seeds` (default 3 contexts x 6 needles = 18).
Metrics: **correct** (expected number appears in the greedy reply, first 12
tokens), **agree** (reply identical to full-cache reply), **answer NLL**
(teacher-forced mean NLL per answer token; `ans p` = exp(-NLL), smooth so
it separates configs with equal accuracy), compress ms per context
(encode + decode, CUDA-synced), generate ms per new token.
`compress_stream` on both codecs compresses one stream at a time.

Results (Llama-3.2-1B, ~1.6K-token contexts, uniform codes, no recent window;
logs/needle_kv_mix*.log):

| budget | config (K/V) | correct | ans p | compress ms |
|---|---|---|---|---|
| ~14% | pca16 | 0/18 | 0.001 | 27 |
| | lex5 | 3/18 | 0.024 | 671 |
| | pca16/lex5 | 0/18 | 0.001 | 293 |
| | lex8/pca8 | 0/18 | 0.002 | 493 |
| ~20% | lex8 | **10/18** | **0.267** | 948 |
| | pca24/lex8 | 1/18 | 0.004 | 484 |
| | lex8/pca24 | 4/18 | 0.053 | 476 |
| | lex11/pca16 | 3/18 | 0.039 | 689 |
| ~27% | pca32 | 1/18 | 0.023 | 13 |
| | lex11 | **16/18** | **0.789** | 1372 |
| | pca32/lex11 | 9/18 | 0.105 | 698 |
| | lex11/pca32 | 10/18 | 0.248 | 697 |
| | pca16/lex16 | 0/18 | 0.000 | 1100 |

Conclusions: Lexico for both streams wins at every budget. Mixing halves
compress time but costs far more quality than it saves; keys are the more
sensitive stream (PCA keys + rich Lexico values collapses to 0/18), but PCA
values also hurt. Generate time per token is identical across configs because
Level 1 rebuilds a full FP16 cache; codes only change memory, not decode
speed. Encode speed is the cost to attack (OMP), not the codec choice.

## Progress log (2026-10-24, ~7:00 PM IST): allocator + H2O baseline

- `scripts/dimmer.py`: custom attention `dimmer` (registered with
  transformers' AttentionInterface; eager math) that records H2O scores at
  prefill (attention received, summed over queries and the GQA group) and
  hides evicted tokens per (layer, KV head) with a mask. `allocate()`: levels
  per middle (layer, head, token) slot = evict / Lexico {1,2,3,4,6,8,11,16}
  atoms for K and V / exact; piece value = score x drop in relative K+V error
  / extra bytes, cummin along levels, stable greedy fill of the byte budget
  (global, per layer or per head), asserts bytes <= budget. H2O baseline = the
  same allocator with levels {evict, exact}. Level 1: memory is counted from
  codes; evicted/compressed slots still sit in the FP16 tensor.
- `lexico.omp(..., errors=list)` returns the exact error of every prefix.
- `test_generation.py`: `h2o:<budget>` and `dkv:<budget>[:floorN|layer|head|mean|flat|powX|kwX]`
  configs, recent window 32 exact for every config (default), level histograms.
- `diagnose_scores.py`: percentile of needle tokens under each score.

**Key finding: H2O scores do not see post-compression questions.** Needle
answer tokens sit at the 31-35th percentile of H2O scores (mean-normalized:
~45th, i.e. random). Allocating by H2O moves bytes *away* from the facts.

18 needles, ~1.6K tokens, 32 recent exact (logs/needle_alloc_v1.log):

| config | bytes | correct | ans p | compress ms |
|---|---|---|---|---|
| lex3 (uniform) | 0.105 | 0/18 | 0.002 | 344 |
| dkv:0.10:flat | 0.100 | 1/18 | 0.010 | 2365 |
| lex5 (uniform) | 0.151 | 5/18 | 0.030 | 670 |
| h2o:0.15 | 0.150 | 0/18 | 0.003 | 138 |
| dkv:0.15 (H2O scores) | 0.150 | 0/18 | 0.012 | 2414 |
| dkv:0.15:pow0.3 | 0.150 | 6/18 | 0.103 | 2421 |
| **dkv:0.15:flat** | 0.150 | **10/18** | **0.269** | 2380 |
| dkv:0.15:flat:kw2 | 0.150 | 8/18 | 0.225 | 2373 |
| lex8 (uniform) | 0.220 | 9/18 | 0.307 | 949 |
| **dkv:0.20:flat** | 0.200 | **14/18** | **0.694** | 2374 |

`flat` = ignore scores, allocate atoms by error drop per byte only
(variable-rate Lexico). At 15% it doubles uniform Lexico and matches uniform
Lexico at 22%; at 20% it beats uniform at 22% by 5 needles. floor1 changes
nothing (flat never evicts). 10% is a cliff for every method.
Cost: compress ~2.4 s per 1.6K-token context (OMP to 16 atoms on every slot).

## Progress log (2026-10-24, ~7:45 PM IST): H2O without eviction; question in prompt

- H2O-scored allocation with no eviction (floor1/2/3) still 0-1/18 in
  `after` mode; the more weight H2O gets (pow 1 -> 0.3 -> 0.1 -> flat), the
  better the score (0 -> 6 -> 10 -> 10/18). Eviction was not the problem;
  H2O moves atoms away from facts (logs/needle_h2o_noevict.log).
- New: `--mode prompt` (question at the end of the prefilled text, one
  prefill + compression per question) and `snap` scores (attention from the
  last 32 queries only, SnapKV-style; `prefill_with_scores` returns
  {"sum", "snap"}).

Question in prompt, 18 questions (logs/needle_prompt_mode.log):

| config | bytes | correct | ans p | compress ms |
|---|---|---|---|---|
| lex3 | 0.105 | 1/18 | 0.010 | 344 |
| h2o:0.10:snap (SnapKV eviction) | 0.100 | 12/18 | 0.381 | 22 |
| dkv:0.10:flat | 0.100 | 4/18 | 0.120 | 2399 |
| **dkv:0.10:snap** | 0.100 | **14/18** | **0.570** | 2452 |
| lex5 | 0.151 | 7/18 | 0.106 | 615 |
| h2o:0.15 (sum) | 0.150 | 1/18 | 0.016 | 40 |
| h2o:0.15:snap | 0.150 | 10/18 | 0.404 | 24 |
| dkv:0.15 (sum) | 0.150 | 4/18 | 0.088 | 2454 |
| **dkv:0.15:snap** | 0.150 | **17/18** | **0.912** | 2443 |
| dkv:0.15:snap:floor1 | 0.150 | 15/18 | 0.809 | 2446 |
| dkv:0.15:snap:pow0.3 | 0.150 | 16/18 | 0.715 | 2442 |
| dkv:0.15:flat | 0.150 | 13/18 | 0.482 | 2423 |

Conclusions: with the question visible, the original DimmerKV idea works
and is the best config tested: SnapKV-window scores + graded Lexico levels
(17/18 at 15%, 14/18 at 10%) beat SnapKV-style eviction at equal bytes
(10/18, 12/18) and error-only allocation. Summed H2O scores stay poor even
here (question tokens are ~20 of ~1,600 queries). The score signal is what
matters: summed attention misleads, question attention is excellent, no
question -> use flat. Natural design: flat by default, re-allocate /
refine with question-window scores when a question arrives.

## Progress log (2026-10-24, ~8:30 PM IST): faster OMP

- `lexico.omp` now keeps an orthonormal basis of the chosen atoms
  (Gram-Schmidt, applied twice) so each step is a projection, not a linear
  solve; coefficients are solved once at the end; the 4,096-atom search runs
  in FP16 on GPU. The old version is kept as `omp_reference`.
- `scripts/profile_allocator.py`: per-stage timing + check that OMP prefix
  errors match a direct refit (max relative mismatch 1.4e-7).

| | before | after |
|---|---|---|
| OMP 16 atoms, one layer/stream (12.4K vectors) | 70 ms | 33 ms |
| allocate() at 15%, 1.6K-token context | 2,483 ms | 1,392 ms |
| Lexico encode s=8, ms per 1K tokens | 581 | 347 |
| Lexico encode s=16, ms per 1K tokens | 1,304 | 674 |

Codec errors identical to 4 decimals; needle results identical (lex5 5/18,
h2o 0/18, dkv:0.15:flat 10/18, dkv:0.20:flat 14/18; logs/needle_fast_omp.log).
Remaining OMP time is the FP16 atom search, memory-bound at ~1.3 ms/step
(matmul 0.4 ms + abs/argmax over [12.4K, 4096]); batching layers does not
help. At 3K tokens: prefill 1.0 s, allocate 2.6 s (OMP 2.1 s, rest 0.45 s).
4K prefill measured 7.5 s on the 8 GB card (eager attention memory pressure),
not a reliable number. Allocation is still ~1.4-2.5x prefill time.

## Progress log (2026-10-24, ~9:00 PM IST): full suite with the new OMP

`scripts/compare_omp.py` (32 layer/streams x 8,176 held-out vectors):
97.7% of atom slots identical to `omp_reference`; stored-code error
identical to 4 decimals at 4/8/16 atoms; 2.02x faster (1,366 -> 675 ms).

Full needle suite rerun (logs/suite_*_fastomp.log), 18 questions each.
Correct (ans p), compress ms in brackets:

| memory | config | question after compression | question in prompt |
|---|---|---|---|
| 10% | uniform lex3 | 0 (0.002) [230] | 1 (0.010) [215] |
| | H2O / SnapKV eviction | 0 (0.002) [55] | 12 (0.381) [44] |
| | dkv flat | 1 (0.010) [1313] | 4 (0.119) [1328] |
| | dkv snap | - | 13 (0.482) [1375] |
| 15% | uniform lex5 | 5 (0.028) [295] | 7 (0.104) [303] |
| | H2O (sum) eviction | 0 (0.003) [21] | 1 (0.016) [23] |
| | SnapKV eviction | - | 10 (0.404) [23] |
| | dkv H2O sum | 1 (0.011) [1332] | - |
| | dkv flat | 10 (0.262) [1284] | 13 (0.521) [1361] |
| | dkv snap | - | 17 (0.909) [1382] |
| 20-22% | uniform lex8 (22%) | 10 (0.289) [482] | 12 (0.453) [494] |
| | H2O / SnapKV eviction | 0 (0.003) [22] | 13 (0.436) [23] |
| | dkv flat | 14 (0.684) [1298] | 17 (0.852) [1368] |
| | dkv snap | - | 17 (0.869) [1380] |

Every result matches the old-OMP runs within 1 question (noise level).
New: at 20% with the question in the prompt, flat and snap both reach 17/18
(SnapKV eviction 13/18). Compression is ~1.3 s per 1.6K context for every
allocator config (was ~2.4 s).
