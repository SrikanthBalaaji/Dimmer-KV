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

## Progress log (2026-10-24, ~5:00 PM IST): codec rework + first behavioral test

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

## Progress log (2026-10-24, ~5:35 PM IST): Lexico codec + timing

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

## Progress log (2026-10-24, ~5:45 PM IST): K/V codec mix + better needle metrics

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

## Progress log (2026-10-24, ~6:35 PM IST): allocator + H2O baseline

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

## Progress log (2026-10-24, ~7:15 PM IST): H2O without eviction; question in prompt

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

## Progress log (2026-10-24, ~7:35 PM IST): faster OMP

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

## Progress log (2026-10-24, ~7:50 PM IST): full suite with the new OMP

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

## Progress log (2026-10-24, ~8:50 PM IST): kvtc + DimmerKV per-token rungs

Implemented the kvtc idea (Staniszewski & Łańcucki, ICLR 2026; on our
capstone list): per token, keys with RoPE removed from all 16 layers × 8 KV
heads form one 8,192-number vector (likewise values); one PCA basis fitted
on ~100K WikiText tokens; coefficients quantized in groups of 64 with
0/2/4/8 bits and a per-token FP16 shift and scale per group; a greedy
"ladder" of bit allocations from 1/64 to 1/2 of FP16 size. No DEFLATE yet.
First 4 tokens (sinks) and last 32 exact. (`scripts/kvtc.py`, `fit_kvtc.py`;
calibration takes 1.3 min on the laptop.)

- `kvtc@b`: every token on the finest rung that fits (kvtc as published).
- `dkvtc@b`: **DimmerKV flat on kvtc**: each token's key and value get their
  own rung, chosen greedily by error drop per byte under the same budget.

18 needles, ~1.6K-token contexts (logs/kvtc_*.log). Correct / ans p:

| budget | method | bytes | after compression | question in prompt | ppl (full 15.84) | compress ms |
|---|---|---|---|---|---|---|
| 4% (25×) | kvtc | 0.038 | 16 / 0.753 | 16 / 0.775 | 16.24 | ~170 |
| | dkvtc | 0.040 | **18 / 0.775** | **17 / 0.814** | **16.14** | ~250 |
| 5% | kvtc | 0.043 | 17 / 0.767 | 16 / 0.776 | 16.09 | ~130 |
| | dkvtc | 0.050 | 16 / 0.874 | 16 / 0.880 | 16.02 | ~190 |
| 7% | kvtc | 0.063 | 17 / 0.916 | 17 / 0.908 | 15.95 | ~120 |
| | dkvtc | 0.070 | 17 / 0.958 | **18 / 0.967** | 15.93 | ~185 |
| | SnapKV eviction | 0.070 | — | 8 / 0.315 | 17.34 (H2O) | ~25 |
| 10% | kvtc | 0.084 | 17 / 0.940 | 18 / 0.941 | 15.96 | ~120 |
| | dkvtc | 0.100 | 17 / 0.950 | 18 / 0.966 | 15.92 | ~190 |
| | H2O / SnapKV eviction | 0.100 | 0 / 0.002 | 12 / 0.381 | 16.99 | ~30 |
| | DimmerKV flat on Lexico | 0.100 | 1 / 0.010 | 13 / 0.482 (snap) | 17.68 | ~1,350 |

What this shows:

- **kvtc is a different league from everything else we built.** At 4% memory
  (25×) it still answers 16/18 needles with perplexity +2.6%; H2O at 10%
  answers 0/18. Cross-layer PCA with RoPE removed is the key, as the paper's
  ablation says. It is also ~10× faster to compress than our Lexico allocator.
- **Per-token rungs (DimmerKV) add a consistent but modest gain on top.**
  Uniform kvtc cannot use the whole budget (its rungs are coarse), so compare
  along the bytes axis: at equal bytes, dkvtc's answer probability is higher
  at every point (e.g. dkvtc 0.070 bytes → 0.958 vs kvtc 0.084 bytes → 0.940,
  question after compression), i.e. roughly 15–25% fewer bytes for the same
  needle quality. On perplexity the two are within noise except at 4%.
- Correct counts are saturated (16–18/18) at these budgets; ans p and
  perplexity carry the comparison. Budgets below ~4% are dominated by the 36
  exact tokens in 1.6K contexts, so the tighter regime needs 4K+ contexts.

Also added (committed earlier): `methods.py` (shared method names), `eval_perplexity.py`, `run_grid.py` (CSV grid runner). Perplexity on Lexico-era methods, 4 chunks: full 15.84; 10%: lex 18.82, H2O 16.99, SnapKV 17.14, dkv flat 17.68, dkv snap 17.27; 15%: H2O 16.71, dkv flat 16.53; 20%: dkv flat 16.05 (logs/ppl_1b.log).

## Progress log (2026-10-24, ~9:55 PM IST): full comparison grid

New for this grid: `kvtcf@b` (kvtc with evenly spaced tokens moved one rung
up so it spends the whole budget; the fair equal-bytes baseline, since plain
kvtc's coarse rungs leave budget unused) and `dkvtc@b:snap` (DimmerKV on
kvtc with each token's error weighted by question-window attention). Run with
`scripts/run_grid.py`; raw rows in `results/grid_1b_ctx1500.csv` and
`results/grid_1b_ctx3000.csv`. Each cell is correct / answer probability.

**Run A**: ~1.6K-token contexts, 36 needles per mode, 8 perplexity chunks.

Question after compression:

| method | 4% | 7% | 10% | 15% | 20% |
|---|---|---|---|---|---|
| Lexico, uniform | 0 / .00* | 0 / .00* | 0 / .00 | 1 / .00 | 16 / .20 |
| H2O | 1 / .00 | 1 / .00 | 0 / .00 | 0 / .00 | 0 / .00 |
| SnapKV | 1 / .00 | 2 / .00 | 2 / .00 | 4 / .01 | 6 / .03 |
| DimmerKV on Lexico, flat | 0 / .00 | 0 / .00 | 1 / .01 | 16 / .23 | 30 / .70 |
| kvtc | 33 / .77 | 35 / .93 | 35 / .95 | 36 / .97 | 36 / .98 |
| kvtc, filled | 35 / .79 | 35 / .94 | 34 / .94 | 36 / .97 | 36 / .98 |
| DimmerKV on kvtc, flat | 36 / .81 | 35 / .97 | 35 / .96 | 36 / .98 | 36 / .98 |
| DimmerKV on kvtc, question scores | 36 / .86 | 36 / .95 | 36 / .97 | 36 / .98 | 36 / .98 |

\* uniform Lexico cannot go below 1 atom per vector; it uses 5% there.

Question in the prompt:

| method | 4% | 7% | 10% | 15% | 20% |
|---|---|---|---|---|---|
| SnapKV | 10 / .20 | 16 / .31 | 24 / .49 | 24 / .57 | 30 / .63 |
| DimmerKV on Lexico, question scores | 7 / .15 | 16 / .34 | 23 / .46 | 33 / .79 | 35 / .92 |
| kvtc, filled | 34 / .79 | 35 / .93 | 35 / .95 | 36 / .98 | 36 / .98 |
| DimmerKV on kvtc, flat | 34 / .82 | 36 / .97 | 36 / .97 | 36 / .98 | 36 / .98 |
| DimmerKV on kvtc, question scores | **36 / .93** | 36 / .98 | 36 / .98 | 36 / .98 | 36 / .98 |

Perplexity (full 12.38): 4%: H2O 14.31, DimmerKV-Lexico flat 17.78, kvtc
filled 12.55, DimmerKV-kvtc flat 12.55, question scores 12.53. 10%: H2O
13.56, kvtc filled 12.39, DimmerKV-kvtc 12.40 / 12.36. 20%: H2O 13.10,
DimmerKV-Lexico flat 12.80, kvtc family 12.39.

**Run B**: ~3K-token contexts, 24 needles per mode, 6 perplexity chunks,
3–5% memory (20–33×). Full: 24/24, perplexity 11.89, prefill ~0.9 s.

| method | after 3% | after 3.5% | after 4% | after 5% | prompt 3% | prompt 3.5% | prompt 4% | prompt 5% | ppl 3% / 4% / 5% |
|---|---|---|---|---|---|---|---|---|---|
| H2O | 0 / .00 | 0 / .00 | 0 / .00 | 0 / .00 | 0 / .00 | 0 / .01 | 0 / .01 | 0 / .01 | 13.80 / 13.55 / 13.41 |
| SnapKV | 0 / .00 | 0 / .00 | 0 / .00 | 0 / .00 | 9 / .33 | 10 / .40 | 12 / .44 | 14 / .51 | 13.76 / 13.48 / 13.33 |
| kvtc, filled | 23 / .76 | 22 / .79 | 24 / .87 | 24 / .89 | 22 / .76 | 22 / .79 | 23 / .85 | 24 / .89 | 12.14 / 12.05 / 12.03 |
| DimmerKV on kvtc, flat | 23 / .78 | 23 / .86 | 23 / .87 | 24 / .94 | 22 / .80 | 24 / .88 | 24 / .88 | 24 / .94 | 12.15 / 12.06 / 12.02 |
| DimmerKV on kvtc, question scores | 22 / .80 | 23 / .87 | 24 / .89 | 24 / .92 | **24 / .91** | **24 / .94** | **24 / .94** | 24 / .94 | **12.07 / 12.03 / 11.98** |

Compression time per 3K-token context: kvtc ~240 ms, DimmerKV on kvtc ~350 ms.

What the grid shows:

- **Eviction collapses at high compression**: H2O 0/24 everywhere at 3–5%;
  SnapKV 9–14/24 even with the question visible.
- **kvtc-style coding keeps nearly everything** at 20–33×: 22–24/24 needles,
  perplexity +1–2%.
- **DimmerKV on kvtc beats kvtc at equal bytes in every needle column**, by
  answer probability. The largest gain is question-aware rungs when the
  question is visible (3%: .76 → .91; 3.5%: .79 → .94; Run A 4%: .79 → .93).
  Flat rungs alone add +.02–.07. On perplexity question-scored rungs are best
  at every budget in Run B, but by small margins (≤0.07).
- Correct counts sit near the ceiling for the kvtc family, so answer
  probability carries the comparison; differences of 1–2 needles are noise.
  Still one 1B model.
- Fixed during this run: `run_grid.py` printed budgets with 2 decimals, which
  merged 0.035 and 0.04; now `%g`, and the affected rows were rerun.

## Progress log (2026-10-24, ~10:10 PM IST): real storage + memory/time/accuracy benchmark

- `kvtc_storage.py`: kvtc / DimmerKV-TC caches serialized to real bytes
  (rung ids, bit-packed codes, FP16 shift/scale, exact tokens), DEFLATE via
  zlib, restored exactly (max abs diff 0.0). `KVTCCodec.compress` keeps the
  chosen rungs in `last_choice`.
- `bench_memory.py`: counted/stored/DEFLATE size, compress/DEFLATE/restore
  time vs re-prefill, needles (after) and perplexity per method.
- Preliminary (1 context): restore ~80-100 ms vs ~340 ms prefill; DEFLATE
  ~1.1x on top; stored size = counted + rung ids. Full run (1.5K/3K, 24
  needles) in progress -> results/memory_time_accuracy.csv. FINDINGS.md
  updated: new TL;DR (DimmerKV on kvtc), section 8, caveats, next steps.
- Full bench_memory run done (results/memory_time_accuracy.csv): 3K ctx,
  3%: kvtc family 37-41x after DEFLATE, 23/24 needles, ppl 11.58 vs 11.32;
  restore ~0.15 s vs prefill 0.9-3.8 s (laptop prefill timing unstable at 3K).
  H2O/SnapKV/Lexico at 10-20x: 0-1/24. FINDINGS.md section 8 updated.

## Progress log (2026-10-24, ~10:25 PM IST): serving benchmark

`scripts/bench_serving.py`. Scenario: a chat assistant keeps many
conversations' KV caches in host RAM between turns; a follow-up question
arrives for one of them. ~3.1K-token conversations, 24 questions (question
after compression), 32 generated tokens per answer, RAM budget 8 GiB.
Output: `results/serving.csv`.

| way of keeping the conversation | stored per conversation | conversations in 8 GiB RAM | restore | time to first token | decode tokens/s | extra GPU memory while answering | needles / ans p |
|---|---|---|---|---|---|---|---|
| full cache kept on GPU | 96.3 MiB (on GPU) | — | 0.7 ms | 21 ms | 65 | 236 MiB | 24/24 / .95 |
| full cache in RAM (FP16) | 96.3 MiB | 85 | 4.6 ms | 25 ms | 64 | 236 MiB | 24/24 / .95 |
| recompute (store only the text) | ~0 | unlimited | 840 ms | 891 ms | 61 | 2,658 MiB | 24/24 / .95 |
| H2O 5% | 4.9 MiB | 1,685 | 0.3 ms* | 22 ms | 58 | 139 MiB* | **0/24** / .00 |
| H2O 10% | 9.7 MiB | 846 | 0.6 ms* | 22 ms | 60 | 139 MiB* | **0/24** / .00 |
| kvtc filled 3% | 2.6 MiB | 3,127 | 146 ms | 171 ms | 60 | 596 MiB | 23/24 / .76 |
| **DimmerKV-kvtc 3%** | **2.6 MiB** | **3,127** | 154 ms | 176 ms | 61 | 534 MiB | 23/24 / **.78** |
| kvtc filled 5% | 4.3 MiB | 1,902 | 160 ms | 181 ms | 65 | 608 MiB | 24/24 / .89 |
| **DimmerKV-kvtc 5%** | 4.3 MiB | 1,896 | 166 ms | 188 ms | 64 | 534 MiB | 24/24 / **.94** |

\* H2O's restore only times the transfer of its kept tokens; its Level 1
cache stays on the GPU, so its restore time and extra GPU memory are
understated.

What it shows:

- **Capacity**: DimmerKV-kvtc at 3% fits **3,127 conversations** in 8 GiB of
  RAM where uncompressed FP16 fits 85 (**37×**), with 23/24 needles. H2O at a
  larger size (4.9 MiB) answers 0/24.
- **Latency**: time to first token from the compressed copy is ~0.18 s,
  **~5× faster than recomputing** the conversation (0.89 s), but slower than
  copying an uncompressed cache from RAM (25 ms). Compression pays when RAM
  (or network) is the limit, not when everything already fits.
- **Generation speed** is unchanged (60–65 tokens/s for all methods): after
  restore the model works on a normal FP16 cache.
- **GPU memory** while answering is not reduced (Level 1); decoding needs
  ~0.3 GiB of temporary space on top of the restored cache. Recompute needs
  2.6 GiB (eager attention over the whole context).
- **DimmerKV vs kvtc** at equal size: same capacity and speed (+6–7 ms to
  restore), higher answer probability (.78 vs .76 at 3%, .94 vs .89 at 5%).


## Progress log (2026-10-24, ~11:05 PM IST): long conversations (8K)

Prefill is now chunked (512 tokens at a time; scores summed across chunks,
checked against one-pass prefill: identical up to FP16 rounding), so 8K
contexts fit on the 8 GB laptop. All answering uses the same eager attention
so decode speeds compare like with like; the recompute baseline prefills in
1,024-token chunks with PyTorch SDPA, its fastest path here. 4 contexts × 6
needles = 24 questions, 32 generated tokens per answer, RAM budget 8 GiB.
Raw rows: `results/serving_8k.csv`, `logs/prompt_8k.log`.

**Serving, question after compression** (~8.1K tokens; FP16 cache 252 MiB):

| way of keeping the conversation | stored | conversations in 8 GiB | restore | time to first token | tokens/s | needles / ans p |
|---|---|---|---|---|---|---|
| full cache kept on GPU | 252 MiB (GPU) | — | 2 ms | 35 ms | 46 | 21/24 / .75 |
| full cache in RAM (FP16) | 252 MiB | 32 | 13 ms | 47 ms | 47 | 21/24 / .75 |
| recompute (store only text) | ~0 | unlimited | 1,358 ms | 1,384 ms | 45 | 21/24 / .75 |
| H2O 2% | 5.2 MiB | 1,586 | 0.3 ms* | 34 ms | 46 | **0/24** / .00 |
| H2O 5% | 12.7 MiB | 643 | 0.8 ms* | 35 ms | 46 | **0/24** / .00 |
| kvtc filled 2% | 4.5 MiB | 1,808 | 390 ms | 416 ms | 46 | 17/24 / .51 |
| DimmerKV-kvtc 2% | 4.5 MiB | 1,809 | 386 ms | 423 ms | 46 | 17/24 / .49 |
| kvtc filled 3% | 6.7 MiB | 1,220 | 403 ms | 433 ms | 46 | 20/24 / .58 |
| DimmerKV-kvtc 3% | 6.7 MiB | 1,217 | 410 ms | 449 ms | 46 | 20/24 / .58 |
| kvtc filled 5% | 11.2 MiB | 732 | 438 ms | 471 ms | 46 | 21/24 / .71 |
| DimmerKV-kvtc 5% | 11.2 MiB | 732 | 445 ms | 478 ms | 46 | **21/24 / .74** |

\* H2O's restore only times the transfer of its kept tokens (see section 9).

**Accuracy, question in the prompt** (24 questions; full cache 21/24, .75):

| method | 2% | 3% |
|---|---|---|
| SnapKV | — | 13 / .36 |
| kvtc filled | 17 / .51 | 21 / .60 |
| DimmerKV-kvtc flat | 17 / .51 | 21 / .63 |
| **DimmerKV-kvtc question scores** | **19 / .62** | **22 / .76** |

What it shows:

- The 1B model itself is weaker at 8K tokens (full cache 21/24, .75).
- **Capacity**: at 5%, DimmerKV-kvtc matches the full cache (21/24) while
  fitting **23× more conversations** in RAM (732 vs 32); H2O at the same size
  answers 0/24.
- **Latency**: first token in ~0.45–0.48 s from the compressed copy, ~3×
  faster than recomputing (1.38 s); slower than copying an uncompressed cache
  from RAM (47 ms). Decode speed is unchanged (45–47 tokens/s).
- **Question visible**: DimmerKV with question scores at 3% matches the full
  cache (22/24, .76 vs 21/24, .75) where kvtc gets .60 and SnapKV 13/24;
  at 2% it keeps 19/24 vs 17/24 for kvtc. The strongest DimmerKV result.
- **Question not visible**: at 8K tokens flat per-token rungs are tied with
  kvtc (the +.02–.05 gain at 1.6–3K tokens does not carry over).
- 16K-token run: first attempt ran out of pinned host memory (Windows limit)
  after the accuracy pass; rerunning with pageable RAM fallback.


## Progress log (2026-10-25, ~0:15 AM IST): progressive tiers, AWS 3B

Design (`scripts/progressive.py`, `fit_progressive.py`):

```
encode once (kvtc: RoPE removed -> cross-layer PCA -> 8-bit codes per group of 64)
document --prefill--> per-token DimmerKV prefix length --> bit-planes, ordered by error per byte
   hot tier (2-3% of FP16): each token's first planes      (DimmerKV picks how many)
   cold tier: the remaining planes of every token          (up to 8 bits everywhere)
question arrives -> score the hot cache with the question -> fetch cold planes for the
                    top tokens only -> rebuild those tokens -> answer
memory pressure  -> demote a conversation by dropping planes (no re-encode)
```

- Codes are nested: the top m bits of an 8-bit code are a valid m-bit code
  (decoded at the bin centre), so precision is cut by truncating planes.
- The (group, plane) order is learned once on calibration data by error
  removed per byte, scoring a group by its best run of upcoming planes (a
  dominant key component otherwise had its first plane postponed).
- **The progressive code is more accurate than kvtc's 0/2/4/8-bit ladder at
  every size** (1B, calibration sample): keys 0.071 vs 0.080 rel. error at
  1/50 of FP16, values 0.071 vs 0.085 at 1/16; finer 1-bit steps help.
- Methods: `prog@b` (hot tier only, DimmerKV flat), `prog@b:snap`,
  `prog@b:r<frac>` (refine the top `frac` of tokens at question time).
- Encode ~0.6 s + hot rebuild ~0.34 s for an 8K-token context on the laptop
  after an optimization (per-group error lookup, int32 decode); comparable
  to kvtc.

**1B, 8K tokens, question asked after compression** (24 questions; full
cache 21/24, ans p .749; `logs/prog_8k_after.log`):

| method | hot tier | fetched per question | correct | ans p |
|---|---|---|---|---|
| H2O | 2% | — | 0/24 | .001 |
| kvtc filled | 2% | — | 17/24 | .511 |
| DimmerKV-kvtc | 2% | — | 17/24 | .492 |
| progressive, hot only | 2% | — | 18/24 | .499 |
| **progressive + refine top 1%** | 2% | **0.5% of FP16** | **19/24** | **.705** |
| progressive + refine top 2% | 2% | 1.0% | 19/24 | .674 |
| progressive + refine top 5% | 2% | 2.6% | 19/24 | .645 |
| kvtc filled | 3% | — | 20/24 | .578 |
| progressive + refine top 2% | 3% | 1.0% | 19/24 | .685 |

- **Refinement fixes most of the "question after compression" weak spot**:
  fetching the top 1% of tokens (0.5% extra bytes) lifts answer
  probability from ~.50 to .705 (full cache .749), above kvtc at 3%.
  Correct answers rise from 17 to 19/24 (full 21/24).
- Refining more tokens did not help further (.705 → .674 → .645); likely
  noise at 24 questions, possibly the top 1% already holds what matters.
- On 3K tokens (6 questions): hot 3% .913, + refine 2% .954, full .966.

**Llama-3.2-3B on AWS (L40S), 8K tokens, question after compression**
(36 questions; first task of the AWS grid; `results/aws_3b_ctx8000_needle_after.txt`).
kvtc calibration for 3B (28,672 features per token) took 5.1 min.

| method | 2% | 3% | 5% |
|---|---|---|---|
| full cache | 36/36 (.991) | | |
| H2O | 0/36 (.000) | 0/36 (.000) | 0/36 (.000) |
| SnapKV | 1/36 (.000) | 1/36 (.000) | 1/36 (.000) |
| kvtc filled | 36/36 (.988) | 36/36 (.991) | 36/36 (.992) |
| DimmerKV-kvtc | 36/36 (.989) | 36/36 (.992) | 36/36 (.992) |
| progressive, hot only | 36/36 (.986) | 36/36 (.991) | 36/36 (.992) |
| progressive + refine 2% | 36/36 (.993) | 36/36 (.992) | 36/36 (.991) |

- **The 3B model is far more robust**: every kvtc-family method keeps
  36/36 at 2% of FP16 (50× smaller) on 8K tokens; eviction answers 0–1/36.
- This test is too easy for 3B to separate the variants (all ≥ .983); it
  needs budgets below 2%, which needs a zero-byte (eviction) rung, or
  harder tasks. The rest of the AWS grid (question in the prompt,
  perplexity) and a 16K-token serving run are still running.


AWS setup: `ubuntu@34.203.234.146` (g6e.2xlarge, L40S 46 GB), key
`~/.ssh/hashcode-key.pem`. Our code in `~/dimmerkv` (copied with
`git archive`), env `~/dimmerkv-env` (system PyTorch 2.14 + transformers
5.20.2, datasets 5.1.0). Pipeline `~/dimmerkv/run_3b.sh` under nohup:
fit_kvtc → fit_progressive → run_grid (8K, both modes + ppl) →
bench_serving 16K; logs in `~/dimmerkv/logs`, results in
`~/dimmerkv/results`. Leave the team's `~/lexico` and `~/lexico-env` alone.

## Progress log (2026-10-25, ~0:30 AM IST): eviction level, faster restore, 3B 16K

**Partial-basis decode.** Rebuilding the cache multiplied every token by the
full features × features PCA basis, although at low budgets most groups of
components carry 0 bits. `kvtc.inverse_transform` now multiplies only the
groups that are non-zero for some token. 1B, 8K tokens: restore 390 → **93 ms**
at 2% (11 of 128 key groups used) and 234 ms at 5%; the restored cache is
identical. Applied to kvtc, byte storage and the progressive codec.

**Eviction level (`dkvtc@b:ev`).** A zero-byte level below the coarsest
rung: an evicted key hides the token from attention (its value is then
dropped too); an evicted value alone decodes to the mean. This lets DimmerKV
go below kvtc's minimum (~2% of FP16 at 8K tokens) and mix dropping with
graded precision in one budget. At budgets that do not need it, it changes
nothing (identical results at 2–3% without a question).

1B, 8K tokens, 24 questions (`logs/evict_8k_*.log`; full cache 21/24, .756):

| question | method | memory | correct | ans p |
|---|---|---|---|---|
| in prompt | SnapKV | 1.5% | 6/24 | .182 |
| in prompt | **DimmerKV-kvtc, question scores + eviction** | **1.5% (67×)** | **21/24** | **.704** |
| in prompt | SnapKV | 2% | 10/24 | .295 |
| in prompt | DimmerKV-kvtc, question scores | 2% | 19/24 | .607 |
| in prompt | **DimmerKV-kvtc, question scores + eviction** | **2% (50×)** | **22/24** | **.765** |
| after | H2O | 1.5% | 0/24 | .001 |
| after | DimmerKV-kvtc + eviction (flat) | 1.5% | 1/24 | .004 |
| after | kvtc filled / DimmerKV-kvtc (± eviction) | 2% | 17/24 | .48–.50 |

- **With the question visible, DimmerKV with eviction matches the full cache
  at 1.5–2% of its memory (50–67× smaller)**: it drops 14–38% of tokens the
  question does not need, spends the bytes on finer precision for relevant
  ones, and keeps the rest coarse. SnapKV (keep-or-drop with the same scores)
  keeps 6–10/24.
- Without the question, eviction does not help: nothing tells it what is
  safe to drop. That case is handled by progressive refinement (section 11).

**3B on AWS, 16K-token conversations** (18 questions, question after
compression, before the partial-basis decode; `results/serving_3b_16k.csv`):

| way of keeping it | stored | conversations in 8 GiB | time to first token | correct |
|---|---|---|---|---|
| full cache in RAM | 1,753 MiB | 4 | 189 ms | 18/18 |
| recompute (chunked SDPA) | ~0 | unlimited | 1,434 ms | 18/18 |
| H2O 2% | 35 MiB | 230 | 61 ms | 0/18 |
| kvtc / DimmerKV-kvtc 2% | 31 MiB | 263 | ~1,670 ms | 18/18 |
| kvtc / DimmerKV-kvtc 5% | 78 MiB | 104 | ~1,960 ms | 18/18 |

- 57× more conversations per GiB with every answer correct; H2O answers none.
- On the L40S, restoring from the compressed copy with the full-basis decode
  was slower than re-running prefill (1.6–1.9 s vs 1.4 s); this motivated the
  partial-basis decode above. A rerun with it is queued on AWS.


## Progress log (2026-10-25, ~2:35 AM IST): attention directly over the compressed cache

`scripts/compressed_attention.py` (+ hook in `dimmer.py`, coefficients exposed
by `kvtc.py`): keys rebuilt one layer at a time, values used in coefficient
space, exact tokens in a small cache. 1B, 8K tokens: same answers 60/60;
GPU memory held per conversation 253 MB -> 18-31 MB at 2%; ~0.9x speed at 2%,
0.6-0.7x at 5%. Details in FINDINGS.md section 13; test:
`scripts/test_compressed_attention.py`.

## Progress log (2026-10-25, ~3:40 AM IST): demo direct mode, RULER-style tasks

- Demo: toggle "Answer directly from the compressed cache" (on by default);
  cards and leaderboard show GPU memory held while answering; eviction methods
  report their kept-token size.
- `scripts/eval_ruler.py`: variable tracking (8K) and key-value retrieval (4K),
  20 samples, both modes. Variable tracking: DimmerKV question-aware 0.92 vs
  kvtc 0.82 (full 0.96). Key-value retrieval: kvtc 0.65-0.70 at 5% beats
  DimmerKV 0.35-0.55 (dense, equally important tokens). Eviction collapses on
  both. FINDINGS.md section 14; results/ruler_*.txt.

## Progress log (2026-10-25, ~4:10 AM IST): precision floor, demo backup, review deck

- `dkvtc@b:snap:floor1` (kvtc's uniform rung for every token, leftover by
  question score) ties kvtc on key-value retrieval (0.65 vs 0.70/0.65) and keeps
  the variable-tracking lead (0.90-0.91 vs 0.81-0.82). Partial floors and key
  weighting did not help. FINDINGS.md section 15.
- `docs/demo_backup.html` from `scripts/demo_backup.py` (recorded demo scenarios).
- Review 2 deck: https://claude.ai/artifact/BefpqPTgntqBAiG4ax3cm7 (private).
