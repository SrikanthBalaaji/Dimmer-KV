# DimmerKV

**Dim the KV cache instead of deleting it.** DimmerKV shrinks a language model's memory of what it has
read (the KV cache) to 1.5–5% of its normal size while keeping the facts a later question needs. It
keeps every token at a precision chosen for that token, and sharpens the right tokens when the
question arrives.

Built at Hashcode 14.0 (Track 01, LLM Inference) by Team Deadoop: Srikanth Balaaji, Sushanth B,
T S Sathya Kiran.

| | |
|---|---|
| **50–67× smaller** | cache that still matches the full cache on hidden-fact questions (question known, 8K tokens) |
| **57× more conversations per GiB** | 3B model, 16K-token conversations, 18/18 answers correct |
| **0 / 36** | answers that H2O-style eviction gets when the question comes after compression |
| **2.3× faster** first answer token | than re-reading the conversation (3B, 16K tokens) |

Full write-up with the architecture diagram: [`docs/DimmerKV_overview.html`](docs/DimmerKV_overview.html)
(download and open in a browser). Every experiment and number: [`FINDINGS.md`](FINDINGS.md).
Recorded demo scenarios (if the live demo cannot run): [`docs/demo_backup.html`](docs/demo_backup.html).

---

## The problem

For every token a model reads, it stores a key and a value vector in every layer and attention head.
On Llama-3.2-3B that is about 112 KB per token, so a 16K-token conversation needs **1.75 GB**. A
server that keeps conversations between turns runs out of memory fast, and rebuilding a conversation
from its text means re-reading all of it.

The usual fix is **eviction** (H2O, SnapKV): keep the tokens that got the most attention, delete the
rest. That fails when the question comes *after* compression. Before the question exists, the facts
it will need were never heavily attended. In our tests the hidden facts ranked at the 30–35th
percentile of H2O's attention scores, so H2O deleted them and answered **0 of 36** questions on the 3B
model, where the full cache answered 36 of 36.

## Our solution

DimmerKV does not make a keep-or-delete decision. Every token keeps some precision ("brightness"),
and a single byte budget is spent where it removes the most error.

1. **Calibrate once per model** (kvtc-style). For each token, undo RoPE on the keys and join all layers
   and heads into one vector (8,192 numbers on 1B, 28,672 on 3B), likewise the values. Fit one PCA
   basis on ~100K WikiText tokens. Learn an order of (group of 64 components, bit-plane) pieces by
   error removed per byte.
2. **Compress once per conversation.** Quantize every token once at 8 bits per group. The
   **DimmerKV allocator** measures each token's error at every precision step and hands out
   upgrades greedily by error removed per byte, from "dropped" (0 bytes) through coarse to fine bit-planes.
   - Without the question, tokens are weighted equally (**flat**): bytes go to tokens that are hard to compress.
   - With the question in the prompt, errors are weighted by the **question's attention**, so relevant tokens get finer precision.
   - **Recommended setting (`dkvtc@b:snap:floor1`)**: every token first gets kvtc's uniform precision (a
     floor), and only the leftover bytes are spent by question score. This is never worse than kvtc on
     the tasks we tested.
   - The first 4 and last 32 tokens stay exact.
3. **Store in tiers.** Bit-pack and DEFLATE. A **hot** tier holds each token's first bit-planes; a **cold**
   tier holds the rest. Under memory pressure a conversation is demoted by dropping planes, with no
   re-encoding.
4. **Restore and refine per question.** Decode only the PCA groups that carry bits and re-apply RoPE.
   When the question arrives, score the restored cache with it and **fetch the full bit-planes for its
   top ~1% of tokens**.
5. **Answer, without rebuilding the cache.** Attention runs over the compressed tokens directly: keys are
   rebuilt one layer at a time and discarded, values are never decoded (attention weights the small
   coefficient vectors, then applies the basis once per head). Answers are identical to the
   fully decoded cache, and GPU memory while answering drops 8–14× at a 2% budget.

## What is new

| | Eviction (H2O, SnapKV) | kvtc (ICLR 2026) | **DimmerKV** |
|---|---|---|---|
| Unit of decision | keep or delete a token | same precision for every token | **precision per token**, including 0 |
| Importance signal | attention (before the question) | none | **error per byte**, plus the question's attention when known |
| Question after compression | facts lost | uniform loss | **question-time refinement** fetches detail for the tokens the question needs |
| Coding | none | fixed 0/2/4/8-bit allocation | **progressive bit-planes**: one encoding, any precision by truncation |
| Storage | single tier | single tier | **hot/cold tiers**, demote by dropping planes |

Specific contributions:
- **Per-token precision on top of transform coding.** kvtc gives every token the same bits; DimmerKV
  allocates bit-planes per token under one budget and beats kvtc at equal bytes.
- **A precision floor at kvtc's level.** Per-token precision alone lost to kvtc on dense lookups where
  every token matters equally; giving every token kvtc's precision first and spending only the leftover
  by question score ties kvtc there and keeps the lead on multi-hop tracking.
- **Eviction as a precision level.** A zero-byte level lets one allocator mix dropping and graded
  precision. With question scores this matched the full cache at 1.5–2% of its memory.
- **Progressive bit-plane coding.** Nested 8-bit codes ordered by error per byte are 10–30% more
  accurate than kvtc's fixed bit ladder at the same size, and they make tiers and refinement possible.
- **Question-time refinement** (score the coarse cache, then fetch detail for the top tokens). It fixes
  most of the gap when the question comes after compression, for 0.5% extra bytes.
- **Partial-basis decode.** Only PCA groups that carry bits are multiplied, so restoring is 4× faster
  and beats re-reading the conversation.
- **Attention in the compressed domain.** Keys decoded per layer on the fly and values used directly in
  coefficient space, so the full FP16 cache never exists while answering.

## Performance

Llama-3.2-1B on an RTX 5060 laptop (8 GB); Llama-3.2-3B on an AWS L40S. Budgets are the compressed
size as a share of the FP16 KV cache (2% = 50× smaller). "ans p" is the model's average probability
of the correct answer. The test hides facts such as *"The access code for the Orion locker is 7311"*
in WikiText and asks about them.

**Head to head: H2O alone, Lexico alone, DimmerKV** (1B, ~3.1K-token conversations, FP16 cache
96 MiB, 24 questions asked after compression, RTX 5060 laptop; `results/memory_time_accuracy.csv`,
`results/serving.csv`)

| method | stored per conversation | smaller by | conversations in 8 GiB RAM | compress | restore | first answer token | tokens/s | correct | perplexity |
|---|---|---|---|---|---|---|---|---|---|
| Full cache in RAM | 96.3 MiB | 1× | 85 | — | 5 ms | 25 ms | 64 | 24/24 | 11.32 |
| Recompute from the text | ~0 | — | unlimited | — | 840 ms (prefill) | 891 ms | 61 | 24/24 | 11.32 |
| H2O 5% | 4.9 MiB | 20× | 1,685 | 28 ms | <1 ms\* | 22 ms | 58 | 0/24 | 12.72 |
| H2O 10% | 9.7 MiB | 10× | 846 | 30 ms | <1 ms\* | 22 ms | 60 | 0/24 | 12.29 |
| Lexico 10% (uniform) | 9.2 MiB† | 10× | ~890† | 362 ms | ~30–45 ms‡ | — | — | 0/24 | 13.90 |
| **DimmerKV 3%** | **2.6 MiB** | **37×** | **3,127** | 348 ms | 154 ms | 176 ms | 61 | **23/24** | **11.58** |
| **DimmerKV 5%** | 4.3 MiB | 22× | 1,896 | 360 ms | 166 ms | 188 ms | 64 | **24/24** | **11.45** |

DimmerKV here is the question-unaware setting (`dkvtc@b`, sizes after bit-packing and DEFLATE).
\* H2O's restore only copies its kept tokens; its cache layout is simulated, so this is a lower bound.
† Lexico's size is counted from its codes (no byte format written); conversations computed from it.
‡ Lexico decode time from the codec benchmark (9–15 ms per 1K tokens); Lexico was not in the serving
benchmark, so it has no first-token or tokens/s figure.

- **Speed: H2O is fastest.** It only drops tokens (about 30 ms to compress, 22 ms to the first token).
  DimmerKV takes about 0.35 s to compress and 0.18 s to the first token, still about 5× faster than
  re-reading the conversation. Lexico compresses at about the same speed as DimmerKV at this length,
  but its sparse search (OMP) grows with the number of atoms (0.35–0.67 s per 1K tokens at 8–16 atoms).
  Generation speed is the same for all methods (58–65 tokens/s), because answering here uses a normal
  FP16 cache.
- **Memory: DimmerKV is smallest and the only one that keeps the answers.** At 3% it stores less than
  H2O at 5% or Lexico at 10%, fits 1.9× and 3.5× more conversations than them, answers 23/24 and stays
  within 2.3% of full perplexity. H2O and Lexico answer 0/24 at these sizes. H2O failed at every budget
  we tried up to 20% when the question comes later; uniform Lexico needed 22–27% of FP16 to answer
  10–16 of 18 (1.6K tokens).
- **GPU memory while answering** (8K tokens, 2%): H2O holds only its kept tokens (about 5 MiB);
  DimmerKV answering directly over the codes holds 18–31 MB, against 253 MB for the full cache. H2O is
  smaller here, but it has deleted the facts.

**Question known before compression** (8K-token documents)

| method | 1B, 1.5% | 1B, 2% | 3B, 2% |
|---|---|---|---|
| Full cache (100%) | 21/24 | 21/24 | 36/36 · .992 |
| SnapKV (evict by question attention) | 6/24 | 10/24 | 34/36 · .913 |
| kvtc (same precision for all) | — | 17/24 | 35/36 · .978 |
| **DimmerKV, question scores (+ eviction on 1B)** | **21/24** | **22/24** | **36/36 · .986** |
| **DimmerKV progressive + refinement** | — | — | **36/36 · .992** |

**Question asked after compression** (8K-token documents)

| method | 1B, 2% | 3B, 2% |
|---|---|---|
| Full cache | 21/24 · .75 | 36/36 · .99 |
| H2O | 0/24 · .00 | 0/36 · .00 |
| kvtc | 17/24 · .51 | 36/36 · .99 |
| **DimmerKV progressive, refine top 1% (+0.5% fetched)** | **19/24 · .71** | **36/36 · .99** |

**Serving: 3B model, 16K-token conversations, 18 questions**

| how a conversation is kept | stored | conversations in 8 GiB RAM | first answer token | correct |
|---|---|---|---|---|
| Full cache in RAM | 1,753 MiB | 4 | 189 ms | 18/18 |
| Re-read the text (chunked SDPA prefill) | ~0 | unlimited | 1,434 ms | 18/18 |
| H2O, 2% | 35 MiB | 230 | 61 ms | **0/18** |
| **kvtc / DimmerKV, 2%** | **31 MiB** | **263 (57×)** | **488–627 ms** | **18/18** |

Generation speed is the same for every method (about 26 tokens/s on 3B, 46 on 1B) when the cache is
rebuilt before answering.

**Answering directly from the compressed cache** (1B, 8K tokens, 12 questions per method)

| method | same answers as rebuilt cache | GPU memory held per conversation | tokens/s (rebuilt → direct) |
|---|---|---|---|
| kvtc 2% | 12/12 | 253 MB → **18 MB** | 45 → 41 |
| DimmerKV 2% | 12/12 | 253 MB → **22 MB** | 45 → 39 |
| DimmerKV + question scores + eviction 2% | 12/12 | 253 MB → **31 MB** | 31 → 31 |
| kvtc 5% | 12/12 | 253 MB → 55 MB | 43 → 29 |

**RULER-style tasks (1B, 20 samples per cell)**: the two long-context tasks the kvtc paper reports

| method | variable tracking 8K, 3%, question in prompt | variable tracking 8K, 2%, question later | key-value retrieval 4K, 5%, question later |
|---|---|---|---|
| Full cache | 0.96 | 0.96 | 0.65 |
| H2O | 0.15 | 0.02 | 0.00 |
| SnapKV | 0.34 | 0.00 | 0.00 |
| kvtc | 0.82 | 0.82 | **0.70** |
| DimmerKV + question scores + eviction | **0.92** | 0.84 | 0.40 |
| DimmerKV progressive + refine top 1% | 0.87 | **0.87** | 0.55 |
| **DimmerKV + question scores + floor at kvtc's level** | 0.91 | **0.90** | **0.65** |

DimmerKV wins where importance is uneven (multi-hop chains in filler). On a dense table of random
UUIDs, where every token matters equally, plain per-token allocation lost to kvtc; giving every token
kvtc's precision first and spending only the leftover by question score (`dkvtc@b:snap:floor1`)
ties kvtc there while keeping the lead on variable tracking.

**Perplexity: 3B, 8K tokens** (full cache 8.33; lower is better)

| method | 2% | 3% | 5% |
|---|---|---|---|
| H2O | 10.64 | 10.41 | 10.11 |
| SnapKV | 10.25 | 9.92 | 9.51 |
| kvtc | 8.48 | 8.42 | 8.34 |
| **DimmerKV, question scores** | **8.46** | **8.37** | 8.35 |

**Improvements along the way** (same tests)

| change | before | after |
|---|---|---|
| Per-token allocation instead of uniform (Lexico, 15%, question later) | 5/18 | 10/18 |
| kvtc transform instead of Lexico (1B, 4%, question later) | 0/36 | 33–36/36 |
| Progressive bit-planes vs kvtc's bit ladder (reconstruction error, 1B) | — | 10–30% lower at equal size |
| Question-time refinement (1B, 8K, 2%, question later) | ans p .50 | ans p .71 (full .75) |
| Question scores + eviction level (1B, 8K, 1.5%, question known) | SnapKV 6/24 | 21/24 (= full cache) |
| Partial-basis decode (restore time, 1B 8K / 3B 16K, 2%) | 390 ms / 1,597 ms | 93 ms / 416 ms |
| Faster OMP for Lexico (encode time) | 70 ms/stream | 33 ms/stream |
| Attention over codes instead of a rebuilt cache (GPU memory while answering, 1B 8K, 2%) | 253 MB | 18–31 MB |

## Run it

```powershell
# Python 3.11, torch 2.11 (CUDA), transformers 5.20.2, datasets 5.1.0, gradio 6.30
$py = "$env:USERPROFILE\.venvs\hashcode\Scripts\python.exe"
# One-time calibration (checkpoints are not in git; refit per model)
& $py scripts\fit_kvtc.py                # checkpoints\kvtc.pt (~1.3 min on 1B)
& $py scripts\fit_progressive.py         # checkpoints\progressive.pt
# Demo: http://127.0.0.1:7860
& $py scripts\demo_app.py
# Evaluations
& $py scripts\test_generation.py --context-tokens 8000 --configs kvtcf@0.02 prog@0.02:r0.01 h2o:0.02
& $py scripts\test_generation.py --mode prompt --context-tokens 8000 --configs h2o:0.02:snap dkvtc@0.02:snap:ev
& $py scripts\eval_perplexity.py --configs kvtcf@0.03 dkvtc@0.03:snap h2o:0.03
& $py scripts\bench_serving.py --context-tokens 8000 --methods h2o:0.02 kvtcf@0.02 dkvtc@0.02
& $py scripts\eval_ruler.py --task vt --context-tokens 8000 --configs kvtcf@0.03 dkvtc@0.03:snap:ev h2o:0.03
& $py scripts\eval_ruler.py --task kv --context-tokens 4000 --configs kvtcf@0.05 dkvtc@0.05:snap:floor1
& $py scripts\test_compressed_attention.py --context-tokens 8000 --methods kvtcf@0.02 dkvtc@0.02
& $py scripts\run_grid.py --model meta-llama/Llama-3.2-1B-Instruct --budgets 0.02 0.05 --templates "h2o:{b}" "kvtcf@{b}" "dkvtc@{b}"
```

Method names (`h2o:0.02`, `kvtcf@0.03`, `dkvtc@0.02:snap:ev`, `prog@0.02:r0.01`, …) are documented at
the top of [`scripts/methods.py`](scripts/methods.py).

## Repository layout

| path | what it is |
|---|---|
| `scripts/kvtc.py`, `fit_kvtc.py` | kvtc-style transform (RoPE removal, cross-layer PCA, grouped quantization), DimmerKV per-token allocation, eviction level |
| `scripts/progressive.py`, `fit_progressive.py` | progressive bit-planes, hot/cold tiers, question-time refinement |
| `scripts/kvtc_storage.py` | real byte storage (bit-packing, DEFLATE) and partial-basis restore |
| `scripts/dimmer.py` | attention hook: H2O / question-window scores, per-head eviction masks, chunked prefill |
| `scripts/compressed_attention.py` | attention directly over the compressed cache (keys per layer, values never decoded) |
| `scripts/methods.py` | one name per method, shared by every script and the demo |
| `scripts/test_generation.py`, `eval_perplexity.py`, `bench_serving.py`, `bench_memory.py`, `run_grid.py` | needle tests, perplexity, serving benchmark, storage/time benchmark, CSV grid runner |
| `scripts/demo_app.py`, `demo_backup.py` | Gradio demo (scenarios, side-by-side result cards, all-methods leaderboard, per-token heat map) and its static backup |
| `scripts/lexico.py`, `pca_codec.py`, … | earlier codecs, kept as ablations |
| `docs/DimmerKV_overview.html` | overview page with the architecture diagram |
| `FINDINGS.md`, `results/` | every experiment, and the raw result CSVs |

## Limits

- **Answering directly from the compressed cache** works for the kvtc family and gives identical answers,
  but costs speed at larger budgets (about 0.9× at 2%, 0.6–0.7× at 5%). The progressive codec still
  rebuilds an FP16 cache, and prefill still builds the full cache once before compressing.
- **Without the question, per-token precision alone is only slightly better than kvtc** (often tied at
  8K tokens); the clear gains come from question scores and question-time refinement.
- **Per-token precision can lose to uniform kvtc** when every token matters equally: on RULER-style
  key-value retrieval (random UUIDs) kvtc scores 0.65–0.70 at 5% vs 0.35–0.55 for DimmerKV variants.
  A floor at kvtc's precision plus question scores ties kvtc (0.65); the flat variant does not.
- **The 3B model is near the ceiling** of our needle test at 2%, so methods are hard to separate there.
  Multi-hop variable tracking is done (above); 8B and LongBench are next.
- **Our kvtc is simplified**: greedy bit allocation instead of the paper's dynamic program, fixed groups
  of 64, zlib on the CPU. The numbers are ours, not a reproduction of the paper.
- **The recommended floor setting is only tested on the RULER-style tasks so far** (20 samples each);
  the needle and perplexity tables above come from the earlier settings.
- **Question-time refinement counts its fetch but does not move it yet**: the cold tier sits in the same
  GPU tensors as the hot tier, so a real copy from host RAM or disk is not in the timings, and scoring
  runs the question through the model once before answering.
- **Small samples**: 18–36 questions or 20 RULER samples per setting, so differences of 1–2 answers
  (about 0.1 on RULER) are noise.

## Credits

- **kvtc**: Staniszewski & Łańcucki, *KV Cache Transform Coding for Compact Storage in LLM Inference*,
  ICLR 2026. Cross-layer PCA, RoPE removal and grouped quantization come from this paper.
- **H2O** (Zhang et al., NeurIPS 2023) and **SnapKV**: eviction baselines, and the attention-score ideas
  behind our importance weights.
- **Lexico**: sparse dictionary coding of KV vectors; our first codec and an ablation.
- **Loki** and **Mooncake**: score-then-fetch and tiered KV storage, which shaped question-time
  refinement and the hot/cold tiers.
