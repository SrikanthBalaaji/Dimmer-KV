# DimmerKV — experimental findings so far

Status as of Sat 24 Oct 2026, ~10:30 PM IST. Model: Llama-3.2-1B-Instruct (FP16,
16 layers, 8 KV heads, head dim 64) on the LOQ laptop (RTX 5060 8 GB).
Branch: `codec-rework`. Raw numbers and the run-by-run log are in `CLAUDE.md`;
logs in `logs/` (not committed).

## TL;DR

1. **Lexico beats PCA at every memory budget.** At 27% of FP16 KV memory,
   uniform Lexico (11 atoms) answers 16/18 needle questions; uniform PCA
   (32 components) answers 1/18.
2. **Keys are the fragile stream.** Mixing codecs (PCA keys + Lexico values,
   or the reverse) is always worse than Lexico for both.
3. **H2O scores cannot see a question asked after compression.** The hidden
   facts rank at the 30–35th percentile of H2O scores. Plain H2O at 15%
   memory: 0/18. Our allocator driven by H2O scores: 0/18.
4. **The allocator works when it ignores scores** and gives each token the
   atoms its own reconstruction error needs (`flat`): **10/18 at 15%**
   (uniform Lexico: 5/18) and **14/18 at 20%** (uniform Lexico needs 22% for
   9/18).
5. **Cost: compression time.** The allocator takes ~1.3 s per 1.6K-token
   context after the OMP speed-up (was ~2.4 s; prefill itself ~0.6 s). Answer speed is unchanged because Level 1
   rebuilds a normal cache; memory savings are *counted* from the codes.

This changes the pitch from "keep important tokens bright" to
**"give each token the bytes its own content needs"**. Attention-based
importance is still useful, but only once the question is known (see Next
steps).

## How we test

**Needle test (main metric)** — `scripts/test_generation.py`

- 6 facts with random 4-digit numbers ("The access code for the Orion locker
  is 7311.") are inserted into ~1,500 tokens of WikiText-2 (test split).
- The model reads only the context; its KV cache is compressed; *then* each
  question is fed on top of the compressed cache. The question is genuinely
  after compression, so the compressor cannot know what matters.
- 3 contexts × 6 facts = 18 questions per config, greedy decoding, same
  prompts for every config. Full cache scores 18/18.
- The first token (attention sink) and the last 32 context tokens stay exact
  in every config.

Metrics:

| metric | meaning |
|---|---|
| correct | the expected number appears in the reply (first 12 tokens) |
| agree | the reply equals the full-cache reply exactly |
| ans p | exp(−mean NLL) of the correct answer tokens, teacher-forced; smooth, so it separates configs with equal accuracy |
| bytes | counted compressed bytes / FP16 KV bytes of the context |
| compress ms | encode + allocation + decode for the context, CUDA-synced |

18 questions is small: differences of 1–2 are noise; gaps like 1 vs 16 are
real.

**Codec benchmark** — `scripts/bench_codecs.py`: reconstruction error on
held-out WikiText cache vectors, bytes per vector, encode/decode time per 1K
tokens.

## 1. Codecs

**PCA**: a fixed basis per (layer, KV head), fitted on 16K WikiText tokens
(BOS excluded), full 64×64 ordered basis stored so any prefix length k
works. INT8 coefficients + one FP16 scale per vector (k + 2 bytes).
Per-head bases beat one shared basis per layer everywhere; INT8 costs no
measurable quality versus FP16.

**Lexico**: per-layer dictionary of 4,096 unit atoms for K and for V
(trained with OMP + Adam, 1,500 steps, ~19 min). Each vector picks its own s
atoms by OMP. 12-bit coefficient + 12-bit index per atom, plus an FP16 scale:
3s + 2 bytes. The 12-bit rounding costs almost nothing.

Held-out reconstruction (relative MSE, lower is better; FP16 vector = 128 B):

| codec | bytes/vector | share of FP16 | key err | value err | encode ms / 1K tok | decode ms / 1K tok |
|---|---|---|---|---|---|---|
| PCA k=16 | 18 | 14% | 0.171 | 0.460 | 18 | 11 |
| PCA k=32 | 34 | 27% | 0.065 | 0.256 | 14 | 9 |
| PCA k=48 | 50 | 39% | 0.019 | 0.109 | 26 | 15 |
| Lexico s=4 | 14 | 11% | 0.110 | 0.263 | 287 | 9 |
| Lexico s=8 | 26 | 20% | 0.047 | 0.104 | 581 | 12 |
| Lexico s=11 | 35 | 27% | 0.025 | 0.050 | 838 | 15 |
| Lexico s=16 | 50 | 39% | 0.008 | 0.014 | 1304 | 18 |

Lexico at 4 atoms matches PCA at 32 components on values with less than half
the bytes. Encoding is 20–80× slower than PCA; decoding costs the same.

Low reconstruction error does not guarantee correct answers: PCA k=48 (key
error 0.019) still loses facts. Always check the needle test.

## 2. Uniform codes and mixed K/V

Every middle token gets the same code (18 questions):

| memory | config (key/value) | correct | ans p | compress ms |
|---|---|---|---|---|
| ~14% | PCA16 | 0/18 | 0.001 | 27 |
| | Lexico 5 | 3/18 | 0.024 | 671 |
| | PCA16 / Lexico 5 | 0/18 | 0.001 | 293 |
| ~20% | **Lexico 8** | **10/18** | **0.267** | 948 |
| | Lexico 8 / PCA24 | 4/18 | 0.053 | 476 |
| | PCA24 / Lexico 8 | 1/18 | 0.004 | 484 |
| ~27% | PCA32 | 1/18 | 0.023 | 13 |
| | **Lexico 11** | **16/18** | **0.789** | 1372 |
| | Lexico 11 / PCA32 | 10/18 | 0.248 | 697 |
| | PCA32 / Lexico 11 | 9/18 | 0.105 | 698 |
| | PCA16 / Lexico 16 | 0/18 | 0.000 | 1100 |

(These runs used no exact recent window, so bytes are slightly lower than in
section 3.)

Why keys matter most: attention scores are exponentiated dot products with
the keys, so a small key error can send attention to the wrong sentence. A
value error only blurs what is read from the right place.

## 3. Allocator and H2O

`scripts/dimmer.py`:

- A custom attention function (registered with transformers'
  AttentionInterface, same math as eager) records H2O scores during prefill
  (attention each token received, summed over queries and the 4 query heads
  sharing each KV head) and hides evicted tokens per (layer, KV head).
- The allocator gives every middle (layer, KV head, token) slot one level:
  evict, Lexico with {1, 2, 3, 4, 6, 8, 11, 16} atoms for both K and V, or
  exact. Piece value = score × drop in relative K+V error / extra bytes;
  `cummin` keeps codes as prefixes; a greedy fill stops at the byte budget,
  and the code asserts the budget is never exceeded.
- The H2O baseline is the same allocator with only {evict, exact}.

**Why H2O fails here** (`scripts/diagnose_scores.py`): needle answer tokens
rank at the 31–35th percentile of H2O scores in all three contexts, below
the average token. Dividing by the number of queries that could see each
token gives ~45th percentile, i.e. random. Before the question, nothing makes
the model attend to "7311".

Results (18 questions, sink + last 32 tokens exact everywhere):

| memory | config | correct | ans p | compress ms |
|---|---|---|---|---|
| 10% | uniform Lexico 3 | 0/18 | 0.002 | 344 |
| | allocator, flat | 1/18 | 0.010 | 2365 |
| 15% | uniform Lexico 5 | 5/18 | 0.030 | 670 |
| | H2O (13% of tokens exact, rest evicted) | 0/18 | 0.003 | 138 |
| | allocator, H2O scores | 0/18 | 0.012 | 2414 |
| | allocator, scores^0.3 | 6/18 | 0.103 | 2421 |
| | **allocator, flat** | **10/18** | **0.269** | 2380 |
| | allocator, flat, key error ×2 | 8/18 | 0.225 | 2373 |
| 20–22% | uniform Lexico 8 (22%) | 9/18 | 0.307 | 949 |
| | **allocator, flat (20%)** | **14/18** | **0.694** | 2374 |

With `flat`, easy tokens get 2–4 atoms and harder tokens 6–11; nothing is
evicted at 15–20%, so a "never evict" floor changes nothing. 10% memory is a
cliff for every method tried.

## 4. H2O without eviction, and the question in the prompt

**Dimming instead of evicting does not rescue H2O scores** when the question
comes after compression (15% memory, 18 questions):

| config | evicted | correct | ans p |
|---|---|---|---|
| allocator, H2O scores | 4% | 0/18 | 0.012 |
| same, never evict (≥1 atom) | 0% | 0/18 | 0.011 |
| same, ≥3 atoms | 0% | 1/18 | 0.011 |
| H2O softened (score^0.3), ≥1 atom | 0% | 6/18 | 0.102 |
| H2O barely on (score^0.1), ≥1 atom | 0% | 10/18 | 0.219 |
| no scores (flat) | 0% | 10/18 | 0.269 |

The more weight H2O gets, the worse the result. The damage comes from where
bytes go, not from eviction: the facts are low-scored, so they get 1–3 atoms
while well-attended filler gets 8–16.

**With the question in the prompt** (`--mode prompt`: the question is at the
end of the text the model reads before compression; one prefill and
compression per question), we added a SnapKV-style score (`snap`) that
counts attention from only the last 32 queries, where the question is:

| memory | config | correct | ans p | compress ms |
|---|---|---|---|---|
| 10% | uniform Lexico 3 | 1/18 | 0.010 | 344 |
| | SnapKV eviction (keep 8% exact) | 12/18 | 0.381 | 22 |
| | allocator, flat | 4/18 | 0.120 | 2399 |
| | **allocator, snap scores** | **14/18** | **0.570** | 2452 |
| 15% | uniform Lexico 5 | 7/18 | 0.106 | 615 |
| | H2O eviction (summed scores) | 1/18 | 0.016 | 40 |
| | SnapKV eviction (keep 13% exact) | 10/18 | 0.404 | 24 |
| | allocator, summed H2O scores | 4/18 | 0.088 | 2454 |
| | **allocator, snap scores** | **17/18** | **0.912** | 2443 |
| | allocator, snap scores, never evict | 15/18 | 0.809 | 2446 |
| | allocator, flat | 13/18 | 0.482 | 2423 |

What this shows:

- **With the question visible, the original DimmerKV idea works and is the
  best config we have:** question-window scores plus graded Lexico levels
  beat SnapKV-style eviction at the same bytes (17 vs 10/18 at 15%; 14 vs
  12/18 at 10%). Graded levels beat keep-or-drop.
- **Summed H2O scores are poor even here**: the question is ~20 of ~1,600
  queries, so it barely moves the sum. The score signal is what matters,
  not the allocator.
- **Without a question, flat is right; with one, question-window scores win
  by a wide margin.** That suggests one system: compress with flat when the
  context arrives, and re-allocate or refine with question-window scores
  when a question comes in.

## 5. Faster OMP and full rerun

OMP now keeps an orthonormal basis of the chosen atoms (Gram-Schmidt), so
each step is a projection instead of a linear solve, with one solve at the
end; the search over 4,096 atoms runs in FP16. Checked against the original
(`scripts/compare_omp.py`, 32 layer/streams × 8,176 held-out vectors): the
same atom in 97.7% of slots, identical stored-code error to 4 decimals at
4/8/16 atoms, **2.0× faster**. The allocator went from ~2.4 s to ~1.3 s per
1.6K-token context. The remaining time is the FP16 atom search, which is
memory-bound (~1.3 ms per step).

Full needle suite rerun with the new OMP (18 questions; correct, ans p):

| memory | config | question after compression | question in prompt |
|---|---|---|---|
| 10% | uniform Lexico 3 | 0 (0.002) | 1 (0.010) |
| | H2O / SnapKV eviction | 0 (0.002) | 12 (0.381) |
| | DimmerKV flat | 1 (0.010) | 4 (0.119) |
| | DimmerKV snap | — | **13 (0.482)** |
| 15% | uniform Lexico 5 | 5 (0.028) | 7 (0.104) |
| | H2O eviction (summed) | 0 (0.003) | 1 (0.016) |
| | SnapKV eviction | — | 10 (0.404) |
| | DimmerKV flat | **10 (0.262)** | 13 (0.521) |
| | DimmerKV snap | — | **17 (0.909)** |
| 20–22% | uniform Lexico 8 (22%) | 10 (0.289) | 12 (0.453) |
| | H2O / SnapKV eviction | 0 (0.003) | 13 (0.436) |
| | DimmerKV flat | **14 (0.684)** | 17 (0.852) |
| | DimmerKV snap | — | **17 (0.869)** |

(H2O rows in the "after" column use summed scores; SnapKV rows in the
"prompt" column use the last-32-query window. "—" = not run: snap scores
have no question to see in the "after" mode.)

Every number matches the old-OMP runs within one question. At 20% with the
question in the prompt, flat and snap both reach 17/18.

## 6. kvtc transform coding, and DimmerKV per-token rungs on top

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

## Caveats

- One small model, ~1.6K-token contexts, synthetic needles, 18 questions.
  Perplexity, LongBench and GSM8K are not run yet.
- Level 1 only: memory is counted from the codes; the GPU tensor is still
  FP16, so peak GPU memory and answer speed do not change yet.
- Compression time (~1.3 s per 1.6K tokens) is still ~2x prefill; it
  includes a 16-atom OMP on every slot.
- One unexplained detail: a few vectors reconstruct very badly under the
  per-head PCA basis (minimum cosine about −0.8) while the 5th percentile is
  fine; probably near-zero vectors. PCA is now only a baseline, so this is
  low priority.

## Next steps (proposed)

1. ~~Faster OMP~~ done (2×, section 5). Further options: cap codes at 11
   atoms (~30% less), compress off the critical path, a fused search
   kernel on the Linux GPU.
2. **Attention-aware error**: weight key error by query statistics from
   calibration, so "error" means error that changes attention. Free at
   question time; might move the 10% cliff.
3. **Refine on demand** (plan extension 3): when the question arrives, score
   the compressed tokens against it and fetch exact copies of the top few
   from CPU memory. Section 4 shows question-window scores are an excellent
   signal (17/18 at 15% when available before compression), so this is now
   the most promising extension. CPU-side bytes must be reported separately.
4. **Wider checks**: perplexity, 4K contexts, more needles, then the main runs
   on the big GPU.
5. **Demo and pitch**: update the "H2O vs DimmerKV" story and the per-token
   heat map to show atoms per token (error-driven) rather than attention
   brightness.

## Reproduce

```powershell
$py = "$env:USERPROFILE\.venvs\hashcode\Scripts\python.exe"
& $py scripts\fit_pca_basis.py --per-head        # checkpoints\pca_basis.pt
& $py scripts\train_lexico.py --steps 1500       # checkpoints\lexico_dict.pt, ~19 min
& $py scripts\bench_codecs.py                    # codec error / bytes / time
& $py scripts\test_generation.py                 # needle test, default configs
& $py scripts\test_generation.py --configs lex5 h2o:0.15 dkv:0.15 dkv:0.15:flat
& $py scripts\diagnose_scores.py                 # needle rank under H2O scores
& $py scripts	est_generation.py --mode prompt --configs h2o:0.15:snap dkv:0.15:snap
& $py scripts\compare_omp.py                     # new vs original OMP
& $py scripts\profile_allocator.py               # allocator stage timings
```

Checkpoints are gitignored; refit them on each machine, and refit after
changing the model.
