# DimmerKV — experimental findings so far

Status as of Sun 25 Oct 2026, ~1:40 AM IST. Model: Llama-3.2-1B-Instruct (FP16,
16 layers, 8 KV heads, head dim 64) on the LOQ laptop (RTX 5060 8 GB).
Branch: `codec-rework`. Raw numbers and the run-by-run log are in `CLAUDE.md`;
logs in `logs/` (not committed); grid results as CSV in `results/`.

## TL;DR

Current best system: **DimmerKV on kvtc** — kvtc-style transform coding
(cross-layer PCA with RoPE removed, grouped 0/2/4/8-bit quantization) as the
codec, with DimmerKV choosing a precision rung per token under one byte budget,
by error per byte (`flat`) or weighted by question-window attention when the
question is visible (`snap`). Sections 6–8.

1. **kvtc-style coding changes the scale.** At 3–5% of FP16 KV memory
   (20–33×), it keeps 22–24/24 needles and perplexity within 1–2% of the full
   cache on 3K-token contexts. Our earlier Lexico-based system needed ~20% for
   similar needle accuracy (section 7).
2. **DimmerKV on kvtc beats kvtc at equal bytes in every needle column.** The
   clearest gain is question-aware rungs when the question is visible
   (3% memory: answer probability .76 → .91; 22 → 24/24). Flat rungs alone add
   +.02–.07. Perplexity differences are small (≤0.07).
3. **Eviction collapses at high compression.** H2O answers 0/24 at every
   3–5% budget and 0/36 at 4–20% on short contexts when the question comes
   after compression: before the question exists, the facts rank at the
   30–35th percentile of H2O scores (section 3). SnapKV, which sees the
   question, keeps only 9–14/24 at 3–5%.
4. **Per-token allocation is the robust part of the original idea.** On Lexico
   it doubled needle accuracy at equal bytes (section 3); on kvtc it is a
   smaller but consistent gain. Summed H2O attention as the importance signal
   hurts; question-window attention helps.
5. **Real storage and time** (section 8): the compressed cache is written to
   real bytes (bit-packed, DEFLATE) and restored exactly. At 3% memory on a
   3K-token context it is 37× smaller than FP16 (100 MB → 2.7 MB) with 23/24
   needles; restoring takes ~0.16 s versus 0.9–3.8 s to recompute the prefill
   (~86 vs 342 ms at 1.6K tokens). Eviction and Lexico at 10–20× answer
   0–1/24.
6. Codec history: per-head PCA < Lexico < kvtc-style cross-layer PCA. Keys are
   the fragile stream (section 2).

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

## 7. Full comparison grid (all methods, both modes, perplexity)

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

## 8. Memory, time and accuracy (real storage)

`scripts/kvtc_storage.py` writes a kvtc / DimmerKV-TC cache to one byte
string: per stream, each compressed token's rung id (1 byte), then per rung
the bit-packed 2/4/8-bit codes and the FP16 shift and scale per (token,
group), then the exact tokens (4 sinks + 32 recent) as FP16. It is then
DEFLATE-compressed with zlib (level 6, CPU), as kvtc does with nvCOMP.
`decode_bytes` restores the per-layer cache tensors; the restored cache is
**identical** to the in-memory compressed one (max abs difference 0.0).

`scripts/bench_memory.py` reports, per method and context length: counted,
stored and DEFLATE size; compression, DEFLATE and restore time
(inflate + unpack + inverse transform + re-apply RoPE), against re-running
prefill; and needle accuracy (question after compression) and perplexity on
the same contexts. Output: `results/memory_time_accuracy.csv`.

Full run: 4 contexts × 6 needles = 24 per setting (question after
compression), 4 perplexity chunks (context + 512 scored tokens). Sizes are
relative to the FP16 KV cache; "×" is the reduction after DEFLATE (counted
size for methods without real storage). Times per context, CUDA-synced;
DEFLATE runs on the CPU.

**~1.6K-token contexts** (full cache: 24/24, ans p .98, ppl 16.53; FP16 KV
≈ 50 MB per context; prefill 342 ms):

| method | stored | after DEFLATE | × | compress ms | DEFLATE ms | restore ms | needles / ans p | ppl |
|---|---|---|---|---|---|---|---|---|
| full FP16 + DEFLATE | 1.000 | 0.909 | 1.1 | — | 1,416 | 236 | 24 / .98 | 16.53 |
| Lexico 10% | 0.082 (counted) | — | 12.2 | 225 | — | — | 0 / .00 | 19.31 |
| DimmerKV-Lexico flat 10% | 0.100 (counted) | — | 10 | 1,334 | — | — | 1 / .01 | 18.15 |
| H2O 5% | 0.050 (counted) | — | 20 | 21 | — | — | 0 / .00 | 18.36 |
| H2O 10% | 0.100 (counted) | — | 10 | 21 | — | — | 0 / .00 | 17.66 |
| SnapKV 10% | 0.100 (counted) | — | 10 | 21 | — | — | 0 / .00 | 17.86 |
| kvtc 5% | 0.043 | 0.039 | 25.6 | 120 | 62 | 78 | 23 / .78 | 16.64 |
| kvtc filled 5% | 0.050 | 0.045 | 22.0 | 119 | 73 | 80 | 24 / .84 | 16.59 |
| **DimmerKV-kvtc 5%** | 0.050 | 0.046 | 21.9 | 183 | 71 | 86 | 21 / **.87** | 16.61 |
| kvtc 10% | 0.084 | 0.075 | 13.3 | 117 | 126 | 92 | 23 / .94 | 16.55 |
| kvtc filled 10% | 0.100 | 0.090 | 11.1 | 116 | 149 | 100 | 22 / .93 | 16.57 |
| **DimmerKV-kvtc 10%** | 0.100 | 0.090 | 11.1 | 176 | 148 | 100 | 23 / **.95** | 16.58 |

**~3.1K-token contexts** (full cache: 24/24, ans p .95, ppl 11.32; FP16 KV
≈ 100 MB per context; prefill 0.9–3.8 s on the laptop, see note):

| method | stored | after DEFLATE | × | compress ms | DEFLATE ms | restore ms | needles / ans p | ppl |
|---|---|---|---|---|---|---|---|---|
| full FP16 + DEFLATE | 1.000 | 0.908 | 1.1 | — | 3,079 | 523 | 24 / .95 | 11.32 |
| Lexico 10% | 0.096 (counted) | — | 10.4 | 362 | — | — | 0 / .00 | 13.90 |
| DimmerKV-Lexico flat 10% | 0.100 (counted) | — | 10 | 2,416 | — | — | 0 / .00 | 13.44 |
| H2O 5% | 0.050 (counted) | — | 20 | 28 | — | — | 0 / .00 | 12.72 |
| H2O 10% | 0.100 (counted) | — | 10 | 30 | — | — | 0 / .00 | 12.29 |
| SnapKV 10% | 0.100 (counted) | — | 10 | 28 | — | — | 0 / .00 | 12.21 |
| kvtc 3% | 0.027 | 0.024 | **41.2** | 243 | 89 | 148 | 23 / .75 | 11.62 |
| kvtc filled 3% | 0.030 | 0.027 | 36.8 | 257 | 106 | 153 | 23 / .76 | 11.58 |
| **DimmerKV-kvtc 3%** | 0.030 | 0.027 | 36.8 | 348 | 104 | 157 | 23 / **.78** | 11.58 |
| kvtc 5% | 0.042 | 0.038 | 26.5 | 245 | 159 | 154 | 24 / .87 | 11.48 |
| kvtc filled 5% | 0.050 | 0.045 | 22.4 | 244 | 185 | 165 | 24 / .89 | 11.48 |
| **DimmerKV-kvtc 5%** | 0.050 | 0.045 | 22.3 | 360 | 181 | 171 | 24 / **.94** | **11.45** |
| kvtc 10% | 0.094 | 0.084 | 11.9 | 249 | 318 | 185 | 24 / .90 | 11.34 |
| kvtc filled 10% | 0.100 | 0.090 | 11.1 | 251 | 341 | 198 | 24 / .91 | 11.33 |
| DimmerKV-kvtc 10% | 0.100 | 0.090 | 11.1 | 353 | 347 | 197 | 24 / .92 | 11.37 |

Every restored kvtc-family cache matched the in-memory compressed cache
exactly (max abs difference 0.0).

Observations:

- **Memory**: the kvtc family stores the cache 11–41× smaller than FP16
  (e.g. 100 MB → 2.7 MB at 3% on a 3K context: 37× after DEFLATE) while
  keeping 23–24/24 needles and perplexity within 2.7% of the full cache.
  Eviction (H2O, SnapKV) and Lexico at 10–20× answer 0–1/24 needles when the
  question comes after compression, and their perplexity is 8–23% worse.
- **Time**: compression takes 0.12–0.36 s per context, restoring from bytes
  0.08–0.2 s. Restoring is ~4× faster than re-running prefill at 1.6K tokens
  (86 vs 342 ms) and 6–24× faster at 3K (157–171 ms vs 0.9–3.8 s). DimmerKV's
  per-token allocation costs ~60–110 ms more than uniform kvtc to compress and
  nothing extra to restore.
- **Accuracy at equal bytes** (kvtc filled vs DimmerKV-kvtc): DimmerKV has the
  higher answer probability in 5 of 6 settings (+.01 to +.05); needle counts
  and perplexity are within noise of each other. The gain is consistent but
  small without question scores (section 7 has the question-aware results).
- **Storage format**: stored size equals the counted size plus the per-token
  rung ids (≤0.1%), so the byte accounting used in earlier sections holds.
  DEFLATE adds ~1.1× on top of quantization (kvtc reports ~1.23× with GPU
  nvCOMP) and only 1.1× in total on raw FP16. DEFLATE runs on the CPU here;
  a GPU codec would cut its 60–350 ms.
- **Note on prefill time**: the 3K prefill measured 3.8 s in this run but
  0.9–2.4 s in earlier runs on the same laptop (eager attention near the
  8 GB limit), so the 3K speed-up range is wide; the 1.6K numbers are stable.
- Peak GPU memory is unchanged (Level 1 decodes back to FP16); the savings
  are in storage and transfer size, which is kvtc's use case.

## Caveats

- One small model (Llama-3.2-1B), 1.6K–3K-token contexts, synthetic needles
  (18–36 per setting), 4–8 perplexity chunks. LongBench and GSM8K not run.
- The kvtc family is near the ceiling on needle counts at 3–20% memory;
  answer probability carries the comparison. Differences of 1–2 needles are
  noise.
- Our kvtc is "kvtc-style", simplified: greedy bit ladder instead of the
  paper's DP, fixed groups of 64, zlib on CPU, 4 sinks + 32 recent exact
  (paper: 128). Numbers are ours, not a reproduction of the paper.
- Level 1 only: compressed caches are decoded back to FP16 before attention,
  so peak GPU memory and decode speed do not change; savings are in storage
  and transfer.
- `dkvtc:snap` in the "question after compression" mode uses the last 32
  context tokens as its window (no question); its gains there are small.

## Next steps (proposed)

1. **Bigger models on AWS**: kvtc calibration for 3B/8B on the GPU with a
   truncated basis (~10K components, as in the paper: 8B has 32,768 features
   per token), then `run_grid.py` with the kvtc family at 2–5% on 4–8K
   contexts and 50+ needles.
2. **Eviction as a rung**: a zero-byte option in the ladder so DimmerKV can
   drop clearly useless tokens at extreme budgets (kvtc lists eviction as
   future work).
3. **Refine on demand** for questions after compression: score compressed
   tokens against the question and restore exact copies of the top few from
   CPU storage. Question-window scores are a strong signal (section 7).
4. **Harder tests**: multi-hop needles (variable tracking), longer contexts,
   LongBench subset with the question in the prompt.
5. **Demo and pitch**: "kvtc compresses every token alike; DimmerKV decides
   per token". Heat map of rungs per token; storage size / restore time vs
   recompute; H2O failing after compression.

## Reproduce

```powershell
$py = "$env:USERPROFILE\.venvs\hashcode\Scripts\python.exe"
# Calibration (checkpoints are gitignored; refit per machine and per model)
& $py scripts\fit_kvtc.py                        # checkpoints\kvtc.pt, ~1.3 min
& $py scripts\train_lexico.py --steps 1500       # checkpoints\lexico_dict.pt, ~19 min
& $py scripts\fit_pca_basis.py --per-head        # checkpoints\pca_basis.pt
# Evaluation
& $py scripts\test_generation.py --configs kvtcf@0.05 dkvtc@0.05 dkvtc@0.05:snap h2o:0.05
& $py scripts\test_generation.py --mode prompt --configs kvtcf@0.03 dkvtc@0.03:snap h2o:0.03:snap
& $py scripts\eval_perplexity.py --configs kvtcf@0.05 dkvtc@0.05 h2o:0.05
& $py scripts\run_grid.py --model meta-llama/Llama-3.2-1B-Instruct --dictionary checkpoints\lexico_dict.pt `
      --budgets 0.03 0.05 --templates "h2o:{b}" "kvtcf@{b}" "dkvtc@{b}" "dkvtc@{b}:snap" --context-tokens 3000
& $py scripts\bench_memory.py                    # memory / time / accuracy table
# Diagnostics
& $py scripts\bench_codecs.py                    # PCA vs Lexico error, bytes, time
& $py scripts\diagnose_scores.py                 # needle rank under H2O scores
& $py scripts\compare_omp.py                     # new vs original OMP
& $py scripts\profile_allocator.py               # Lexico allocator stage timings
```

Method names are documented at the top of `scripts/methods.py`.
