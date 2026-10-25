# DimmerKV — experimental findings so far

Status as of Sun 25 Oct 2026, ~6:05 AM IST. Model: Llama-3.2-1B-Instruct (FP16,
16 layers, 8 KV heads, head dim 64) on the LOQ laptop (RTX 5060 8 GB).
Branch: `main` (kept identical to `codec-rework`). Raw numbers and the run-by-run log are in `CLAUDE.md`;
logs in `logs/` (not committed); grid results as CSV in `results/`.

## Recap of all testing (Sat 24 Oct 3 PM → Sun 25 Oct 6 AM)

Runs on Llama-3.2-1B-Instruct, RTX 5060 laptop (8 GB), unless marked
3B (Llama-3.2-3B-Instruct on an AWS L40S). Main test: hidden
4-digit facts ("needles") in WikiText; the cache is compressed, then each
question is asked; plus perplexity, timing, and real storage size.
"X%" = compressed cache size as a share of the normal FP16 cache
(3% ≈ 33× smaller). Details per step are in sections 1–15 below.

| # | what we tested | result |
|---|---|---|
| 1 | Per-head PCA codec (INT8) vs Lexico (sparse dictionary) | Lexico far more accurate per byte; PCA useless below ~40% |
| 2 | Uniform codes, and PCA/Lexico mixes for keys vs values | Keys are the fragile stream; Lexico on both is best |
| 3 | DimmerKV allocator on Lexico with H2O scores | H2O scores rank the facts *below average* before the question → 0/18 |
| 4 | Allocator ignoring scores ("flat", bytes by error per byte) | Doubles uniform Lexico at equal bytes (10 vs 5/18 at 15%) |
| 5 | H2O as importance without eviction; question in the prompt | Still fails after compression; with the question visible, question-window scores (SnapKV-style) + graded precision win (17/18 at 15%) |
| 6 | Faster OMP (Lexico encoder) | 2× faster, identical results |
| 7 | kvtc-style transform coding (cross-layer PCA, RoPE removed, 0/2/4/8-bit groups) | ~25× compression with near-full accuracy; far better than everything before |
| 8 | DimmerKV on top of kvtc (per-token precision), full grid of 9 methods × budgets × 2 modes × perplexity | DimmerKV beats kvtc at equal bytes; biggest gain with question scores |
| 9 | Real storage (bit-packed + DEFLATE) and restore | Restored cache identical; 37× smaller at 3% on 3K tokens |
| 10 | Serving benchmark in MB / ms / tokens/s (3K and 8K tokens) | 23–37× more conversations per GiB of RAM, ~3–5× faster first token than recomputing, same generation speed |
| 11 | Progressive bit-plane tiers + question-time refinement (hot 2% + fetch top 1% of tokens) | Fixes most of the "question later" weak spot at 8K: ans p .50 → .71 (full .75), 17 → 19/24 |
| 12 | 3B on AWS, 8K tokens | All kvtc-family methods 36/36 at 2% (50×); H2O/SnapKV 0–1/36; test too easy to separate variants |
| 13 | Eviction as a precision level, question in the prompt, 8K | Matches the full cache at 1.5–2% (21–22/24 vs 21/24); SnapKV 6–10/24 |
| 14 | Partial-basis decode | Restore 390 → 93 ms at 2% on 8K tokens, identical cache |
| 15 | 3B serving at 16K on AWS | 57× more conversations per GiB, 18/18 correct; H2O 0/18 |
| 16 | Attention directly over the compressed cache (keys per layer, values never decoded) | Same answers 60/60; GPU memory while answering 253 MB → 18–31 MB at 2%, ~0.9× speed |
| 17 | RULER-style variable tracking (8K) and key-value retrieval (4K) | Variable tracking: DimmerKV question-aware 0.92 vs kvtc 0.82 (full 0.96); key-value: kvtc 0.65–0.70 beats DimmerKV 0.35–0.55; eviction collapses on both |
| 18 | Precision floor at kvtc's step + question scores | Ties kvtc on key-value retrieval (0.65 vs 0.70/0.65), keeps the variable-tracking lead (0.90–0.91 vs 0.81–0.82) |
| 19 | Final setting vs uncompressed cache, 8K tokens | 3%: 21/24 like the full cache, 37× smaller, 38× more conversations per 8 GiB, first token 0.44 s (recompute 1.27 s, RAM copy 0.05 s); direct mode peak GPU 7× lower |

**Headline comparison** (8K-token conversations, 24 questions):

| method | size | conversations in 8 GiB RAM | first token | needles, question later | needles, question in prompt (3%) |
|---|---|---|---|---|---|
| full cache in RAM | 252 MiB | 32 | 47 ms | 21/24 | 21/24 |
| recompute from text | ~0 | unlimited | 1,384 ms | 21/24 | 21/24 |
| H2O 2–5% | 5–13 MiB | 643–1,586 | 35 ms | 0/24 | — |
| SnapKV 3% | — | — | — | — | 13/24 |
| kvtc 5% | 11.2 MiB | 732 | 471 ms | 21/24 | 21/24 (.60) |
| **DimmerKV-kvtc 5%** | 11.2 MiB | 732 | 478 ms | **21/24** | **22/24 (.76) with question scores** |

**Bottom line**
- Eviction (H2O, SnapKV) loses facts asked about after compression; this is
  the core failure DimmerKV addresses.
- kvtc-style transform coding is the right codec (credited to Staniszewski &
  Łańcucki, ICLR 2026); Lexico and PCA remain as ablations.
- DimmerKV's per-token precision on kvtc adds a consistent gain at short
  contexts and a clear gain when the question is visible (matches the full
  cache at 3% on 8K tokens); without the question it ties kvtc at 8K.
- Savings are in storage and transfer (RAM, disk, network), and, with
  attention run directly over the codes (section 13), in GPU memory while
  answering (253 MB → 18–31 MB at 2%, same answers, ~0.9× speed).
- On RULER-style tasks DimmerKV leads kvtc on variable tracking but per-token
  precision alone lost on dense key-value lookups; a floor at kvtc's precision
  plus question scores (`dkvtc@b:snap:floor1`, section 15) ties kvtc there and
  keeps the lead. This is the recommended setting.

**Open items**: the recommended setting (`:snap:floor1`) on the needle test
and perplexity; a budget sweep with 50+ samples per cell and confidence
intervals; an ablation table (transform, per-token precision, question
scores, floor, eviction); 8B; a real cold-tier transfer in the serving
benchmark; attention over the codes for the progressive codec. See "Next
steps" below.

## TL;DR

Current best system: **DimmerKV on kvtc** — kvtc-style transform coding
(cross-layer PCA with RoPE removed, grouped 0/2/4/8-bit quantization) as the
codec, with DimmerKV choosing a precision rung per token under one byte budget,
by error per byte (`flat`) or weighted by question-window attention when the
question is visible (`snap`). Sections 6–8. Recommended setting:
`dkvtc@b:snap:floor1`: every token first gets kvtc's uniform precision, then
the leftover bytes go by question score (section 15), which is never worse
than kvtc on the tasks tested. Options built on top: an eviction level
(section 12), progressive tiers with question-time refinement (section 11),
and attention directly over the codes (section 13).

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
6. **Serving** (section 9): in a chat-assistant scenario, 8 GiB of RAM holds
   3,127 compressed 3K-token conversations vs 85 uncompressed (37×); first
   answer token in ~0.18 s vs 0.89 s to recompute; generation speed unchanged.
7. **Eviction as a precision level** (section 12): with the question in the
   prompt, DimmerKV with question scores and a zero-byte level matches the
   full cache at 1.5–2% of its memory (8K tokens: 21–22/24 vs 21/24;
   SnapKV 6–10/24).
8. **Progressive tiers + refinement** (section 11): keep 2% hot, fetch the
   question's top 1% of tokens at full precision (0.5% extra bytes) → the
   "question after compression" case at 8K goes from ans p .50 to .71
   (full .75). On 3B, every kvtc-family method keeps 36/36 needles at 2%
   (50×) on 8K tokens; H2O 0/36.
9. **Long conversations** (section 10, 8K tokens): question visible →
   DimmerKV with question scores at 3% matches the full cache (22/24 vs
   21/24) where kvtc gets 21/24 at lower confidence and SnapKV 13/24; question
   not visible → DimmerKV ties kvtc, 23× more conversations per GiB at 5%
   with full-cache accuracy; H2O 0/24.
10. Codec history: per-head PCA < Lexico < kvtc-style cross-layer PCA. Keys are
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

## 9. Serving benchmark in absolute units

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

## 10. Long conversations (8K tokens)

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
  after the accuracy pass; a rerun with a pageable-RAM fallback was stopped
  before finishing, so there are no 16K results.

## 11. Progressive tiers with question-time refinement, and 3B on AWS

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

## 12. Eviction as a precision level, faster restore, 3B at 16K

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

## 13. Answering directly from the compressed cache (no full decode)

Until now compression saved storage and transfer only: before answering, the
cache was rebuilt to full FP16 on the GPU. `scripts/compressed_attention.py`
attends over the compressed tokens directly:

- **Keys, decoded one layer at a time:** when attention reaches layer *l*,
  only that layer's keys are rebuilt (coefficients × that layer's slice of
  the cross-layer basis + mean, then RoPE at their positions), used, and
  discarded.
- **Values, never decoded:** the output Σₙ pₙ vₙ equals
  (Σₙ pₙ cₙ) · B_v + mean_v · Σₙ pₙ, so attention weights the small
  coefficient vectors and applies the basis once per head.
- Coefficients are kept in FP16, grouped by precision step with one shared
  basis of the components actually used; tiny groups are merged. Evicted
  keys are left out. Sinks, the recent window and new question/answer tokens
  sit in a small ordinary cache; both parts share one softmax.

Test (`scripts/test_compressed_attention.py`, 1B, ~8K tokens, 12 questions
per method, question after compression; `logs/compressed_attention_8k_*.log`):

| method | same answer as rebuilt cache | GPU memory held per conversation | extra working memory while answering | tokens/s (rebuilt → direct) |
|---|---|---|---|---|
| kvtc 2% | 12/12 (max logit diff 0.047) | 253 MB → **18.3 MB** | 506 → 63 MB | 45.3 → 40.5 |
| DimmerKV 2% | 12/12 (0.039) | 253 MB → **22.1 MB** | 506 → 62 MB | 44.5 → 38.6 |
| DimmerKV + question scores + eviction 2% | 12/12 (0.047) | 253 MB → **31.4 MB** | 506 → 64 MB | 30.8 → 31.2 |
| kvtc 5% | 12/12 (0.055) | 253 MB → 54.6 MB | 507 → 62 MB | 42.5 → 29.2 |
| DimmerKV 5% | 12/12 (0.063) | 253 MB → 60.0 MB | 506 → 66 MB | 39.6 → 23.1 |

- **Same answers on all 60 questions**; logits differ only by FP16 rounding.
- **GPU memory while answering drops 8–14× at 2%** (held) and ~8× in
  working memory; the saving is no longer only in storage.
- **Speed**: about 0.87–1.0× of the rebuilt cache at 2%, 0.58–0.69× at 5%
  (more components to rebuild per layer per token).
- Not counted per conversation: the kept basis rows (24–200 MB here), a
  per-model constant like the weights. Prefill still builds the full cache
  once before compressing (streaming compression during prefill is a next
  step). Coefficients are held as FP16, ~3× the packed storage size.
- Two earlier variants were measured and replaced: one union of components
  for all tokens (faster at 2%, 25–32 MB held) and per-step blocks with
  their own basis copies (18 MB held but up to 721 MB of duplicated basis
  and 20 tok/s with many steps).

## 14. RULER-style long-context tasks (variable tracking, key-value retrieval)

`scripts/eval_ruler.py`: the two long-context tasks the kvtc paper reports,
generated locally. Llama-3.2-1B, 20 samples per cell, both question modes,
greedy decoding (raw logs in `results/ruler_*.txt`).

- **Variable tracking** (multi-hop): a 5-variable chain `VAR A = 12345`,
  `VAR B = VAR A`, … in repeated filler, ~8K tokens; the model must list every
  variable holding the value. Score = share of the chain's variables named.
- **Key-value retrieval** ("lost in the middle"): a JSON object of random UUID
  pairs, ~4K tokens; the model must return one key's value. Score = exact match.

Variable tracking, 8K tokens (full cache 0.96):

| method | after, 2% | after, 3% | prompt, 2% | prompt, 3% |
|---|---|---|---|---|
| H2O | 0.02 | 0.11 | 0.04 | 0.15 |
| SnapKV | 0.00 | 0.01 | 0.24 | 0.34 |
| kvtc | 0.82 | 0.81 | 0.79 | 0.82 |
| DimmerKV (flat) | 0.80 | 0.84 | 0.77 | 0.78 |
| DimmerKV + question scores + eviction | 0.84 | **0.89** | **0.86** | **0.92** |
| DimmerKV progressive + refine top 1% | **0.87** | 0.88 | 0.84 | 0.87 |

Key-value retrieval, 4K tokens (full cache 0.65):

| method | after, 3% | after, 5% | prompt, 3% | prompt, 5% |
|---|---|---|---|---|
| H2O | 0.00 | 0.00 | 0.00 | 0.00 |
| SnapKV | 0.00 | 0.00 | 0.00 | 0.05 |
| kvtc | 0.05 | **0.70** | 0.05 | **0.65** |
| DimmerKV (flat) | 0.10 | 0.45 | **0.20** | 0.50 |
| DimmerKV + question scores + eviction | **0.20** | 0.40 | 0.15 | 0.35 |
| DimmerKV progressive + refine top 1% | 0.00 | 0.55 | 0.10 | 0.55 |

- **Eviction collapses on both tasks** (H2O ≤ 0.15, SnapKV ≤ 0.34 even with the
  question visible).
- **Variable tracking: DimmerKV's question-aware variants beat kvtc** by
  +0.05–0.10 (0.92 vs 0.82 at 3% with the question in the prompt; full 0.96).
  Flat per-token precision alone is tied with kvtc.
- **Key-value retrieval: plain kvtc is best at 5%** (0.65–0.70, matching the full
  cache); DimmerKV variants get 0.35–0.55. Every UUID is random and equally
  important, so uniform precision is the right allocation; moving bytes between
  tokens starves some UUIDs, and one wrong character fails an exact match. At
  3% every method collapses (≤ 0.20).
- **Takeaway:** per-token precision helps when importance is uneven (facts in
  filler, assignment chains) and can hurt on dense lookup tables where every
  token matters equally. A minimum precision per token is the obvious fix to
  test next. 20 samples per cell: differences below ~0.1 are within noise.

## 15. Fixing the dense-lookup loss: a precision floor at kvtc's level

Section 14 found DimmerKV losing to kvtc on key-value retrieval. Tried, all
on the same 20 samples (raw logs `results/ruler_*floor*.txt`, `*_kw.txt`):

| variant (key-value 4K, 5%) | question after | question in prompt |
|---|---|---|
| full cache | 0.65 | 0.65 |
| kvtc (uniform) | 0.70 | 0.65 |
| DimmerKV, floor at half the budget (`:floor`) | 0.45 | 0.50 |
| DimmerKV, floor at 80% (`:floor0.8`) | 0.40 | — |
| DimmerKV, key errors ×4 / ×16 (`:kw4`, `:kw16`) | 0.45 / 0.20 | — |
| DimmerKV flat, floor at kvtc's own rung (`:floor1`) | 0.40 | 0.60 |
| **DimmerKV + question scores + floor at kvtc's rung (`:snap:floor1`)** | **0.65** | **0.65** |

Variable tracking 8K, 3% with the same floor:

| variant | question after | question in prompt |
|---|---|---|
| full cache | 0.96 | 0.96 |
| kvtc | 0.81 | 0.82 |
| DimmerKV + question scores + floor (`:snap:floor1`) | **0.90** | **0.91** |

- Partial floors put every token one precision step below kvtc's uniform
  step; UUID lookups need every token at least at that step, so they did not
  help. Weighting key errors did not help either.
- **A floor at kvtc's own step plus question-scored top-ups is never worse
  than kvtc on both tasks here**: it ties kvtc on key-value retrieval and
  keeps the +0.09 lead on variable tracking. This is the recommended safe
  setting (`dkvtc@b:snap:floor1`).
- The flat variant with the same floor still scores 0.40 on key-value
  retrieval (question after); with kvtc above the full cache (0.70 vs 0.65),
  this task is noisy for the 1B model, so treat ±0.2 there with caution.
- `scripts/demo_backup.py` writes `docs/demo_backup.html`: the demo's three
  scenarios and leaderboard recorded from the live model, for use if the
  live demo cannot run.

## 16. Final setting vs the uncompressed cache: time and memory (8K tokens)

Final setting `dkvtc@b:snap:floor1`, Llama-3.2-1B, RTX 5060 laptop, ~8K-token
conversations, 24 questions asked after compression (12 for the direct-mode
test). Raw rows: `results/serving_8k_final.csv`, `results/grid_1b_ctx8000_final.csv`,
`logs/compressed_attention_8k_final.log`.

| | full cache kept on GPU | full cache in RAM | recompute from text | **final 2%** | **final 3%** |
|---|---|---|---|---|---|
| stored per conversation | 252 MiB (GPU) | 252 MiB | ~0 | **4.5 MiB (56×)** | **6.8 MiB (37×)** |
| conversations in 8 GiB RAM | — | 32 | unlimited | **1,806** | **1,209** |
| compress (once) | — | — | — | 1.5 s | 2.5 s |
| restore | 2 ms | 15 ms | 1,240 ms | 268 ms | 387 ms |
| time to first token | 42 ms | 48 ms | 1,267 ms | 310 ms | 437 ms |
| generation speed | 39–47 tok/s | 47 tok/s | 45 tok/s | 47 tok/s | 47 tok/s |
| peak extra GPU memory while answering (rebuilt cache) | 613 MiB | 613 MiB | 742 MiB | 1,607 MiB | 1,647 MiB |
| correct / ans p | 21/24 / .76 | 21/24 / .76 | 21/24 / .76 | 16/24 / .43 | **21/24 / .62** |

Answering directly over the codes (12 questions, same replies as the rebuilt
cache 12/12 at both budgets):

| | rebuilt cache | direct, final 2% | direct, final 3% |
|---|---|---|---|
| GPU memory held for the conversation | 253 MiB | 181 MiB | **41 MiB (6×)** |
| peak extra GPU memory | 506 MiB | **72 MiB (7×)** | **70 MiB (7×)** |
| generation speed | 46 tok/s | 22 tok/s | 29 tok/s |

Needle test, same contexts, question after compression (correct / ans p;
full 21/24 / .756): kvtc 17/24 / .500 at 2% and 19/24 / .581 at 3%; final
setting 16/24 / .432 and **21/24 / .623**; H2O and SnapKV 0/24 at both;
progressive + refine top 1% 19/24 / .673 and 20/24 / .688.

- **At 3% the final setting keeps the full cache's answer count (21/24) in
  37× less storage**, 38× more conversations per 8 GiB, with first token in
  0.44 s, 2.9× faster than re-reading the conversation (1.27 s) but slower
  than copying an uncompressed cache from RAM (48 ms). Confidence is lower
  than the full cache (.62 vs .76). Generation speed is unchanged.
- **At 2% the floor cannot act** (kvtc only affords its coarsest step there),
  so the setting equals plain question scores and is below kvtc (16 vs 17/24).
  Without the question, refinement is the better 2% option (19/24).
- **The rebuild path costs GPU memory**: decoding needs ~1.6 GiB of temporary
  space vs 0.6 GiB for the full cache. Answering directly over the codes fixes
  this (peak 70 MiB, 7× less) at about half the generation speed here.
- **Direct mode at 2% held 181 MiB**, far above the 18–31 MB measured for
  plain DimmerKV in section 13. Likely cause (not yet checked): the floor +
  question-score mix spreads tokens over many precision steps, and merging
  small blocks stores them at the widest member's width. To check, then fix
  by merging blocks by bytes.
- Laptop timings at 8K are under GPU memory pressure: the scoring prefill
  (eager attention) took 7.4 s per conversation (SDPA recompute: 1.2 s), and
  compression 1.5–2.5 s vs ~0.35 s at 3K tokens.

## Caveats

- Mostly one small model (Llama-3.2-1B; 3B on AWS), up to 16K-token contexts,
  synthetic needles and RULER-style tasks (18–36 questions or 20 samples per
  setting), 4–8 perplexity chunks. LongBench and GSM8K not run.
- Per-token precision can lose to uniform kvtc on dense lookup tasks where every
  token matters equally (key-value retrieval, section 14); a floor at kvtc's
  step with question scores closes the gap (section 15), the flat variant does not.
- The kvtc family is near the ceiling on needle counts at 3–20% memory;
  answer probability carries the comparison. Differences of 1–2 needles are
  noise.
- Our kvtc is "kvtc-style", simplified: greedy bit ladder instead of the
  paper's DP, fixed groups of 64, zlib on CPU, 4 sinks + 32 recent exact
  (paper: 128). Numbers are ours, not a reproduction of the paper.
- Answering directly from the compressed cache (section 13) is implemented for
  the kvtc family; the progressive codec and Lexico still rebuild an FP16
  cache. Prefill still builds the full cache once before compressing.
- `dkvtc:snap` in the "question after compression" mode uses the last 32
  context tokens as its window (no question); its gains there are small.
- The recommended setting (`:snap:floor1`) has so far been run only on the
  RULER-style tasks (20 samples each); the needle and perplexity numbers in
  sections 7–12 come from the earlier settings.
- Question-time refinement keeps the cold tier in the same GPU tensors as the
  hot tier: fetched bytes are counted and the decode is timed, but a real copy
  from host RAM or disk is not yet measured. Scoring also runs the question
  through the model once more before answering.

## Next steps (proposed)

Done from the earlier list: eviction as a zero-byte level (section 12),
refinement on demand as progressive tiers (section 11), multi-hop variable
tracking (section 14), 3B on AWS (sections 11–12), the demo.

1. **Make the recommended setting's case robust**: `:snap:floor1` vs kvtc at
   budgets of 2–10%, 50+ samples per cell with confidence intervals, on the
   needle test, both RULER-style tasks and perplexity.
2. **Ablation table**: add one part at a time (transform, per-token
   precision, question scores, floor, eviction) on the same tests.
3. **8B on AWS** (Llama-3.1-8B, 32,768 features per token, ~14% more than
   3B; calibration ~8–10 min expected), then 3B/8B with the floor setting.
4. **Real cold tier**: keep cold planes in host RAM, time the copy of the
   refined tokens in the serving benchmark.
5. **Attention over the codes for the progressive codec**, and compression
   streamed during prefill so the full cache never exists.
6. **A real-document benchmark** (two LongBench tasks) if time allows.

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
& $py scripts\fit_progressive.py                 # checkpoints\progressive.pt (after fit_kvtc.py)
& $py scripts\test_generation.py --context-tokens 8000 --configs kvtcf@0.02 prog@0.02 prog@0.02:r0.01
& $py scripts\eval_ruler.py --task kv --context-tokens 4000 --configs kvtcf@0.05 dkvtc@0.05:snap:floor1
# Diagnostics
& $py scripts\bench_codecs.py                    # PCA vs Lexico error, bytes, time
& $py scripts\diagnose_scores.py                 # needle rank under H2O scores
& $py scripts\compare_omp.py                     # new vs original OMP
& $py scripts\profile_allocator.py               # Lexico allocator stage timings
```

Method names are documented at the top of `scripts/methods.py`.
