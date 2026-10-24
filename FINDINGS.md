# DimmerKV — experimental findings so far

Status as of Sat 24 Oct 2026, ~7 PM IST. Model: Llama-3.2-1B-Instruct (FP16,
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
5. **Cost: compression time.** The allocator takes ~2.4 s per 1.6K-token
   context (prefill itself ~0.5 s). Answer speed is unchanged because Level 1
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

## Caveats

- One small model, ~1.6K-token contexts, synthetic needles, 18 questions.
  Perplexity, LongBench and GSM8K are not run yet.
- Level 1 only: memory is counted from the codes; the GPU tensor is still
  FP16, so peak GPU memory and answer speed do not change yet.
- Compression time includes a 16-atom OMP on every slot; not optimized.
- One unexplained detail: a few vectors reconstruct very badly under the
  per-head PCA basis (minimum cosine about −0.8) while the 5th percentile is
  fine; probably near-zero vectors. PCA is now only a baseline, so this is
  low priority.

## Next steps (proposed)

1. **Faster OMP**: grow the solution step by step instead of re-solving, and
   use FP16 for the atom search. Target: compression faster than prefill
   before the 4–8K main runs.
2. **Attention-aware error**: weight key error by query statistics from
   calibration, so "error" means error that changes attention. Free at
   question time; might move the 10% cliff.
3. **Refine on demand** (plan extension 3): when the question arrives, score
   the compressed tokens against it and fetch exact copies of the top few
   from CPU memory. This is where attention scores become useful again.
   CPU-side bytes must be reported separately.
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
```

Checkpoints are gitignored; refit them on each machine, and refit after
changing the model.
