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
