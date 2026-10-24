"""DimmerKV demo (Gradio): compress a document's KV cache, then ask about it.

Tab 1 compares two methods side by side on the same document, question and
memory budget: the answer, whether it is right, stored size, conversations
per 8 GiB of RAM, compression and restore time. Tab 2 shades every token by
the bytes each method kept for it (dark = dropped, bright = exact). Tab 3
shows the measured results.

Run from the repo root: python scripts/demo_app.py  (opens on 127.0.0.1:7860)
"""

from __future__ import annotations

import html
import math
import time
import zlib

import gradio as gr
import torch

import dimmer
import kvtc_storage
from bench_codecs import synchronize
from common import DEFAULT_MODEL, build_cache, kv_config, load_model, wikitext_token_chunks
from compressed_attention import CompressedContext, greedy_answer
from methods import Methods
from test_generation import build_prompt, split_ids, split_ids_in_prompt

CONTEXT_TOKENS = 3000
RAM_GIB = 8
METHODS = {
    "Full cache (no compression)": "full",
    "H2O (evict by attention)": "h2o:{b}",
    "SnapKV (evict, sees the question)": "h2o:{b}:snap",
    "kvtc (same precision for every token)": "kvtcf@{b}",
    "DimmerKV (per-token precision)": "dkvtc@{b}",
    "DimmerKV + question scores + eviction": "dkvtc@{b}:snap:ev",
    "DimmerKV progressive + refine top 1% at question time": "prog@{b}:r0.01",
}

tokenizer, model, device = load_model(DEFAULT_MODEL)
dimmer.install(model)
config = kv_config(model)
methods = Methods(DEFAULT_MODEL, "checkpoints/pca_basis.pt", "checkpoints/lexico_dict.pt", device, 32,
                  model=model, kvtc="checkpoints/kvtc.pt", progressive="checkpoints/progressive.pt")
_fillers = wikitext_token_chunks(tokenizer, "test", CONTEXT_TOKENS, 10)
_prefill_cache: dict = {}


def sample_document(index: int):
    context, qa = build_prompt(tokenizer.decode(_fillers[index][1:]), index)
    return context, qa


def timed(fn):
    synchronize(device)
    started = time.perf_counter()
    result = fn()
    synchronize(device)
    return result, (time.perf_counter() - started) * 1000


def prefill(context: str, question: str, in_prompt: bool):
    """Context ids, question-tail ids, cache tensors, scores, prefill ms.

    The prefilled part is memoized: it is the document alone when the question
    comes after compression (shared by every question), or document + question
    when the question is in the prompt. The question tail is always rebuilt
    for the current question.
    """
    splitter = split_ids_in_prompt if in_prompt else split_ids
    head_ids, tail_ids = splitter(tokenizer, context, question)
    key = (context, in_prompt, question if in_prompt else None)
    if key not in _prefill_cache:
        with torch.inference_mode():
            (cache, scores), ms = timed(lambda: dimmer.prefill_with_scores(model, head_ids[None].to(device)))
        keys = [layer.keys.clone() for layer in cache.layers]
        values = [layer.values.clone() for layer in cache.layers]
        _prefill_cache.clear()
        _prefill_cache[key] = (head_ids, keys, values, scores, ms)
    cached_ids, keys, values, scores, ms = _prefill_cache[key]
    if not torch.equal(cached_ids, head_ids):
        raise RuntimeError("Prefilled tokens do not match the current prompt.")
    return head_ids, tail_ids, keys, values, scores, ms


def answer_with(keys, values, evicted, context_ids, tail_ids) -> str:
    input_ids = torch.cat([context_ids, tail_ids])[None].to(device)
    cache = build_cache([k.clone() for k in keys], [v.clone() for v in values], model)
    dimmer.set_eviction(evicted)
    try:
        generated = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                                   past_key_values=cache, max_new_tokens=16, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
    finally:
        dimmer.set_eviction(None)
    return tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True).strip()


def run_method(label: str, budget: float, head_ids, tail_ids, keys, values, scores, direct: bool = False):
    name = METHODS[label].format(b=f"{budget:g}")
    tokens = head_ids.shape[0]
    full_bytes = tokens * config.fp16_bytes_per_token()
    with torch.inference_mode():
        compressed = methods.apply(name, keys, values, scores)
        restore_ms, fetched = None, 0
        if name.startswith(("kvtcf@", "dkvtc@")) and compressed.evicted is None:
            # Real storage round trip: bytes -> DEFLATE -> restore.
            codec = methods.codec("kvtc")
            data = zlib.compress(kvtc_storage.encode_bytes(codec, keys, values, codec.last_choice, 32), 6)
            _, restore_ms = timed(lambda: kvtc_storage.decode_bytes(codec, zlib.decompress(data)))
            stored = len(data)
        else:
            stored = compressed.used_bytes
        q_keys, q_values = compressed.keys, compressed.values
        refine_ms = None
        if compressed.refine is not None:
            (q_keys, q_values, fetched), refine_ms = timed(lambda: compressed.refine(tail_ids, tokens))
        gpu_held, path = full_bytes, "rebuilt FP16 cache"
        if direct and name.startswith(("kvtcf@", "kvtc@", "dkvtc@")):
            # Answer over the codes: keys rebuilt per layer, values never decoded.
            codec = methods.codec("kvtc")
            context = CompressedContext(codec, keys, values, model)
            exact = build_cache([l.keys.clone() for l in context.exact_cache.layers],
                                [l.values.clone() for l in context.exact_cache.layers], model)
            (reply, _), answer_ms = timed(lambda: greedy_answer(model, tokenizer, context, exact,
                                                                tail_ids[None].to(device), tokens, 16))
            gpu_held, path = context.code_bytes() + context.exact_bytes(), "directly over the codes"
        else:
            reply, answer_ms = timed(lambda: answer_with(q_keys, q_values, compressed.evicted, head_ids, tail_ids))
            if name.startswith("h2o:"):
                # A real eviction cache holds only the kept tokens (here they are masked in a full cache).
                gpu_held, path = compressed.used_bytes, "kept tokens only (evicted tokens are gone)"
            elif direct and name != "full":
                path = "rebuilt FP16 cache (direct mode covers kvtc / DimmerKV-kvtc only)"
    return {
        "name": name, "reply": reply, "stored": stored, "full": full_bytes, "fetched": fetched,
        "compress_ms": compressed.seconds * 1000, "restore_ms": restore_ms, "refine_ms": refine_ms,
        "token_bytes": compressed.token_bytes, "gpu_held": gpu_held, "path": path, "answer_ms": answer_ms,
    }


def card(label: str, result: dict, expected: str | None) -> str:
    verdict = ""
    if expected:
        verdict = "✅ correct" if expected in result["reply"] else f"❌ wrong (expected {expected})"
    stored_mb = result["stored"] / 2**20
    ratio = result["full"] / max(result["stored"], 1)
    per_ram = int(RAM_GIB * 2**30 // max(result["stored"], 1))
    lines = [
        f"### {label}",
        f"**Answer:** `{result['reply'] or '(empty)'}` {verdict}",
        "",
        "| | |",
        "|---|---|",
        f"| Stored size | **{stored_mb:.2f} MB** of {result['full'] / 2**20:.1f} MB (**{ratio:.0f}× smaller**) |",
        f"| Conversations in {RAM_GIB} GiB of RAM | {per_ram:,} |",
        f"| Compress time | {result['compress_ms']:.0f} ms |",
    ]
    if result["restore_ms"] is not None:
        lines.append(f"| Restore from bytes | {result['restore_ms']:.0f} ms |")
    if result["refine_ms"] is not None:
        lines.append(f"| Question-time refinement | fetched {result['fetched'] / 2**20:.2f} MB "
                     f"in {result['refine_ms']:.0f} ms |")
    held = result["gpu_held"] / 2**20
    note = "" if result["gpu_held"] >= result["full"] else f" (**{result['full'] / result['gpu_held']:.0f}× less**)"
    lines.append(f"| GPU memory for this conversation while answering | **{held:.1f} MB**{note} · "
                 f"{result['path']} |")
    lines.append(f"| Answer time | {result['answer_ms']:.0f} ms |")
    return "\n".join(lines)


def heat_map(head_ids, token_bytes: torch.Tensor | None, highlight: str | None) -> str:
    """Tokens shaded by stored bytes on a log scale (dark = dropped, bright = exact)."""
    if token_bytes is None:
        return "<p>No per-token data for this method.</p>"
    full = 2.0 * config.layers * config.kv_heads * config.head_dim * 2
    values = token_bytes.float().cpu().tolist()
    pieces = [tokenizer.decode([int(t)]) for t in head_ids]
    # Mark the tokens of the expected answer.
    marked = set()
    if highlight:
        text, spans = "", []
        for i, piece in enumerate(pieces):
            spans.append((len(text), len(text) + len(piece)))
            text += piece
        start = text.find(highlight)
        if start >= 0:
            marked = {i for i, (a, b) in enumerate(spans) if a < start + len(highlight) and b > start}
    out = []
    for i, (piece, size) in enumerate(zip(pieces, values)):
        if size <= 0:
            style = "background:#1f2328;color:#8b949e;"
        else:
            level = max(0.0, min(1.0, (math.log10(size / full) + 2) / 2))  # 1% -> 0, 100% -> 1
            light = 18 + 62 * level
            style = f"background:hsl(45,95%,{light:.0f}%);color:{'#111' if light > 50 else '#f0f0f0'};"
        if i in marked:
            style += "outline:2px solid #e5484d;font-weight:700;"
        out.append(f'<span title="{size:.0f} B" style="{style}">{html.escape(piece)}</span>')
    legend = ('<p style="font-size:0.85em">Shading = bytes kept per token (log scale): '
              '<span style="background:#1f2328;color:#8b949e;padding:0 4px">dropped</span> '
              '<span style="background:hsl(45,95%,18%);color:#f0f0f0;padding:0 4px">~1% of FP16</span> '
              '<span style="background:hsl(45,95%,49%);color:#111;padding:0 4px">~10%</span> '
              '<span style="background:hsl(45,95%,80%);color:#111;padding:0 4px">exact</span> '
              '· <span style="outline:2px solid #e5484d;padding:0 4px">answer</span></p>')
    return legend + ('<div style="max-height:420px;overflow-y:auto;line-height:1.7;font-family:ui-monospace,'
                     'monospace;font-size:12px;white-space:pre-wrap;word-break:break-word">'
                     + "".join(out) + "</div>")


def token_summary(token_bytes: torch.Tensor | None) -> str:
    """One line: how the method treated the document's tokens."""
    if token_bytes is None:
        return ""
    full = 2.0 * config.layers * config.kv_heads * config.head_dim * 2
    share = (token_bytes.float() / full).cpu()
    groups = {
        "dropped": share <= 0,
        "coarse (<3% of FP16)": (share > 0) & (share < 0.03),
        "medium (3–20%)": (share >= 0.03) & (share < 0.2),
        "fine (20–99%)": (share >= 0.2) & (share < 0.99),
        "exact": share >= 0.99,
    }
    parts = [f"{name} {float(mask.float().mean()):.0%}" for name, mask in groups.items() if bool(mask.any())]
    return "**Tokens:** " + " · ".join(parts)


def resolve_question(sample: int, question_choice: str, custom_question: str):
    context, qa = sample_document(int(sample))
    if custom_question.strip():
        return context, custom_question.strip(), None
    questions = [q for q, _ in qa]
    index = questions.index(question_choice) if question_choice in questions else 0
    return context, qa[index][0], qa[index][1]


def document_header(head_ids, prefill_ms: float, in_prompt: bool) -> str:
    when = "known before compression" if in_prompt else "asked after compression"
    return (f"**Document:** {head_ids.shape[0]:,} tokens · full KV cache "
            f"{head_ids.shape[0] * config.fp16_bytes_per_token() / 2**20:.1f} MB · re-reading it (prefill) "
            f"took {prefill_ms:.0f} ms · question {when}")


def run(sample: int, question_choice: str, custom_question: str, in_prompt: bool, budget_percent: float,
        left: str, right: str, direct: bool = False):
    context, question, expected = resolve_question(sample, question_choice, custom_question)
    head_ids, tail_ids, keys, values, scores, prefill_ms = prefill(context, question, in_prompt)
    budget = budget_percent / 100
    outputs = []
    for label in (left, right):
        try:
            result = run_method(label, budget, head_ids, tail_ids, keys, values, scores, direct)
            outputs.append((card(label, result, expected) + "\n\n" + token_summary(result["token_bytes"]),
                            heat_map(head_ids, result["token_bytes"], expected)))
        except (ValueError, AssertionError) as error:
            outputs.append((f"### {label}\n\n⚠️ {error}\n\n(kvtc needs ≥ ~2.8% at 3K tokens; "
                            "use DimmerKV + eviction or a larger budget.)", ""))
    return (document_header(head_ids, prefill_ms, in_prompt),
            outputs[0][0], outputs[1][0], outputs[0][1], outputs[1][1])


def run_all(sample: int, question_choice: str, custom_question: str, in_prompt: bool, budget_percent: float,
            direct: bool = False):
    """Every method on the same question and budget: a leaderboard and a size chart."""
    import pandas as pd

    context, question, expected = resolve_question(sample, question_choice, custom_question)
    head_ids, tail_ids, keys, values, scores, prefill_ms = prefill(context, question, in_prompt)
    budget = budget_percent / 100
    per_ram = f"Conversations / {RAM_GIB} GiB"
    rows = []
    for label in METHODS:
        short = label.split(" (")[0]
        try:
            r = run_method(label, budget, head_ids, tail_ids, keys, values, scores, direct)
        except (ValueError, AssertionError):
            rows.append({"Method": short, "Answer": "—", "Correct": "budget too small", "Stored MB": None,
                         "× smaller": None, per_ram: None, "GPU MB while answering": None, "Compress ms": None,
                         "Restore / refine ms": None})
            continue
        correct = ("✅" if expected in r["reply"] else "❌") if expected else "?"
        extra = r["restore_ms"] if r["restore_ms"] is not None else r["refine_ms"]
        rows.append({
            "Method": short, "Answer": r["reply"][:24], "Correct": correct,
            "Stored MB": round(r["stored"] / 2**20, 2), "× smaller": round(r["full"] / max(r["stored"], 1), 1),
            per_ram: int(RAM_GIB * 2**30 // max(r["stored"], 1)),
            "GPU MB while answering": round(r["gpu_held"] / 2**20, 1),
            "Compress ms": round(r["compress_ms"]), "Restore / refine ms": None if extra is None else round(extra),
        })
    table = pd.DataFrame(rows)
    chart = table.dropna(subset=["× smaller"])[["Method", "× smaller", "Correct"]]
    return document_header(head_ids, prefill_ms, in_prompt), table, chart


def update_questions(sample: int):
    _, qa = sample_document(int(sample))
    choices = [q for q, _ in qa]
    return gr.update(choices=choices, value=choices[0])


SCENARIOS = {
    "1 · Question after compression (4%)": dict(in_prompt=False, budget=4, left="H2O (evict by attention)",
                                                right="DimmerKV (per-token precision)"),
    "2 · Question known in advance (2%)": dict(in_prompt=True, budget=2, left="SnapKV (evict, sees the question)",
                                               right="DimmerKV + question scores + eviction"),
    "3 · Hard case + refinement (3%)": dict(in_prompt=False, budget=3, left="kvtc (same precision for every token)",
                                            right="DimmerKV progressive + refine top 1% at question time"),
}


def apply_scenario(name: str):
    sc = SCENARIOS[name]
    return (gr.update(value=sc["in_prompt"]), gr.update(value=sc["budget"]), gr.update(value=sc["left"]),
            gr.update(value=sc["right"]))


def architecture_svg() -> str:
    """The architecture diagram from docs/DimmerKV_overview.html (one source of truth)."""
    import re
    from pathlib import Path

    page = Path(__file__).resolve().parent.parent / "docs" / "DimmerKV_overview.html"
    try:
        match = re.search(r'<svg id="dimmerkv-architecture".*?</svg>', page.read_text(encoding="utf-8"), re.S)
    except OSError:
        match = None
    if not match:
        return "<p>Architecture diagram: see docs/DimmerKV_overview.html.</p>"
    return f'<div style="background:#fbfaf7;border-radius:12px;padding:8px;overflow-x:auto">{match.group(0)}</div>'


HOW_IT_WORKS = """
### The pipeline in five steps
1. **Calibrate once per model**: undo RoPE on keys, join all layers and heads of a token into one
   vector, fit one PCA basis, and order (group, bit-plane) pieces by error removed per byte (kvtc-style).
2. **Compress once per conversation**: every token is quantized once at 8 bits; DimmerKV decides how
   many bit-planes each token keeps (dropped, coarse … fine), by error per byte, weighted by the
   question's attention when the question is known. The first 4 and last 32 tokens stay exact.
3. **Store**: bit-pack + DEFLATE; a hot tier with each token's first planes, a cold tier with the rest.
4. **Restore per question**: decode only the PCA groups that carry bits and re-apply RoPE; optionally
   score the restored cache with the question and fetch the full planes for its top ~1% of tokens.
5. **Answer without rebuilding the cache**: keys are rebuilt one layer at a time and discarded, values are
   never decoded (attention weights the coefficients, then applies the basis once per head); answers are
   identical to the fully decoded cache, with 8–14× less GPU memory at 2%.

**Why this pipeline:** eviction (H2O) deletes facts that a later question needs; per-head PCA and Lexico
were too lossy or too slow; kvtc's cross-layer transform keeps nearly everything at 2–5%; DimmerKV's
per-token precision, question scores and question-time refinement spend the remaining bytes where
they matter. Full write-up: `docs/DimmerKV_overview.html`.
"""

RESULTS = """
### Measured results (Llama-3.2-1B on the laptop, Llama-3.2-3B on an AWS L40S)

**Question known before compression, 8K-token documents**

| method | 1B, 1.5% | 1B, 2% | 3B, 2% |
|---|---|---|---|
| full cache | 21/24 | 21/24 | 36/36 |
| SnapKV | 6/24 | 10/24 | 34/36 |
| kvtc | — | 17/24 | 35/36 |
| **DimmerKV + question scores (+ eviction on 1B)** | **21/24** | **22/24** | **36/36** |

**Question asked after compression, 8K tokens** (full cache: 1B 21/24, ans p .75)

| method | 1B, 2% | 3B, 2% |
|---|---|---|
| H2O | 0/24 | 0/36 |
| kvtc | 17/24 (.51) | 36/36 |
| **DimmerKV progressive, refine top 1% (+0.5% fetched)** | **19/24 (.71)** | **36/36** |

**Serving, 3B model, 16K-token conversations** (18 questions)

| way of keeping a conversation | stored | conversations in 8 GiB | first answer token | correct |
|---|---|---|---|---|
| full cache in RAM | 1,753 MiB | 4 | 189 ms | 18/18 |
| re-read the text | ~0 | unlimited | 1,434 ms | 18/18 |
| H2O 2% | 35 MiB | 230 | 61 ms | 0/18 |
| **kvtc / DimmerKV 2%** | **31 MiB** | **263 (57×)** | **488–627 ms** | **18/18** |

**Perplexity, 3B, 8K tokens** (full 8.33): H2O 10.64 · SnapKV 10.25 · kvtc 8.48 · DimmerKV + question
scores 8.46 (2% budget).

Savings are in storage and transfer; during answering the model uses a normal FP16 cache.
Details: `FINDINGS.md`, `docs/DimmerKV_overview.html`.
"""

with gr.Blocks(title="DimmerKV") as demo:
    gr.Markdown("# DimmerKV: dim the KV cache instead of deleting it\n"
                "Compress a document's memory (KV cache) to a few percent, then ask about a fact hidden in it.")
    with gr.Tab("Compress & ask"):
        gr.Markdown("**Quick scenarios** (set the controls and run):")
        with gr.Row():
            scenario_buttons = [gr.Button(name, size="sm") for name in SCENARIOS]
        with gr.Row():
            sample = gr.Dropdown(choices=list(range(10)), value=0, label="Sample document (WikiText + 6 hidden facts)")
            first_qa = sample_document(0)[1]
            question = gr.Dropdown(choices=[q for q, _ in first_qa], value=first_qa[0][0], label="Question")
        custom = gr.Textbox(label="Or type your own question (no answer check)", placeholder="optional")
        with gr.Row():
            budget = gr.Slider(1.5, 20, value=4, step=0.5, label="Memory budget (% of the full FP16 cache)")
            in_prompt = gr.Checkbox(value=False, label="Question known before compression (in the prompt)")
            direct = gr.Checkbox(value=True, label="Answer directly from the compressed cache (no full decode)")
        with gr.Row():
            left = gr.Dropdown(list(METHODS), value="H2O (evict by attention)", label="Method A")
            right = gr.Dropdown(list(METHODS), value="DimmerKV (per-token precision)", label="Method B")
        with gr.Row():
            go = gr.Button("Compress & answer (A vs B)", variant="primary")
            go_all = gr.Button("Run all methods")
        header = gr.Markdown()
        with gr.Row():
            card_a = gr.Markdown()
            card_b = gr.Markdown()
        with gr.Accordion("Per-token heat maps (bytes kept per token; red outline = the answer)", open=True):
            with gr.Row():
                heat_a = gr.HTML()
                heat_b = gr.HTML()
        with gr.Accordion("All methods on this question", open=False) as all_box:
            board = gr.Dataframe(interactive=False, wrap=True)
            chart = gr.BarPlot(x="Method", y="× smaller", color="Correct", title="How much smaller (× vs FP16)",
                               height=320)
    with gr.Tab("How it works"):
        gr.HTML(architecture_svg())
        gr.Markdown(HOW_IT_WORKS)
    with gr.Tab("Results"):
        gr.Markdown(RESULTS)

    sample.change(update_questions, sample, question)
    run_inputs = [sample, question, custom, in_prompt, budget, left, right, direct]
    run_outputs = [header, card_a, card_b, heat_a, heat_b]
    for button, name in zip(scenario_buttons, SCENARIOS):
        button.click(lambda name=name: apply_scenario(name), None, [in_prompt, budget, left, right]) \
            .then(run, run_inputs, run_outputs)
    go.click(run, run_inputs, run_outputs)
    go_all.click(run_all, [sample, question, custom, in_prompt, budget, direct], [header, board, chart]) \
        .then(lambda: gr.update(open=True), None, all_box)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
