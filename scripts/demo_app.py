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


def _fmt_ms(ms) -> str:
    return "—" if ms is None else f"{ms:,.0f} ms"


def card(label: str, result: dict, expected: str | None) -> str:
    """One method's result as an HTML card: answer, verdict, two headline numbers, details."""
    if expected is None:
        pill = '<span class="dk-pill">no answer check</span>'
    elif expected in result["reply"]:
        pill = '<span class="dk-pill ok">Correct</span>'
    else:
        pill = f'<span class="dk-pill bad">Wrong · expected {html.escape(expected)}</span>'
    ratio = result["full"] / max(result["stored"], 1)
    per_ram = int(RAM_GIB * 2**30 // max(result["stored"], 1))
    held = result["gpu_held"] / 2**20
    gpu_ratio = result["full"] / max(result["gpu_held"], 1)
    rows = [
        ("Stored", f'{result["stored"] / 2**20:.2f} MB of {result["full"] / 2**20:.1f} MB'),
        (f"Conversations in {RAM_GIB} GiB RAM", f"{per_ram:,}"),
        ("GPU memory while answering", f"{held:.1f} MB · {html.escape(result['path'])}"),
        ("Compress", _fmt_ms(result["compress_ms"])),
    ]
    if result["restore_ms"] is not None:
        rows.append(("Restore from bytes", _fmt_ms(result["restore_ms"])))
    if result["refine_ms"] is not None:
        rows.append(("Question-time refinement",
                     f'{result["fetched"] / 2**20:.2f} MB in {_fmt_ms(result["refine_ms"])}'))
    rows.append(("Answer", _fmt_ms(result["answer_ms"])))
    details = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows)
    gpu_stat = (f'<div><b>{gpu_ratio:.0f}×</b><span>less GPU memory while answering</span></div>'
                if gpu_ratio >= 1.5 else
                f'<div><b>{held:.0f} MB</b><span>GPU memory while answering</span></div>')
    return f"""<div class="dk-card">
  <div class="dk-card-head"><span class="dk-method">{html.escape(label)}</span>{pill}</div>
  <div class="dk-answer">{html.escape(result["reply"] or "(empty)")}</div>
  <div class="dk-stats"><div><b>{ratio:.0f}×</b><span>smaller than the FP16 cache</span></div>{gpu_stat}</div>
  {token_summary(result["token_bytes"])}
  <dl class="dk-details">{details}</dl>
</div>"""


def error_card(label: str, error: Exception) -> str:
    return f"""<div class="dk-card">
  <div class="dk-card-head"><span class="dk-method">{html.escape(label)}</span><span class="dk-pill warn">Budget too small</span></div>
  <p class="dk-note">{html.escape(str(error))}</p>
  <p class="dk-note">kvtc needs about 2.8% at 3K tokens. Raise the budget or use DimmerKV + eviction.</p>
</div>"""


def heat_map(head_ids, token_bytes: torch.Tensor | None, highlight: str | None) -> str:
    """Tokens shaded by stored bytes on a log scale (dark = dropped, bright = exact)."""
    if token_bytes is None:
        return '<p class="dk-note">No per-token data for this method.</p>'
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
            style = "background:#24272d;color:#8b949e;"
        else:
            level = max(0.0, min(1.0, (math.log10(size / full) + 2) / 2))  # 1% -> 0, 100% -> 1
            light = 18 + 62 * level
            style = f"background:hsl(45,95%,{light:.0f}%);color:{'#111' if light > 50 else '#f0f0f0'};"
        if i in marked:
            style += "outline:2px solid #e5484d;font-weight:700;"
        out.append(f'<span title="{size:.0f} B" style="{style}">{html.escape(piece)}</span>')
    legend = ('<div class="dk-legend"><span>Bytes kept per token, log scale</span>'
              '<i style="background:#24272d"></i>dropped'
              '<i style="background:hsl(45,95%,18%)"></i>~1%'
              '<i style="background:hsl(45,95%,49%)"></i>~10%'
              '<i style="background:hsl(45,95%,80%)"></i>exact'
              '<i style="outline:2px solid #e5484d;background:transparent"></i>answer</div>')
    return legend + '<div class="dk-heat">' + "".join(out) + "</div>"


def token_summary(token_bytes: torch.Tensor | None) -> str:
    """A stacked bar: share of the document's tokens at each precision."""
    if token_bytes is None:
        return ""
    full = 2.0 * config.layers * config.kv_heads * config.head_dim * 2
    share = (token_bytes.float() / full).cpu()
    groups = [
        ("dropped", share <= 0, "#24272d"),
        ("coarse", (share > 0) & (share < 0.03), "hsl(45,95%,26%)"),
        ("medium", (share >= 0.03) & (share < 0.2), "hsl(45,95%,45%)"),
        ("fine", (share >= 0.2) & (share < 0.99), "hsl(45,95%,64%)"),
        ("exact", share >= 0.99, "hsl(45,95%,82%)"),
    ]
    bar, keys = [], []
    for name, mask, color in groups:
        fraction = float(mask.float().mean())
        if fraction <= 0:
            continue
        bar.append(f'<span style="width:{fraction * 100:.2f}%;background:{color}" title="{name} {fraction:.0%}"></span>')
        keys.append(f'<span><i style="background:{color}"></i>{name} {fraction:.0%}</span>')
    return f'<div class="dk-mix"><div class="dk-bar">{"".join(bar)}</div><div class="dk-keys">{"".join(keys)}</div></div>'


def resolve_question(sample: int, question_choice: str, custom_question: str):
    context, qa = sample_document(int(sample))
    if custom_question.strip():
        return context, custom_question.strip(), None
    questions = [q for q, _ in qa]
    index = questions.index(question_choice) if question_choice in questions else 0
    return context, qa[index][0], qa[index][1]


def document_header(head_ids, prefill_ms: float, in_prompt: bool) -> str:
    tokens = head_ids.shape[0]
    chips = [
        ("Document", f"{tokens:,} tokens"),
        ("Full KV cache", f"{tokens * config.fp16_bytes_per_token() / 2**20:.1f} MB"),
        ("Prefill", f"{prefill_ms:,.0f} ms"),
        ("Question", "known before compression" if in_prompt else "asked after compression"),
    ]
    return '<div class="dk-chips">' + "".join(f"<div><span>{k}</span><b>{v}</b></div>" for k, v in chips) + "</div>"


def run(sample: int, question_choice: str, custom_question: str, in_prompt: bool, budget_percent: float,
        left: str, right: str, direct: bool = False):
    context, question, expected = resolve_question(sample, question_choice, custom_question)
    head_ids, tail_ids, keys, values, scores, prefill_ms = prefill(context, question, in_prompt)
    budget = budget_percent / 100
    outputs = []
    for label in (left, right):
        try:
            result = run_method(label, budget, head_ids, tail_ids, keys, values, scores, direct)
            outputs.append((card(label, result, expected), heat_map(head_ids, result["token_bytes"], expected)))
        except (ValueError, AssertionError) as error:
            outputs.append((error_card(label, error), ""))
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
        correct = ("Yes" if expected in r["reply"] else "No") if expected else "?"
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
    return ('<div style="background:#fbfaf7;border:1px solid #e4e1da;border-radius:14px;padding:12px;'
            f'overflow-x:auto">{match.group(0)}</div>')


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

Savings are in storage and transfer; with direct mode on, the GPU also holds only the codes while answering.
Details: `FINDINGS.md`, `docs/DimmerKV_overview.html`.
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.amber,
    neutral_hue=gr.themes.colors.stone,
    radius_size=gr.themes.sizes.radius_md,
    font=["Inter", "ui-sans-serif", "system-ui", "Segoe UI", "sans-serif"],
    font_mono=["JetBrains Mono", "ui-monospace", "Consolas", "monospace"],
).set(
    body_background_fill="#f7f6f3",
    body_background_fill_dark="#111214",
    block_background_fill="#ffffff",
    block_background_fill_dark="#18191c",
    block_border_width="1px",
    block_shadow="none",
    block_label_text_size="*text_sm",
    button_primary_background_fill="#1d1c1a",
    button_primary_background_fill_hover="#3a3834",
    button_primary_text_color="#ffffff",
    button_primary_background_fill_dark="#e3a548",
    button_primary_background_fill_hover_dark="#efb965",
    button_primary_text_color_dark="#111214",
    slider_color="#b7791f",
    slider_color_dark="#e3a548",
)

CSS = """
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }
footer { display: none !important; }
.dk-top { display:flex; align-items:baseline; justify-content:space-between; gap:16px; flex-wrap:wrap;
          padding: 8px 2px 4px; }
.dk-top h1 { font-size: 30px; font-weight: 700; letter-spacing: -0.02em; margin: 0; }
.dk-top h1 em { font-style: normal; color: #b7791f; }
.dk-top p { margin: 4px 0 0; color: var(--body-text-color-subdued); font-size: 15px; }
.dk-top .dk-env { font-size: 13px; color: var(--body-text-color-subdued); font-family: var(--font-mono); }
.dk-section { font-size: 12px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
              color: var(--body-text-color-subdued); margin: 18px 2px 6px; }
button.dk-scenario, .dk-scenario button { justify-content: flex-start !important; text-align: left !important;
                      font-weight: 500 !important; border: 1px solid var(--border-color-primary) !important;
                      background: var(--block-background-fill) !important; min-height: 52px; padding: 0 16px !important; }
button.dk-scenario:hover, .dk-scenario button:hover { border-color: #b7791f !important; }
.dk-chips { display:grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }
.dk-chips div { border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 10px 14px;
                background: var(--block-background-fill); }
.dk-chips span { display:block; font-size: 12px; color: var(--body-text-color-subdued); }
.dk-chips b { font-size: 16px; font-weight: 600; }
.dk-card { border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 18px 20px;
           background: var(--block-background-fill); display:flex; flex-direction:column; gap: 14px; height: 100%; }
.dk-card-head { display:flex; justify-content:space-between; align-items:center; gap: 10px; flex-wrap: wrap; }
.dk-method { font-weight: 600; font-size: 15px; }
.dk-pill { font-size: 12px; font-weight: 600; padding: 3px 10px; border-radius: 999px;
           background: rgba(127,127,127,.14); color: var(--body-text-color-subdued); white-space: nowrap; }
.dk-pill.ok { background: rgba(48,164,108,.14); color: #2f9e67; }
.dk-pill.bad { background: rgba(229,72,77,.13); color: #e5484d; }
.dk-pill.warn { background: rgba(183,121,31,.15); color: #b7791f; }
.dk-answer { font-family: var(--font-mono); font-size: 14px; padding: 10px 12px; border-radius: 8px;
             background: rgba(127,127,127,.08); word-break: break-word; }
.dk-stats { display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.dk-stats b { display:block; font-size: 34px; font-weight: 700; letter-spacing: -0.02em; line-height: 1.1; }
.dk-stats span { font-size: 12px; color: var(--body-text-color-subdued); }
.dk-mix .dk-bar { display:flex; height: 10px; border-radius: 5px; overflow: hidden; background: rgba(127,127,127,.12); }
.dk-mix .dk-bar span { display:block; height: 100%; }
.dk-keys { display:flex; flex-wrap: wrap; gap: 4px 14px; margin-top: 8px; font-size: 12px;
           color: var(--body-text-color-subdued); }
.dk-keys i, .dk-legend i { display:inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 5px;
                           vertical-align: -1px; }
.dk-details { display:grid; grid-template-columns: auto 1fr; gap: 6px 16px; margin: 0; font-size: 13px;
              border-top: 1px solid var(--border-color-primary); padding-top: 12px; }
.dk-details dt { color: var(--body-text-color-subdued); }
.dk-details dd { margin: 0; text-align: right; font-variant-numeric: tabular-nums; }
.dk-note { color: var(--body-text-color-subdued); font-size: 14px; margin: 0; }
.dk-legend { display:flex; flex-wrap: wrap; align-items:center; gap: 4px 12px; font-size: 12px;
             color: var(--body-text-color-subdued); margin-bottom: 8px; }
.dk-legend span { font-weight: 600; margin-right: 4px; }
.dk-heat { max-height: 380px; overflow-y: auto; line-height: 1.75; font-family: var(--font-mono); font-size: 12px;
           white-space: pre-wrap; word-break: break-word; border: 1px solid var(--border-color-primary);
           border-radius: 10px; padding: 12px; }
.dk-heat span { border-radius: 2px; }
.dk-prose { max-width: 920px; }
.dk-prose table { font-size: 14px; }
"""

TOP = f"""<div class="dk-top">
  <div><h1>Dimmer<em>KV</em></h1>
  <p>Compress a document's KV cache to a few percent, then ask about a fact hidden in it.</p></div>
  <div class="dk-env">{DEFAULT_MODEL.split("/")[-1]} · {torch.cuda.get_device_name(0) if torch.cuda.is_available() else device}</div>
</div>"""

with gr.Blocks(title="DimmerKV") as demo:
    gr.HTML(TOP)
    with gr.Tab("Compress & ask"):
        gr.HTML('<div class="dk-section">Scenarios</div>')
        with gr.Row(equal_height=True):
            scenario_buttons = [gr.Button(name, elem_classes="dk-scenario") for name in SCENARIOS]
        gr.HTML('<div class="dk-section">Setup</div>')
        with gr.Group():
            with gr.Row():
                sample = gr.Dropdown(choices=list(range(10)), value=0, scale=1,
                                     label="Document (WikiText + 6 hidden facts)")
                first_qa = sample_document(0)[1]
                question = gr.Dropdown(choices=[q for q, _ in first_qa], value=first_qa[0][0], scale=2,
                                       label="Question")
                custom = gr.Textbox(label="Or your own question (no answer check)", placeholder="Optional", scale=2)
            with gr.Row():
                budget = gr.Slider(1.5, 20, value=4, step=0.5, scale=2, label="Memory budget, % of the FP16 cache")
                in_prompt = gr.Checkbox(value=False, scale=1, label="Question known before compression")
                direct = gr.Checkbox(value=True, scale=1, label="Answer directly from the compressed cache")
            with gr.Row():
                left = gr.Dropdown(list(METHODS), value="H2O (evict by attention)", label="Method A")
                right = gr.Dropdown(list(METHODS), value="DimmerKV (per-token precision)", label="Method B")
        with gr.Row():
            go = gr.Button("Compress and answer", variant="primary", scale=3)
            go_all = gr.Button("Run every method", scale=1)
        gr.HTML('<div class="dk-section">Result</div>')
        header = gr.HTML()
        with gr.Row(equal_height=True):
            card_a = gr.HTML()
            card_b = gr.HTML()
        with gr.Accordion("Per-token view: bytes kept for every token (red outline = the answer)", open=True):
            with gr.Row():
                heat_a = gr.HTML()
                heat_b = gr.HTML()
        with gr.Accordion("Every method on this question", open=False) as all_box:
            board = gr.Dataframe(interactive=False, wrap=True)
            chart = gr.BarPlot(x="Method", y="× smaller", color="Correct", title="Size reduction vs FP16 (×)",
                               height=320)
    with gr.Tab("How it works"):
        gr.HTML(architecture_svg())
        gr.Markdown(HOW_IT_WORKS, elem_classes="dk-prose")
    with gr.Tab("Results"):
        gr.Markdown(RESULTS, elem_classes="dk-prose")

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
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=THEME, css=CSS)
