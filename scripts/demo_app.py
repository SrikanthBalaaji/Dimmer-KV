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


def run_method(label: str, budget: float, head_ids, tail_ids, keys, values, scores):
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
        reply = answer_with(q_keys, q_values, compressed.evicted, head_ids, tail_ids)
    return {
        "name": name, "reply": reply, "stored": stored, "full": full_bytes, "fetched": fetched,
        "compress_ms": compressed.seconds * 1000, "restore_ms": restore_ms, "refine_ms": refine_ms,
        "token_bytes": compressed.token_bytes,
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


def run(sample: int, question_choice: str, custom_question: str, in_prompt: bool, budget_percent: float,
        left: str, right: str):
    context, qa = sample_document(int(sample))
    expected = None
    if custom_question.strip():
        question = custom_question.strip()
    else:
        index = [q for q, _ in qa].index(question_choice) if question_choice in [q for q, _ in qa] else 0
        question, expected = qa[index]
    head_ids, tail_ids, keys, values, scores, prefill_ms = prefill(context, question, in_prompt)
    budget = budget_percent / 100
    outputs = []
    for label in (left, right):
        try:
            result = run_method(label, budget, head_ids, tail_ids, keys, values, scores)
            outputs.append((card(label, result, expected), heat_map(head_ids, result["token_bytes"], expected)))
        except (ValueError, AssertionError) as error:
            outputs.append((f"### {label}\n\n⚠️ {error}\n\n(kvtc needs ≥ ~2.8% at 3K tokens; "
                            "use DimmerKV + eviction or a larger budget.)", ""))
    header = (f"**Document:** {head_ids.shape[0]:,} tokens · full KV cache "
              f"{head_ids.shape[0] * config.fp16_bytes_per_token() / 2**20:.1f} MB · re-reading it (prefill) "
              f"took {prefill_ms:.0f} ms · question {'in the prompt' if in_prompt else 'asked after compression'}")
    return header, outputs[0][0], outputs[1][0], outputs[0][1], outputs[1][1]


def update_questions(sample: int):
    _, qa = sample_document(int(sample))
    choices = [q for q, _ in qa]
    return gr.update(choices=choices, value=choices[0])


RESULTS = """
### Measured results (Llama-3.2-1B on the laptop, 3B on an AWS L40S)

**Question in the prompt, 8K-token documents, 24 questions** (full cache: 21/24)

| method | memory | correct |
|---|---|---|
| SnapKV | 1.5% | 6/24 |
| **DimmerKV + question scores + eviction** | **1.5% (67×)** | **21/24** |
| SnapKV | 2% | 10/24 |
| **DimmerKV + question scores + eviction** | **2% (50×)** | **22/24** |

**Question asked after compression, 8K tokens** (full cache: 21/24, ans p .75)

| method | memory | correct / ans p |
|---|---|---|
| H2O | 2% | 0/24 / .00 |
| kvtc | 2% | 17/24 / .51 |
| **DimmerKV progressive, refine top 1% at question time (+0.5% fetched)** | 2% | **19/24 / .71** |

**Serving, 3B model, 16K-token conversations** (18 questions)

| way of keeping a conversation | stored | conversations in 8 GiB | correct |
|---|---|---|---|
| full cache in RAM | 1,753 MiB | 4 | 18/18 |
| H2O 2% | 35 MiB | 230 | 0/18 |
| kvtc / DimmerKV 2% | 31 MiB | 263 | 18/18 |

Eviction (H2O, SnapKV) loses facts that a question needs; DimmerKV keeps every
token at some precision and spends bytes where they matter. Details: FINDINGS.md.
"""

with gr.Blocks(title="DimmerKV") as demo:
    gr.Markdown("# DimmerKV — dim the KV cache instead of deleting it\n"
                "Compress a document's memory (KV cache), then ask about a fact hidden in it.")
    with gr.Tab("Compress & ask"):
        with gr.Row():
            sample = gr.Dropdown(choices=list(range(10)), value=0, label="Sample document (WikiText + 6 hidden facts)")
            first_qa = sample_document(0)[1]
            question = gr.Dropdown(choices=[q for q, _ in first_qa], value=first_qa[0][0], label="Question")
        custom = gr.Textbox(label="Or type your own question (no answer check)", placeholder="optional")
        with gr.Row():
            budget = gr.Slider(1.5, 20, value=4, step=0.5, label="Memory budget (% of the full FP16 cache)")
            in_prompt = gr.Checkbox(value=False, label="Question known before compression (in the prompt)")
        with gr.Row():
            left = gr.Dropdown(list(METHODS), value="H2O (evict by attention)", label="Method A")
            right = gr.Dropdown(list(METHODS), value="DimmerKV (per-token precision)", label="Method B")
        go = gr.Button("Compress & answer", variant="primary")
        header = gr.Markdown()
        with gr.Row():
            card_a = gr.Markdown()
            card_b = gr.Markdown()
    with gr.Tab("Heat map"):
        gr.Markdown("Bytes kept per token for the last run (method A, then method B).")
        heat_a = gr.HTML()
        heat_b = gr.HTML()
    with gr.Tab("Results"):
        gr.Markdown(RESULTS)
    sample.change(update_questions, sample, question)
    go.click(run, [sample, question, custom, in_prompt, budget, left, right], [header, card_a, card_b, heat_a, heat_b])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
