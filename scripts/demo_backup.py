"""Static backup of the demo: run its scenarios and save one HTML page.

Runs the three one-click scenarios of ``demo_app.py`` (and the all-methods
leaderboard for the first) with the real model, and writes
``docs/demo_backup.html``: each method's answer, size, GPU memory while
answering and timings, plus the per-token heat maps. Use it if the live demo
cannot run.
"""

from __future__ import annotations

import html
import time
from pathlib import Path

import demo_app as app

OUTPUT = Path(__file__).resolve().parent.parent / "docs" / "demo_backup.html"


def card_html(markdown: str) -> str:
    """Render the demo's small markdown cards as HTML (headings, bold, code, two-column tables)."""
    out, rows = [], []

    def inline(text: str) -> str:
        text = html.escape(text)
        parts = text.split("**")
        text = "".join(f"<b>{p}</b>" if i % 2 else p for i, p in enumerate(parts))
        parts = text.split("`")
        return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))

    for line in markdown.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-: ") or not any(cells):
                continue
            rows.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
            continue
        if rows:
            out.append("<table>" + "".join(rows) + "</table>")
            rows = []
        if line.startswith("### "):
            out.append(f"<h3>{inline(line[4:])}</h3>")
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
    if rows:
        out.append("<table>" + "".join(rows) + "</table>")
    return "\n".join(out)


def main() -> None:
    _, qa = app.sample_document(0)
    question = qa[1][0]
    sections = []
    for name, scenario in app.SCENARIOS.items():
        started = time.perf_counter()
        header, card_a, card_b, heat_a, heat_b = app.run(0, question, "", scenario["in_prompt"], scenario["budget"],
                                                         scenario["left"], scenario["right"], True)
        print(f"{name}: {time.perf_counter() - started:.1f} s")
        sections.append(f"""
<section>
  <h2>{html.escape(name)}</h2>
  <p class="meta">{card_html(header)}</p>
  <div class="pair"><div class="card">{card_html(card_a)}</div><div class="card">{card_html(card_b)}</div></div>
  <details><summary>Per-token heat maps (bytes kept per token; red outline = the answer)</summary>
    <div class="pair"><div class="heat">{heat_a}</div><div class="heat">{heat_b}</div></div>
  </details>
</section>""")
    first = app.SCENARIOS[next(iter(app.SCENARIOS))]
    _, table, _ = app.run_all(0, question, "", first["in_prompt"], first["budget"], True)
    leaderboard = table.to_html(index=False, na_rep="—", border=0, classes="board")
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DimmerKV Demo Backup</title>
<style>
  :root {{ --bg:#fbfaf7; --surface:#fff; --text:#1d1c1a; --muted:#5f5c55; --line:#dcd8cf; --accent:#b7791f; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#141413; --surface:#1c1c1a; --text:#ecebe6; --muted:#a9a69d; --line:#3a3934; --accent:#e3a548; }} }}
  :root[data-theme="dark"] {{ --bg:#141413; --surface:#1c1c1a; --text:#ecebe6; --muted:#a9a69d; --line:#3a3934; --accent:#e3a548; }}
  body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }}
  main {{ max-width:1200px; margin:0 auto; padding:24px 16px 64px; }}
  h1 {{ font-size:28px; margin:8px 0; }} h2 {{ font-size:21px; margin:36px 0 8px; color:var(--accent); }}
  h3 {{ font-size:17px; margin:0 0 8px; }} .meta {{ color:var(--muted); }}
  .pair {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:14px; }}
  .card {{ background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:14px; overflow-x:auto; }}
  .card table {{ border-collapse:collapse; width:100%; font-size:14px; }} .card td {{ border-top:1px solid var(--line); padding:5px 6px; vertical-align:top; }}
  details {{ margin-top:10px; }} .heat {{ background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:10px; overflow:hidden; }}
  table.board {{ border-collapse:collapse; width:100%; font-size:14px; background:var(--surface); }}
  table.board th, table.board td {{ border-bottom:1px solid var(--line); padding:6px 8px; text-align:left; }}
  .wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:12px; }}
  code {{ background:rgba(127,127,127,.15); padding:1px 5px; border-radius:4px; }}
</style></head>
<body><main>
<h1>DimmerKV demo — static backup</h1>
<p class="meta">Recorded with Llama-3.2-1B on the laptop GPU from the live demo's three scenarios (sample document 0, question: “{html.escape(question)}”). Answers come directly from the compressed cache where the method supports it.</p>
{''.join(sections)}
<h2>All methods on scenario 1</h2>
<div class="wrap">{leaderboard}</div>
</main></body></html>"""
    OUTPUT.write_text(page, encoding="utf-8")
    print(f"saved: {OUTPUT}")


if __name__ == "__main__":
    main()
