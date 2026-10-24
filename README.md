# DimmerKV

Dim the KV cache instead of deleting it: per-token precision on top of kvtc-style transform coding,
question-aware scores, and question-time refinement (Hashcode 14.0, Team Deadoop).

- Overview, pipeline and architecture diagram: `docs/DimmerKV_overview.html` (open in a browser)
- Detailed experiments and numbers: `FINDINGS.md`; raw results: `results/`
- Demo: `python scripts/demo_app.py` (Gradio, http://127.0.0.1:7860; needs `fit_kvtc.py` and `fit_progressive.py` checkpoints)
