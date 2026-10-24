"""Run the full evaluation grid for one model and append results to a CSV.

For every budget, runs each method template (``{b}`` is replaced by the
budget) on the needle test in both modes and on perplexity. Rows are
appended after each task finishes, so a partial run still leaves results.
Every row carries the model, task, method, bytes, quality metrics and
timings (prefill, compression, generation) plus peak GPU memory.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch

import dimmer
from common import load_model
from eval_perplexity import run_perplexity
from methods import Methods
from test_generation import print_rows, run_needles

DEFAULT_TEMPLATES = ("lex@{b}", "h2o:{b}", "h2o:{b}:snap", "dkv:{b}:flat", "dkv:{b}:snap")
FIELDS = (
    "model", "task", "method", "budget", "bytes_ratio", "n", "correct", "accuracy", "agree", "answer_nll",
    "answer_p", "ppl", "context_tokens", "prefill_ms", "compress_ms", "gen_ms_per_token", "peak_gpu_mb",
    "levels", "finished",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dictionary", required=True, help="Lexico dictionary fit for this model.")
    parser.add_argument("--basis", default="checkpoints/pca_basis.pt", help="Only needed for pca methods.")
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.10, 0.15, 0.20])
    parser.add_argument("--templates", nargs="+", default=list(DEFAULT_TEMPLATES))
    parser.add_argument("--tasks", nargs="+", default=["needle_after", "needle_prompt", "perplexity"],
                        choices=("needle_after", "needle_prompt", "perplexity"))
    parser.add_argument("--context-tokens", type=int, default=4000)
    parser.add_argument("--seeds", type=int, default=9, help="Needle contexts; 6 questions each.")
    parser.add_argument("--ppl-chunks", type=int, default=8)
    parser.add_argument("--ppl-eval-tokens", type=int, default=512)
    parser.add_argument("--recent", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("results/grid.csv"))
    args = parser.parse_args()

    names = [template.format(b=f"{budget:.2f}") for budget in args.budgets for template in args.templates]
    tokenizer, model, device = load_model(args.model)
    dimmer.install(model)
    methods = Methods(args.model, args.basis, args.dictionary, device, args.recent)
    methods.codec("lex")  # Fail early if the dictionary does not match the model.

    args.output.parent.mkdir(parents=True, exist_ok=True)
    new_file = not args.output.exists()
    with args.output.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for task in args.tasks:
            started = time.time()
            print(f"=== {args.model}: {task} ({len(names)} methods)", flush=True)
            if task == "perplexity":
                rows = run_perplexity(model, tokenizer, methods, names, args.context_tokens,
                                      args.ppl_eval_tokens, args.ppl_chunks)
            else:
                rows = run_needles(model, tokenizer, methods, names, task.split("_")[1], args.seeds,
                                   args.context_tokens)
                print_rows(rows)
            for row in rows:
                writer.writerow({**row, "model": args.model, "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
            handle.flush()
            print(f"=== {task} done in {(time.time() - started) / 60:.1f} min", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
