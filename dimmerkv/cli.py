"""``dimmerkv`` command line: calibrate a model, convert old checkpoints, inspect a store."""

from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="dimmerkv", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fit = commands.add_parser("calibrate", help="Fit the PCA basis and rung ladder for a model (once per model).")
    fit.add_argument("--model", required=True, help="Hugging Face model id or path.")
    fit.add_argument("--output", required=True)
    fit.add_argument("--chunk-tokens", type=int, default=1024)
    fit.add_argument("--chunks", type=int, default=100)
    fit.add_argument("--sinks", type=int, default=4)
    fit.add_argument("--group", type=int, default=64)
    fit.add_argument("--sample-every", type=int, default=12)
    fit.add_argument("--text-file", help="Calibrate on this text instead of WikiText-2 (needs no `datasets`).")
    fit.add_argument("--device", help="cuda, mps or cpu (default: best available).")

    convert = commands.add_parser("convert", help="Add RoPE tables and a fingerprint to a scripts/fit_kvtc.py file.")
    convert.add_argument("checkpoint")
    convert.add_argument("--output", required=True)
    convert.add_argument("--model", help="Model id for the RoPE config (default: the one in the checkpoint).")

    stats = commands.add_parser("stats", help="Summarize a conversation store.")
    stats.add_argument("store")
    stats.add_argument("--checkpoint", required=True)

    args = parser.parse_args(argv)
    if args.command == "calibrate":
        from .calibrate import calibrate

        calibrate(args.model, args.output, chunk_tokens=args.chunk_tokens, chunks=args.chunks, sinks=args.sinks,
                  group=args.group, sample_every=args.sample_every, text_file=args.text_file, device=args.device)
    elif args.command == "convert":
        from .calibrate import convert_legacy

        saved = convert_legacy(args.checkpoint, args.output, args.model)
        print(f"saved {args.output} (fingerprint {saved['fingerprint'].hex()})")
    elif args.command == "stats":
        from .codec import read_fingerprint
        from .store import ConversationStore

        print(json.dumps(ConversationStore(args.store, read_fingerprint(args.checkpoint)).stats(), indent=2))


if __name__ == "__main__":
    main()
