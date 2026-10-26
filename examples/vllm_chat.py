"""Multi-turn chat in vLLM with history restored from the DimmerKV store (needs a CUDA GPU).

    dimmerkv calibrate --model meta-llama/Llama-3.2-1B-Instruct --output checkpoints/llama-3.2-1b.pt
    python examples/vllm_chat.py --checkpoint checkpoints/llama-3.2-1b.pt

Turn 1 is answered normally and stored compressed when it finishes. The GPU
prefix cache is then cleared, as if the conversation had been evicted, so
turn 2's history must come back from the store (watch for "DimmerKV" in the
log and the lower time to first token compared with --no-dimmerkv).
"""

from __future__ import annotations

import argparse
import time

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--store", default="~/.cache/dimmerkv/vllm-demo")
    parser.add_argument("--budget", type=float, default=0.03)
    parser.add_argument("--document", help="Text file to discuss (default: a generated filler document).")
    parser.add_argument("--no-dimmerkv", action="store_true")
    args = parser.parse_args()

    kv_transfer_config = None if args.no_dimmerkv else KVTransferConfig(
        kv_connector="DimmerKVConnector",
        kv_connector_module_path="dimmerkv.integrations.vllm.connector",
        kv_role="kv_both",
        kv_connector_extra_config={"checkpoint": args.checkpoint, "store_path": args.store, "budget": args.budget},
    )
    # Leave GPU memory for the codec (basis + compression temporaries).
    llm = LLM(model=args.model, gpu_memory_utilization=0.7, kv_transfer_config=kv_transfer_config)
    greedy = SamplingParams(temperature=0, max_tokens=64)

    text = open(args.document).read() if args.document else "\n".join(
        f"Section {i}: The committee met on day {i} and reviewed budget line {i * 37 % 101}." for i in range(400))
    messages = [{"role": "user", "content": f"{text}\n\nThe secret code is 6142. Summarize the document in one line."}]
    started = time.perf_counter()
    reply = llm.chat(messages, greedy)[0]
    print(f"turn 1 ({len(reply.prompt_token_ids)} prompt tokens, {time.perf_counter() - started:.2f} s):",
          reply.outputs[0].text)
    messages.append({"role": "assistant", "content": reply.outputs[0].text})

    llm.reset_prefix_cache()  # Pretend the conversation was evicted from the GPU while the user was away.
    messages.append({"role": "user", "content": "What was the secret code?"})
    started = time.perf_counter()
    reply = llm.chat(messages, greedy)[0]
    print(f"turn 2 ({len(reply.prompt_token_ids)} prompt tokens, {time.perf_counter() - started:.2f} s):",
          reply.outputs[0].text)


if __name__ == "__main__":
    main()
