"""Solo decode-rate micro-benchmark for the two RouteWise backends.

Loads one model at a time (bf16, greedy), warms up with 8 tokens, then
times a 32-token generation on a short prompt. Written so the router's
output caps and timeouts can be set from measured rates on this shared
VM rather than guessed. Result: results/micro_bench.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS = {
    "small": PROJECT_DIR.parent / "servecore" / "models" / "Qwen2.5-0.5B-Instruct",
    "large": PROJECT_DIR.parent / "cachelab" / "models" / "Qwen2.5-1.5B-Instruct",
}
PROMPT = "What is the capital of France, and name one famous museum there."


def bench(path: Path) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
    model.eval()
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt")

    def gen(n: int) -> float:
        started = time.perf_counter()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=n, do_sample=False,
                           pad_token_id=tokenizer.eos_token_id)
        return time.perf_counter() - started

    gen(8)  # warmup
    seconds = gen(32)
    del model
    return {"seconds_for_32_tokens": round(seconds, 2),
            "tokens_per_second": round(32 / seconds, 3)}


def main() -> None:
    out = {}
    for name, path in MODELS.items():
        out[name] = bench(path)
        print(name, out[name], flush=True)
    (PROJECT_DIR / "results" / "micro_bench.json").write_text(
        json.dumps(out, indent=2)
    )


if __name__ == "__main__":
    main()
