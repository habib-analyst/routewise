"""One RouteWise backend: a single Hugging Face model behind a minimal
OpenAI-style /v1/chat/completions endpoint.

Run one process per backend model:

    python3 backend_server.py --name small \
        --model-path /path/to/Qwen2.5-0.5B-Instruct --port 8101

Generation is greedy (temperature 0) so benchmark numbers are
reproducible. The response carries the backend's own generation time in
the 'routewise_backend_seconds' field so the router and the benchmark can
separate model time from HTTP and queueing overhead.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--model-path", required=True)
parser.add_argument("--port", type=int, required=True)
args = parser.parse_args()

torch.set_num_threads(2)  # this VM has 2 vCPUs; do not oversubscribe
print(f"[{args.name}] loading {args.model_path} ...", flush=True)
_tokenizer = AutoTokenizer.from_pretrained(args.model_path)
_model = AutoModelForCausalLM.from_pretrained(
    args.model_path, dtype=torch.bfloat16
)
_model.eval()


def _rss_mb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS"):
            return int(line.split()[1]) / 1024
    return -1.0


# Pin the weights in anonymous memory. Transformers memory-maps
# safetensors, so the weights live in file-backed pages. On this shared,
# swapless VM those pages are evicted under other tenants' cache
# pressure while the model sits idle, and the next request then pays a
# page-fault storm: measured 8,468 major faults in one generation, and
# 32-token requests taking 46 minutes when both backends were resident.
# Cloning every tensor into fresh storage moves the weights to
# anonymous pages, which a swapless kernel cannot reclaim.
# ROUTEWISE_PIN=0 skips pinning (roughly halves this process's RSS);
# the benchmark's launcher uses it for the small backend when the VM
# is too full to pin both models safely.
if os.environ.get("ROUTEWISE_PIN", "1") != "0":
    _rss_before = _rss_mb()
    with torch.no_grad():
        for _param in _model.parameters():
            _param.data = _param.data.clone()
        for _buf in _model.buffers():
            _buf.data = _buf.data.clone()
    print(f"[{args.name}] weights pinned: RSS {_rss_before:.0f} MB -> "
          f"{_rss_mb():.0f} MB", flush=True)
else:
    print(f"[{args.name}] pinning skipped (ROUTEWISE_PIN=0)", flush=True)

# Self-warmup: fault in anything left (embeddings lookup tables,
# allocator arenas) and run the kernels once, so 'ready' means ready.
_warm_inputs = _tokenizer("Hello.", return_tensors="pt")
with torch.no_grad():
    _model.generate(**_warm_inputs, max_new_tokens=8, do_sample=False,
                    pad_token_id=_tokenizer.eos_token_id)
del _warm_inputs
print(f"[{args.name}] ready on port {args.port}", flush=True)

app = FastAPI(title=f"routewise-backend-{args.name}")


class ChatRequest(BaseModel):
    messages: list[dict]
    max_tokens: int = 64
    model: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "backend": args.name}


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    prompt_text = _tokenizer.apply_chat_template(
        req.messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _tokenizer(prompt_text, return_tensors="pt")
    prompt_tokens = int(inputs["input_ids"].shape[1])

    started = time.perf_counter()
    with torch.no_grad():
        out = _model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id,
        )
    backend_seconds = time.perf_counter() - started

    completion_ids = out[0][prompt_tokens:]
    text = _tokenizer.decode(completion_ids, skip_special_tokens=True)
    return {
        "id": f"chatcmpl-{args.name}",
        "object": "chat.completion",
        "model": args.name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"
                if len(completion_ids) < req.max_tokens
                else "length",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": int(len(completion_ids)),
            "total_tokens": prompt_tokens + int(len(completion_ids)),
        },
        "routewise_backend_seconds": round(backend_seconds, 4),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
