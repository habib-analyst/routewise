"""RouteWise router service.

An OpenAI-style front door in front of two local backends (a small and a
large Qwen2.5 model, each served by backend_server.py). For every request
it:

1. classifies the task (factual / summary / code / reasoning) and scores
   its difficulty (policy.py),
2. assigns a token budget from the caller's quality tier
   (economy / balanced / premium) and enforces it: over-cap prompts are
   truncated at token level, output is capped, and prompts beyond the
   hard context limit are rejected,
3. routes to the cheapest backend the policy says can handle it,
4. appends one JSON line per request to logs/request_log.jsonl with the
   budget vs the actual token usage.

Usage:
    POST /v1/chat/completions
    body: {"model": "routewise-auto", "messages": [...],
           "quality_tier": "balanced", "max_tokens": 128}
    model "small" or "large" forces a backend (the benchmark uses this
    for the all-small and all-large strategies); budgeting still applies.
    quality_tier may also be sent as the X-Quality-Tier header.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from pathlib import Path

import httpx
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer

from policy import (
    HARD_INPUT_LIMIT,
    TIER_BUDGETS,
    choose_backend,
    classify_task,
    difficulty_score,
    plan_budget,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_DIR / "logs" / "request_log.jsonl"

SMALL_URL = os.environ.get("ROUTEWISE_SMALL_URL", "http://127.0.0.1:8101")
LARGE_URL = os.environ.get("ROUTEWISE_LARGE_URL", "http://127.0.0.1:8102")
BACKEND_URLS = {"small": SMALL_URL, "large": LARGE_URL}

# Both backends are Qwen2.5 models and share one tokenizer, so the router
# can count and truncate tokens exactly the way the backends will see
# them. Only the tokenizer is loaded here, never the weights.
TOKENIZER_PATH = os.environ.get(
    "ROUTEWISE_TOKENIZER_PATH",
    str(
        PROJECT_DIR.parent
        / "servecore"
        / "models"
        / "Qwen2.5-0.5B-Instruct"
    ),
)
_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

app = FastAPI(title="routewise-router")


def _prompt_text(messages: list[dict]) -> str:
    return _tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _count_tokens(messages: list[dict]) -> int:
    return len(_tokenizer(_prompt_text(messages))["input_ids"])


def _truncate_messages(messages: list[dict], target_tokens: int) -> list[dict]:
    """Cut the longest user message (head 70% + tail 30% of its tokens)
    until the whole templated prompt fits target_tokens."""
    messages = [dict(m) for m in messages]
    while _count_tokens(messages) > target_tokens:
        idx = max(
            (i for i, m in enumerate(messages) if m["role"] == "user"),
            key=lambda i: len(_tokenizer(messages[i]["content"])["input_ids"]),
            default=None,
        )
        if idx is None:
            break
        ids = _tokenizer(messages[idx]["content"])["input_ids"]
        if len(ids) <= 16:
            break
        keep = max(16, int(len(ids) * 0.6))
        head = int(keep * 0.7)
        tail = keep - head
        messages[idx]["content"] = _tokenizer.decode(ids[:head] + ids[-tail:])
    return messages


def _log(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as fh:
        fh.write(json.dumps(record) + "\n")


@app.get("/health")
def health():
    return {"status": "ok", "backends": BACKEND_URLS, "tiers": list(TIER_BUDGETS)}


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [
            {"id": "routewise-auto", "object": "model"},
            {"id": "small", "object": "model"},
            {"id": "large", "object": "model"},
        ],
    }


@app.get("/stats")
def stats():
    if not LOG_PATH.exists():
        return {"requests": 0}
    rows = [json.loads(line) for line in LOG_PATH.read_text().splitlines() if line]
    by_backend: dict[str, int] = {}
    for row in rows:
        by_backend[row["backend"]] = by_backend.get(row["backend"], 0) + 1
    return {
        "requests": len(rows),
        "by_backend": by_backend,
        "total_prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in rows),
        "total_completion_tokens": sum(
            r["usage"]["completion_tokens"] for r in rows
        ),
        "budget_overruns": sum(
            1
            for r in rows
            if r["usage"]["total_tokens"] > r["budget"]["input_cap"] + r["budget"]["output_cap"]
        ),
    }


@app.post("/v1/chat/completions")
async def chat(request: Request, x_quality_tier: str | None = Header(default=None)):
    body = await request.json()
    messages = body.get("messages", [])
    model_field = body.get("model", "routewise-auto")
    tier = body.get("quality_tier") or x_quality_tier or "balanced"
    if tier not in TIER_BUDGETS:
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"unknown quality_tier: {tier}"}},
        )

    request_id = uuid.uuid4().hex[:12]
    user_text = " ".join(
        m.get("content", "") for m in messages if m.get("role") == "user"
    )
    prompt_tokens = _count_tokens(messages)
    task_type = classify_task(user_text, prompt_tokens)
    score = difficulty_score(task_type, prompt_tokens)
    plan = plan_budget(tier, task_type, prompt_tokens, body.get("max_tokens"))

    record = {
        "request_id": request_id,
        "ts": time.time(),
        "tier": tier,
        "task_type": task_type,
        "difficulty": score,
        "budget": {
            "input_cap": plan["input_cap"],
            "output_cap": plan["output_cap"],
        },
        "actions": list(plan["actions"]),
    }

    if plan["reject"]:
        record.update(
            {
                "backend": None,
                "status": "rejected",
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": 0,
                    "total_tokens": prompt_tokens,
                },
                "latency_s": 0.0,
            }
        )
        _log(record)
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": (
                        f"prompt of {prompt_tokens} tokens exceeds the hard "
                        f"input limit of {HARD_INPUT_LIMIT} tokens"
                    ),
                    "type": "routewise_budget_error",
                }
            },
        )

    if plan["truncate_input_to"] is not None:
        messages = _truncate_messages(messages, plan["truncate_input_to"])
        prompt_tokens = _count_tokens(messages)

    if model_field in ("small", "large"):
        backend, reason = model_field, f"forced by client model field"
        record["actions"].append("backend_forced")
    else:
        backend, reason = choose_backend(tier, task_type, prompt_tokens, score)

    started = time.perf_counter()
    # trust_env=False: backends are localhost; never route them through
    # any proxy configured in the environment.
    resp = None
    async with httpx.AsyncClient(timeout=3600, trust_env=False) as client:
        for attempt in (1, 2):
            try:
                resp = await client.post(
                    f"{BACKEND_URLS[backend]}/v1/chat/completions",
                    json={"messages": messages,
                          "max_tokens": plan["max_new_tokens"]},
                )
                break
            except httpx.ConnectError:
                if attempt == 2:
                    record.update(
                        {
                            "backend": backend,
                            "route_reason": reason,
                            "status": "backend_unreachable",
                            "usage": {"prompt_tokens": prompt_tokens,
                                      "completion_tokens": 0,
                                      "total_tokens": prompt_tokens},
                            "latency_s": round(time.perf_counter() - started, 4),
                            "backend_seconds": None,
                        }
                    )
                    _log(record)
                    return JSONResponse(
                        status_code=502,
                        content={"error": {
                            "message": f"backend '{backend}' unreachable",
                            "type": "routewise_backend_error",
                        }},
                    )
                await asyncio.sleep(3)
    latency = time.perf_counter() - started
    payload = resp.json()

    record.update(
        {
            "backend": backend,
            "route_reason": reason,
            "status": "ok" if resp.status_code == 200 else "backend_error",
            "usage": payload.get(
                "usage",
                {"prompt_tokens": prompt_tokens, "completion_tokens": 0,
                 "total_tokens": prompt_tokens},
            ),
            "latency_s": round(latency, 4),
            "backend_seconds": payload.get("routewise_backend_seconds"),
        }
    )
    _log(record)

    payload["routewise"] = {
        "request_id": request_id,
        "tier": tier,
        "task_type": task_type,
        "difficulty": score,
        "backend": backend,
        "route_reason": reason,
        "budget": record["budget"],
        "actions": record["actions"],
    }
    return JSONResponse(status_code=resp.status_code, content=payload)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8100, log_level="warning")
