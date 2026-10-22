"""RouteWise benchmark orchestrator.

Starts the two backend servers and the router, runs smoke tests
(including budget-enforcement probes), runs the LiteLLM interop check,
then runs the fixed workload under three strategies:

    all_small : every request forced to Qwen2.5-0.5B
    all_large : every request forced to Qwen2.5-1.5B
    routed    : model field 'routewise-auto', the policy decides

Budgets are identical across strategies because each workload item keeps
its own quality tier in all three runs. Results are appended to
results/<strategy>.jsonl after every request, so a crash loses at most
the request in flight. Per-strategy aggregates land in
results/<strategy>_summary.json.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

# This VM's stock no_proxy contains bracketed IPv6 entries that crash
# httpx's URL parser, which would send localhost calls to the egress
# proxy or fail outright. Sanitise before any HTTP happens; child
# processes (backends, router, LiteLLM) inherit the clean value.
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "bench"))
sys.path.insert(0, str(PROJECT_DIR / "router"))

from workload import WORKLOAD, evaluate  # noqa: E402

SMALL_MODEL = (
    PROJECT_DIR.parent / "servecore" / "models" / "Qwen2.5-0.5B-Instruct"
)
LARGE_MODEL = (
    PROJECT_DIR.parent / "cachelab" / "models" / "Qwen2.5-1.5B-Instruct"
)
RESULTS = PROJECT_DIR / "results"
LOGS = PROJECT_DIR / "logs"
ROUTER = "http://127.0.0.1:8100"

STRATEGIES = {
    "all_small": "small",
    "all_large": "large",
    "routed": "routewise-auto",
}


def wait_health(url: str, timeout_s: int = 900) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if httpx.get(f"{url}/health", timeout=5,
                         trust_env=False).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError(f"no health from {url} within {timeout_s}s")


def start_stack(backends: tuple[str, ...] = ("small", "large")) -> list[subprocess.Popen]:
    """Start only the requested backends plus the router.

    Running one backend at a time halves the resident weights, which
    matters on this shared swapless VM: an idle model's pages get
    evicted under foreign cache pressure and the next request pays a
    page-fault storm.
    """
    procs = []
    all_specs = [
        ("small", "backend_small", PROJECT_DIR / "router" / "backend_server.py",
         ["--name", "small", "--model-path", str(SMALL_MODEL),
          "--port", "8101"], "http://127.0.0.1:8101"),
        ("large", "backend_large", PROJECT_DIR / "router" / "backend_server.py",
         ["--name", "large", "--model-path", str(LARGE_MODEL),
          "--port", "8102"], "http://127.0.0.1:8102"),
    ]
    for backend, name, script, extra, url in all_specs:
        if backend not in backends:
            continue
        log = open(LOGS / f"{name}.log", "w")
        procs.append(subprocess.Popen(
            [sys.executable, str(script), *extra],
            stdout=log, stderr=subprocess.STDOUT,
        ))
        wait_health(url)
    log = open(LOGS / "router.log", "w")
    procs.append(subprocess.Popen(
        [sys.executable, str(PROJECT_DIR / "router" / "main.py")],
        stdout=log, stderr=subprocess.STDOUT,
        cwd=str(PROJECT_DIR / "router"),
    ))
    wait_health(ROUTER, timeout_s=900)
    return procs


def post(payload: dict, timeout: float = 3600) -> httpx.Response:
    return httpx.post(
        f"{ROUTER}/v1/chat/completions", json=payload,
        timeout=httpx.Timeout(timeout, connect=60),
        trust_env=False,
    )


def record_error(phase: str, exc: Exception) -> None:
    path = RESULTS / "errors.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    rows.append({"phase": phase, "error": f"{type(exc).__name__}: {exc}",
                 "ts": time.time()})
    path.write_text(json.dumps(rows, indent=2))
    print(f"ERROR in {phase}: {exc}", flush=True)


def smoke_tests() -> dict:
    out: dict = {}

    r = post({"model": "routewise-auto", "quality_tier": "balanced",
              "messages": [{"role": "user",
                            "content": "What is the capital of France?"}]})
    body = r.json()
    out["factual_routes_to"] = body.get("routewise", {}).get("backend")
    out["factual_answer"] = body["choices"][0]["message"]["content"][:80]

    r = post({"model": "routewise-auto", "quality_tier": "balanced",
              "max_tokens": 32,
              "messages": [{"role": "user",
                            "content": "Write a Python function that "
                                       "returns the nth Fibonacci number."}]})
    out["code_routes_to"] = r.json().get("routewise", {}).get("backend")

    # Budget probe 1: economy tier with a long prompt must truncate input
    # to the 256-token cap and cap output at 64.
    long_prompt = ("Summarize the following text.\n\n" +
                   ("The history of the steam engine spans many decades "
                    "of incremental improvements by many inventors. " * 60))
    r = post({"model": "routewise-auto", "quality_tier": "economy",
              "messages": [{"role": "user", "content": long_prompt}]})
    body = r.json()
    out["economy_probe_actions"] = body.get("routewise", {}).get("actions")
    out["economy_probe_usage"] = body.get("usage")

    # Budget probe 2: a prompt beyond the 4096-token hard limit is
    # rejected with HTTP 400 and never reaches a backend.
    huge_prompt = "word " * 6000
    r = post({"model": "routewise-auto", "quality_tier": "premium",
              "messages": [{"role": "user", "content": huge_prompt}]})
    out["over_limit_status"] = r.status_code

    (RESULTS / "smoke.json").write_text(json.dumps(out, indent=2))
    print("smoke:", json.dumps(out)[:400], flush=True)
    return out


def litellm_interop() -> dict:
    """Send three routed requests through the LiteLLM client library
    (OpenAI-compatible mode) to prove a LiteLLM user can sit in front of
    RouteWise unchanged."""
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    import litellm

    results = []
    for prompt in (
        "What is the largest ocean on Earth?",
        "Write Python code to check whether a string is a palindrome.",
        "Sara has 3 boxes with 12 pencils each. She gives away 9 "
        "pencils. How many pencils does she have left?",
    ):
        started = time.perf_counter()
        resp = litellm.completion(
            model="openai/routewise-auto",
            api_base=f"{ROUTER}/v1",
            api_key="sk-routewise-local",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=32,
            timeout=3600,
            extra_body={"quality_tier": "balanced"},
        )
        results.append({
            "prompt": prompt[:50],
            "latency_s": round(time.perf_counter() - started, 3),
            "answer": resp.choices[0].message.content[:80],
            "usage": {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
            },
        })
    out = {"client": "litellm.completion (OpenAI-compatible mode)",
           "requests": results}
    (RESULTS / "litellm_interop.json").write_text(json.dumps(out, indent=2))
    print("litellm interop ok:", len(results), "requests", flush=True)
    return out


def run_strategy(name: str, model_field: str,
                 max_new: int | None = None) -> dict:
    jsonl_path = RESULTS / f"{name}.jsonl"
    done_ids = set()
    if jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            if line:
                done_ids.add(json.loads(line)["item_id"])

    # warmup, not recorded: absorbs first-call lazy costs per backend
    warm = [("small", "What is the capital of France?"),
            ("large", "Write a Python function that adds two numbers.")]
    for backend, prompt in warm:
        if name == "all_small" and backend != "small":
            continue
        if name == "all_large" and backend != "large":
            continue
        post({"model": backend, "quality_tier": "economy",
              "messages": [{"role": "user", "content": prompt}]})

    rows = []
    if jsonl_path.exists():
        rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l]
    new_this_run = 0
    for item in WORKLOAD:
        if item["id"] in done_ids:
            continue
        if max_new is not None and new_this_run >= max_new:
            break
        started = time.perf_counter()
        try:
            resp = post({"model": model_field, "quality_tier": item["tier"],
                         "messages": item["messages"]})
        except Exception as exc:
            # Do not write a row: a later resume retries this item.
            record_error(f"{name}:{item['id']}", exc)
            continue
        if resp.status_code >= 500:
            # Backend died mid-run (this VM OOM-kills them). Same rule:
            # no row, a later resume retries the item.
            record_error(f"{name}:{item['id']}",
                         RuntimeError(f"HTTP {resp.status_code}"))
            continue
        latency = time.perf_counter() - started
        body = resp.json()
        text = body["choices"][0]["message"]["content"] if resp.status_code == 200 else ""
        row = {
            "item_id": item["id"],
            "category": item["category"],
            "tier": item["tier"],
            "status": resp.status_code,
            "backend": body.get("routewise", {}).get("backend"),
            "route_reason": body.get("routewise", {}).get("route_reason"),
            "actions": body.get("routewise", {}).get("actions"),
            "latency_s": round(latency, 3),
            "backend_seconds": body.get("routewise_backend_seconds"),
            "usage": body.get("usage"),
            "quality_pass": evaluate(item, text) if resp.status_code == 200 else False,
            "output_preview": text[:160],
        }
        with jsonl_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        rows.append(row)
        new_this_run += 1
        print(f"[{name}] {item['id']} backend={row['backend']} "
              f"lat={row['latency_s']}s pass={row['quality_pass']}", flush=True)
    # Sum of per-request end-to-end latencies: for a sequential harness
    # this is the honest wall time of the measured requests, and unlike a
    # single start/stop timestamp it survives a crash-and-resume.
    wall = round(sum(r["latency_s"] for r in rows), 2)

    latencies = [r["latency_s"] for r in rows]
    per_category: dict[str, dict] = {}
    for r in rows:
        c = per_category.setdefault(r["category"], {"n": 0, "pass": 0})
        c["n"] += 1
        c["pass"] += int(r["quality_pass"])
    summary = {
        "strategy": name,
        "model_field": model_field,
        "complete": len(rows) == len(WORKLOAD),
        "requests": len(rows),
        "wall_seconds_measured_requests_only": wall,
        "latency_mean_s": round(statistics.mean(latencies), 3),
        "latency_p95_s": round(sorted(latencies)[max(0, int(0.95 * len(latencies)) - 1)], 3),
        "latency_max_s": round(max(latencies), 3),
        "total_prompt_tokens": sum(r["usage"]["prompt_tokens"] for r in rows),
        "total_completion_tokens": sum(r["usage"]["completion_tokens"] for r in rows),
        "quality_pass": sum(int(r["quality_pass"]) for r in rows),
        "quality_total": len(rows),
        "quality_by_category": per_category,
        "backend_counts": {
            b: sum(1 for r in rows if r["backend"] == b)
            for b in ("small", "large")
        },
    }
    (RESULTS / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{name}] summary:", json.dumps(summary), flush=True)
    return summary


def write_manifest() -> None:
    import platform

    import torch
    import transformers

    manifest = {
        "started_ts": time.time(),
        "platform": platform.platform(),
        "cpu": platform.processor() or "x86_64",
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "load_average_at_start": list(os.getloadavg()),
        "micro_bench": json.loads(
            (RESULTS / "micro_bench.json").read_text()
        ) if (RESULTS / "micro_bench.json").exists() else None,
        "note": (
            "This VM is shared; load average during the run is recorded "
            "because it dominates single-request latency here."
        ),
    }
    (RESULTS / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


STRATEGY_BACKENDS = {
    "all_small": ("small",),
    "all_large": ("large",),
    "routed": ("small", "large"),
}


def main() -> None:
    """Two modes:

    Full run (no arguments): prep + all three strategies in one go.
    Only sensible on a quiet machine.

    Chunked run (what this shared VM needs; long-lived processes here
    get reaped without warning, so state lives in the result files and
    every chunk is resumable):
        --phase prep                      smoke + interop, skipping any
                                          whose result file already exists
        --phase strategy --strategy NAME --max-new N
                                          at most N new workload items
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prep", "strategy"))
    parser.add_argument("--strategy", choices=tuple(STRATEGIES))
    parser.add_argument("--max-new", type=int)
    cli = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    if not (RESULTS / "run_manifest.json").exists():
        write_manifest()

    if cli.phase == "strategy":
        procs = start_stack(STRATEGY_BACKENDS[cli.strategy])
        try:
            run_strategy(cli.strategy, STRATEGIES[cli.strategy],
                         max_new=cli.max_new)
        finally:
            for p in procs:
                p.terminate()
            print("stack stopped", flush=True)
        return

    backends = ("small", "large") if cli.phase == "prep" else ("small", "large")
    procs = start_stack(backends)
    try:
        if not (RESULTS / "smoke.json").exists():
            try:
                smoke_tests()
            except Exception as exc:
                record_error("smoke", exc)
        if not (RESULTS / "litellm_interop.json").exists():
            try:
                litellm_interop()
            except Exception as exc:
                record_error("litellm_interop", exc)
        if cli.phase is None:
            for name, model_field in STRATEGIES.items():
                try:
                    run_strategy(name, model_field)
                except Exception as exc:
                    record_error(f"strategy:{name}", exc)
    finally:
        for p in procs:
            p.terminate()
        print("stack stopped", flush=True)


if __name__ == "__main__":
    main()
