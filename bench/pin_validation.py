"""Validate the weight-pinning fix under the condition that broke the
routed setup: both backends resident, the large one idle, and foreign
page-cache pressure in between.

Steps: start both backends (pinned build), one small request, churn
~8 GB through the page cache (reads of unrelated large files), wait,
then time a 32-token code request to the large backend and sample its
RSS and the system major-fault delta. Result:
results/pin_validation.json.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1"

PROJECT_DIR = Path(__file__).resolve().parent.parent
SMALL_MODEL = PROJECT_DIR.parent / "servecore" / "models" / "Qwen2.5-0.5B-Instruct"
LARGE_MODEL = PROJECT_DIR.parent / "cachelab" / "models" / "Qwen2.5-1.5B-Instruct"


def majfaults() -> int:
    for line in Path("/proc/vmstat").read_text().splitlines():
        if line.startswith("pgmajfault"):
            return int(line.split()[1])
    return -1


def rss_mb(pid: int) -> float:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS"):
            return int(line.split()[1]) / 1024
    return -1.0


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
    raise RuntimeError(f"no health from {url}")


def post(port: int, prompt: str, max_tokens: int) -> tuple[float, dict]:
    started = time.perf_counter()
    resp = httpx.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json={"messages": [{"role": "user", "content": prompt}],
              "max_tokens": max_tokens},
        timeout=httpx.Timeout(1800, connect=30), trust_env=False,
    )
    return time.perf_counter() - started, resp.json()


def churn_cache() -> None:
    """Push several GB of unrelated file data through the page cache,
    the way other tenants' I/O evicted the model's file-backed pages."""
    candidates = [
        LARGE_MODEL / "model.safetensors",
        SMALL_MODEL / "model.safetensors",
        PROJECT_DIR.parent / "quantlab" / "artifacts" / "gptq.pt",
        PROJECT_DIR.parent / "quantlab" / "artifacts" / "awq.pt",
    ]
    for _ in range(2):
        for path in candidates:
            if path.exists():
                with path.open("rb") as fh:
                    while fh.read(8 * 1024 * 1024):
                        pass


def main() -> None:
    procs = []
    for name, model, port in (("small", SMALL_MODEL, 8101),
                              ("large", LARGE_MODEL, 8102)):
        log = open(PROJECT_DIR / "logs" / f"pin_{name}.log", "w")
        procs.append(subprocess.Popen(
            [sys.executable, str(PROJECT_DIR / "router" / "backend_server.py"),
             "--name", name, "--model-path", str(model), "--port", str(port)],
            stdout=log, stderr=subprocess.STDOUT))
    try:
        wait_health("http://127.0.0.1:8101")
        wait_health("http://127.0.0.1:8102")
        large_pid = procs[1].pid
        out: dict = {"rss_large_after_start_mb": round(rss_mb(large_pid), 1)}

        secs, _ = post(8101, "What is the capital of France?", 16)
        out["small_request_s"] = round(secs, 2)

        churn_cache()
        out["rss_large_after_churn_mb"] = round(rss_mb(large_pid), 1)
        time.sleep(60)
        out["rss_large_after_idle_mb"] = round(rss_mb(large_pid), 1)

        faults_before = majfaults()
        secs, body = post(
            8102,
            "Write a Python function that returns the nth Fibonacci number.",
            32,
        )
        out["large_request_s"] = round(secs, 2)
        out["large_backend_seconds"] = body.get("routewise_backend_seconds")
        out["large_usage"] = body.get("usage")
        out["pgmajfault_delta"] = majfaults() - faults_before
        out["rss_large_after_gen_mb"] = round(rss_mb(large_pid), 1)
        (PROJECT_DIR / "results" / "pin_validation.json").write_text(
            json.dumps(out, indent=2)
        )
        print(json.dumps(out, indent=2), flush=True)
    finally:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
