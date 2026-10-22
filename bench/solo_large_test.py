"""Decisive experiment: does the large backend generate at micro-bench
rates when it is the ONLY model resident?

Context: with both backends resident, a 96-token code request to the
large backend did not finish within 3600 s, and the backend's RSS sat
at ~1.9 GB of its 3.1 GB of weights, suggesting page-cache thrash on
this shared, swapless VM. This script starts only the large backend,
sends the same request directly to it, and samples RSS and system
pgmajfault while it runs. Result: results/solo_large_test.json.
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
LARGE_MODEL = (
    PROJECT_DIR.parent / "cachelab" / "models" / "Qwen2.5-1.5B-Instruct"
)


def majfaults() -> int:
    for line in Path("/proc/vmstat").read_text().splitlines():
        if line.startswith("pgmajfault"):
            return int(line.split()[1])
    return -1


def rss_kb(pid: int) -> int:
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith("VmRSS"):
            return int(line.split()[1])
    return -1


def main() -> None:
    log = open(PROJECT_DIR / "logs" / "solo_large.log", "w")
    proc = subprocess.Popen(
        [sys.executable, str(PROJECT_DIR / "router" / "backend_server.py"),
         "--name", "large", "--model-path", str(LARGE_MODEL),
         "--port", "8102"],
        stdout=log, stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.time() + 600
        while time.time() < deadline:
            try:
                if httpx.get("http://127.0.0.1:8102/health", timeout=5,
                             trust_env=False).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(3)
        rss_loaded = rss_kb(proc.pid)
        faults_before = majfaults()
        started = time.perf_counter()
        resp = httpx.post(
            "http://127.0.0.1:8102/v1/chat/completions",
            json={"messages": [{"role": "user", "content":
                                 "Write a Python function that returns "
                                 "the nth Fibonacci number."}],
                  "max_tokens": 96},
            timeout=httpx.Timeout(1800, connect=30), trust_env=False,
        )
        seconds = time.perf_counter() - started
        body = resp.json()
        out = {
            "rss_after_load_mb": round(rss_loaded / 1024, 1),
            "rss_after_gen_mb": round(rss_kb(proc.pid) / 1024, 1),
            "pgmajfault_delta_during_gen": majfaults() - faults_before,
            "seconds": round(seconds, 2),
            "usage": body.get("usage"),
            "backend_seconds": body.get("routewise_backend_seconds"),
            "answer_preview": body["choices"][0]["message"]["content"][:120],
        }
        (PROJECT_DIR / "results" / "solo_large_test.json").write_text(
            json.dumps(out, indent=2)
        )
        print(json.dumps(out, indent=2), flush=True)
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
