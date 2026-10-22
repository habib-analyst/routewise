"""Self-healing stack check for the RouteWise benchmark.

Health-checks the two backends and the router; starts whichever is
missing (detached, logging to logs/) and waits for health.

Processes on this VM die silently: the host restarts, and the kernel
OOM killer fires when tenants spike (measured: oom_kill counter
climbing while a backend was pinning its weights). ensure() therefore
watches the child it started: if the child exits before becoming
healthy, it retries, and before starting the large backend it waits
for a memory window (MemAvailable) big enough for the pinning spike.

Usage: python3 ensure_stack.py [small|large|both]   (default both)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

PROJECT_DIR = Path(__file__).resolve().parent.parent
SMALL_MODEL = PROJECT_DIR.parent / "servecore" / "models" / "Qwen2.5-0.5B-Instruct"
LARGE_MODEL = PROJECT_DIR.parent / "cachelab" / "models" / "Qwen2.5-1.5B-Instruct"
LOGS = PROJECT_DIR / "logs"

SERVICES = {
    "small": ("http://127.0.0.1:8101", "backend_small.log",
              [sys.executable, str(PROJECT_DIR / "router" / "backend_server.py"),
               "--name", "small", "--model-path", str(SMALL_MODEL),
               "--port", "8101"], None),
    "large": ("http://127.0.0.1:8102", "backend_large.log",
              [sys.executable, str(PROJECT_DIR / "router" / "backend_server.py"),
               "--name", "large", "--model-path", str(LARGE_MODEL),
               "--port", "8102"], None),
    "router": ("http://127.0.0.1:8100", "router.log",
               [sys.executable, str(PROJECT_DIR / "router" / "main.py")],
               str(PROJECT_DIR / "router")),
}

# MemAvailable (GB) required before attempting a start. The large
# backend peaks while cloning its weights into anonymous memory.
MIN_AVAILABLE_GB = {"small": 1.1, "large": 3.2, "router": 0.6}


def mem_available_gb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) / 1024 / 1024
    return 0.0


def healthy(url: str) -> bool:
    try:
        return httpx.get(f"{url}/health", timeout=10,
                         trust_env=False).status_code == 200
    except Exception:
        return False


def ensure(name: str, attempts: int = 4) -> None:
    url, log_name, cmd, cwd = SERVICES[name]
    if healthy(url):
        print(f"{name}: already up", flush=True)
        return
    for attempt in range(1, attempts + 1):
        # Wait for a memory window before paying the startup spike.
        gate_deadline = time.time() + 900
        while mem_available_gb() < MIN_AVAILABLE_GB[name]:
            if time.time() > gate_deadline:
                raise RuntimeError(
                    f"{name}: MemAvailable stayed below "
                    f"{MIN_AVAILABLE_GB[name]} GB for 900s"
                )
            time.sleep(15)
        log = open(LOGS / log_name, "a")
        env = dict(os.environ)
        # All models and tokenizers are local files. Without this, a
        # from_pretrained call can stall for many minutes on a hub
        # reachability check through this VM's proxy.
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"
        if name == "small" and mem_available_gb() < 2.2:
            # Tight memory: start the small backend unpinned (about
            # half the RSS). The large backend stays pinned; it is the
            # one that stalls when its pages are evicted.
            env["ROUTEWISE_PIN"] = "0"
            print("small: starting unpinned (tight memory)", flush=True)
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                cwd=cwd, start_new_session=True, env=env)
        deadline = time.time() + 900
        while time.time() < deadline:
            if healthy(url):
                print(f"{name}: started and healthy (attempt {attempt})",
                      flush=True)
                return
            if proc.poll() is not None:
                print(f"{name}: child exited (code {proc.returncode}) "
                      f"before health; retrying", flush=True)
                break
            time.sleep(5)
        else:
            proc.terminate()
            print(f"{name}: no health within 900s (attempt {attempt}); "
                  f"retrying", flush=True)
        time.sleep(15)
    raise RuntimeError(f"{name}: could not start after {attempts} attempts")


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    names = ["small", "large"] if which == "both" else [which]
    for name in names:
        ensure(name)
    ensure("router")


if __name__ == "__main__":
    main()
