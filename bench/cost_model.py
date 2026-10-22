"""RouteWise unit economics, computed only from measured benchmark data.

Every input number comes from results/<strategy>_summary.json and the
per-request results/<strategy>.jsonl files produced by run_bench.py.
The only external inputs are the two labelled price assumptions below.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_DIR / "results"

ASSUMPTIONS = {
    "instance": "AWS c7i-flex.large (2 vCPU, 4 GiB), On-Demand, us-east-1",
    "instance_price_usd_per_hour": 0.0848,
    "instance_price_source": (
        "RunsOn EC2 finder listing for c7i-flex.large, prices fetched "
        "2026-10-20: on-demand $0.0848/hr in us-east-1"
    ),
    "hardware_equivalence": (
        "This VM (2 x86-64 vCPUs, 7.75 GiB RAM, no GPU) is treated as "
        "equivalent to that instance class for cost purposes."
    ),
    "serving_model": (
        "Sequential single-request serving, no batching. A strategy "
        "costs instance_price x its measured wall time for the workload's "
        "requests. Model load time and idle time are excluded. Both "
        "backends share the one instance in the routed deployment."
    ),
    "gpu_reference_only": (
        "An AWS g5.xlarge (1x A10G) lists at about $1.006/hr On-Demand "
        "(Thunder Compute and Wring EC2 guides, Oct 2026). No GPU "
        "throughput was measured on this VM, so no GPU per-token cost "
        "is derived; the figure is context only."
    ),
}


def per_backend_decode_rates() -> dict:
    """Measured decode throughput per backend from the benchmark rows:
    sum of completion tokens / sum of backend generation seconds."""
    totals: dict[str, dict[str, float]] = {}
    for strategy in ("all_small", "all_large", "routed"):
        path = RESULTS / f"{strategy}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line:
                continue
            row = json.loads(line)
            backend = row["backend"]
            if backend is None or not row.get("backend_seconds"):
                continue
            slot = totals.setdefault(backend, {"tokens": 0, "seconds": 0.0})
            slot["tokens"] += row["usage"]["completion_tokens"]
            slot["seconds"] += row["backend_seconds"]
    rates = {}
    price = ASSUMPTIONS["instance_price_usd_per_hour"]
    for backend, slot in totals.items():
        tok_per_s = slot["tokens"] / slot["seconds"]
        rates[backend] = {
            "completion_tokens": int(slot["tokens"]),
            "backend_seconds": round(slot["seconds"], 2),
            "note": (
                "backend_seconds covers prefill plus decode, so this is "
                "an effective end-to-end rate, not a pure decode rate."
            ),
            "effective_tokens_per_second_incl_prefill": round(tok_per_s, 3),
            "usd_per_1m_completion_tokens": round(
                price / (tok_per_s * 3600) * 1_000_000, 2
            ),
        }
    return rates


def main() -> None:
    price = ASSUMPTIONS["instance_price_usd_per_hour"]
    strategies = {}
    for name in ("all_small", "all_large", "routed"):
        path = RESULTS / f"{name}_summary.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text())
        if not summary.get("complete", False):
            continue
        wall = summary["wall_seconds_measured_requests_only"]
        cost = price * wall / 3600
        n = summary["requests"]
        total_tokens = (
            summary["total_prompt_tokens"] + summary["total_completion_tokens"]
        )
        strategies[name] = {
            "requests": n,
            "wall_seconds": wall,
            "cost_for_workload_usd": round(cost, 5),
            "cost_per_1000_requests_usd": round(cost / n * 1000, 2),
            "cost_per_1m_total_tokens_usd": round(
                cost / total_tokens * 1_000_000, 2
            ),
            "quality_pass": summary["quality_pass"],
            "quality_total": summary["quality_total"],
            "cost_per_quality_passed_request_usd": round(
                cost / max(summary["quality_pass"], 1), 5
            ),
        }

    out = {
        "assumptions": ASSUMPTIONS,
        "per_backend_decode_economics": per_backend_decode_rates(),
        "strategies": strategies,
    }
    if "routed" in strategies:
        routed = strategies["routed"]["cost_per_1000_requests_usd"]
        out["savings"] = {
            "routed_vs_all_large_cost_per_1k_pct": round(
                (1 - routed / strategies["all_large"]["cost_per_1000_requests_usd"])
                * 100,
                1,
            ),
            "routed_vs_all_small_cost_per_1k_pct": round(
                (1 - routed / strategies["all_small"]["cost_per_1000_requests_usd"])
                * 100,
                1,
            ),
        }
    (RESULTS / "cost_model.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
