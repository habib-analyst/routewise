"""RouteWise charts, drawn only from results/*.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS = PROJECT_DIR / "results"
CHARTS = PROJECT_DIR / "charts"

STRATEGIES = ["all_small", "all_large", "routed"]
LABELS = ["all to 0.5B", "all to 1.5B", "routed"]
COLORS = ["#8ecae6", "#ffb703", "#219ebc"]


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def chart_latency() -> None:
    summaries = [load(f"{s}_summary.json") for s in STRATEGIES]
    means = [s["latency_mean_s"] for s in summaries]
    p95s = [s["latency_p95_s"] for s in summaries]
    x = range(len(STRATEGIES))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar([i - width / 2 for i in x], means, width, label="mean latency", color="#219ebc")
    ax.bar([i + width / 2 for i in x], p95s, width, label="p95 latency", color="#ffb703")
    ax.set_xticks(list(x))
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("seconds per request")
    ax.set_title("RouteWise: latency by strategy (20 requests, CPU only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS / "latency_by_strategy.png", dpi=150)
    plt.close(fig)


def chart_cost_quality() -> None:
    costs = load("cost_model.json")["strategies"]
    summaries = {s: load(f"{s}_summary.json") for s in STRATEGIES}
    per_1k = [costs[s]["cost_per_1000_requests_usd"] for s in STRATEGIES]
    quality = [
        summaries[s]["quality_pass"] / summaries[s]["quality_total"] * 100
        for s in STRATEGIES
    ]
    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.bar(range(len(STRATEGIES)), per_1k, 0.5, color=COLORS)
    ax1.set_xticks(range(len(STRATEGIES)))
    ax1.set_xticklabels(LABELS)
    ax1.set_ylabel("USD per 1000 requests")
    ax2 = ax1.twinx()
    ax2.plot(range(len(STRATEGIES)), quality, color="#d90429", marker="o",
             linewidth=2, label="quality check pass rate")
    ax2.set_ylabel("quality checks passed (%)")
    ax2.set_ylim(0, 110)
    ax1.set_title("RouteWise: cost per 1000 requests vs quality pass rate")
    fig.tight_layout()
    fig.savefig(CHARTS / "cost_vs_quality.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    chart_latency()
    chart_cost_quality()
    print("charts written to", CHARTS)
