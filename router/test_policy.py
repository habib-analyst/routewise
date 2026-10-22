"""Unit tests for the RouteWise policy (no models needed).

Run: python3 -m pytest test_policy.py -q   (or plain python3 test_policy.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from policy import (  # noqa: E402
    choose_backend,
    classify_task,
    difficulty_score,
    plan_budget,
)


def test_classification():
    assert classify_task("Summarize the following text", 100) == "summary"
    assert classify_task("Write a Python function to sort a list", 20) == "code"
    assert classify_task("Debug this bug in my script", 20) == "code"
    assert (
        classify_task("A coin is flipped. What is the probability of heads?", 30)
        == "reasoning"
    )
    assert classify_task("What is the capital of France?", 10) == "factual"


def test_difficulty_ordering():
    assert difficulty_score("reasoning", 50) > difficulty_score("factual", 50)
    assert difficulty_score("factual", 800) > difficulty_score("factual", 10)
    assert 0.0 <= difficulty_score("code", 5000) <= 1.0


def test_budget_caps():
    plan = plan_budget("economy", "factual", 100, None)
    assert plan["max_new_tokens"] == 32  # task default under the 48 cap
    plan = plan_budget("economy", "code", 100, None)
    assert plan["max_new_tokens"] == 48  # task default 96 capped to tier
    assert "output_capped_to_tier_budget" in plan["actions"]

    plan = plan_budget("balanced", "factual", 2000, None)
    assert plan["truncate_input_to"] == 1024
    assert not plan["reject"]

    plan = plan_budget("premium", "factual", 5000, None)
    assert plan["reject"] is True

    plan = plan_budget("balanced", "code", 100, 999)
    assert plan["max_new_tokens"] == 96  # balanced output cap


def test_routing():
    assert choose_backend("economy", "reasoning", 100, 0.8)[0] == "small"
    assert choose_backend("premium", "code", 100, 0.8)[0] == "large"
    assert choose_backend("premium", "factual", 12, 0.2)[0] == "small"
    assert choose_backend("balanced", "code", 30, 0.7)[0] == "large"
    assert choose_backend("balanced", "reasoning", 30, 0.8)[0] == "large"
    assert choose_backend("balanced", "factual", 12, 0.2)[0] == "small"
    assert choose_backend("balanced", "summary", 150, 0.4)[0] == "small"
    assert choose_backend("balanced", "summary", 500, 0.5)[0] == "large"


if __name__ == "__main__":
    test_classification()
    test_difficulty_ordering()
    test_budget_caps()
    test_routing()
    print("all policy tests passed")
