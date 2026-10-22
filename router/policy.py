"""RouteWise routing policy: request classification, difficulty scoring,
per-tier token budgets, and the routing decision.

Pure functions only (no model, no I/O) so the policy can be unit tested
without loading any weights. Token counting is injected by the caller.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Task classification
# ---------------------------------------------------------------------------

CODE_KEYWORDS = (
    "code", "function", "python", "javascript", "typescript", "debug",
    "bug", "implement", "program", "script", "regex", "sql query",
    "stack trace", "compile", "algorithm in",
)
REASONING_KEYWORDS = (
    "step by step", "reason", "prove", "calculate", "how many",
    "if every", "if all", "logic", "puzzle", "probability",
    "what is the probability", "solve", "derive", "explain why",
)
SUMMARY_KEYWORDS = (
    "summarize", "summarise", "summary", "tl;dr", "tldr",
    "condense", "shorten this", "key points of",
)

TASK_TYPES = ("factual", "summary", "code", "reasoning")


def classify_task(text: str, prompt_tokens: int) -> str:
    """Classify the request into one of TASK_TYPES from the user text.

    Order matters: explicit summarisation and code requests win over the
    generic reasoning keywords because both often contain words like
    'calculate' or 'explain'.
    """
    lowered = text.lower()
    if any(k in lowered for k in SUMMARY_KEYWORDS):
        return "summary"
    if any(k in lowered for k in CODE_KEYWORDS):
        return "code"
    if any(k in lowered for k in REASONING_KEYWORDS):
        return "reasoning"
    return "factual"


def difficulty_score(task_type: str, prompt_tokens: int) -> float:
    """A 0..1 heuristic difficulty score for the request.

    Built from the task type (code and reasoning are where the small model
    loses most quality; see the QuantLab spot checks and the benchmark in
    bench/) plus prompt length (long prompts stress a small context and
    small-model attention).
    """
    base = {
        "factual": 0.15,
        "summary": 0.35,
        "code": 0.70,
        "reasoning": 0.75,
    }[task_type]
    length_component = min(prompt_tokens / 800.0, 1.0) * 0.3
    return round(min(base + length_component, 1.0), 3)


# ---------------------------------------------------------------------------
# Token budgets per quality tier
# ---------------------------------------------------------------------------

TIER_BUDGETS = {
    # tier: (max prompt tokens accepted after truncation, max output tokens)
    # Output caps are sized for CPU serving: on this VM the large backend
    # decodes at well under 1 tok/s under load, so a 320-token premium
    # answer would take many minutes. The caps keep worst-case latency
    # bounded, which is what a budget is for.
    "economy": {"input_cap": 256, "output_cap": 48},
    "balanced": {"input_cap": 1024, "output_cap": 96},
    "premium": {"input_cap": 2048, "output_cap": 128},
}

# Hard context safety limit for these two backends on this machine.
HARD_INPUT_LIMIT = 4096

DEFAULT_OUTPUT_BY_TASK = {
    "factual": 32,
    "summary": 64,
    "code": 96,
    "reasoning": 96,
}


def plan_budget(tier: str, task_type: str, prompt_tokens: int,
                requested_max_tokens: int | None) -> dict:
    """Decide the enforced budget for one request.

    Returns a dict with the caps and the enforcement actions to apply:
    - 'truncate_input_to': token count the prompt must be cut to (or None)
    - 'max_new_tokens': the output cap actually granted
    - 'reject': True when the prompt is beyond the hard context limit
    """
    budget = TIER_BUDGETS[tier]
    actions: list[str] = []

    if prompt_tokens > HARD_INPUT_LIMIT:
        return {
            "input_cap": budget["input_cap"],
            "output_cap": budget["output_cap"],
            "truncate_input_to": None,
            "max_new_tokens": 0,
            "reject": True,
            "actions": ["rejected_over_hard_limit"],
        }

    truncate_to = None
    if prompt_tokens > budget["input_cap"]:
        truncate_to = budget["input_cap"]
        actions.append("input_truncated_to_tier_cap")

    wanted = requested_max_tokens or DEFAULT_OUTPUT_BY_TASK[task_type]
    max_new = min(wanted, budget["output_cap"])
    if max_new < wanted:
        actions.append("output_capped_to_tier_budget")

    return {
        "input_cap": budget["input_cap"],
        "output_cap": budget["output_cap"],
        "truncate_input_to": truncate_to,
        "max_new_tokens": max_new,
        "reject": False,
        "actions": actions,
    }


# ---------------------------------------------------------------------------
# Routing decision
# ---------------------------------------------------------------------------

def choose_backend(tier: str, task_type: str, prompt_tokens: int,
                   score: float) -> tuple[str, str]:
    """Return (backend_name, reason). Backends: 'small' (Qwen2.5-0.5B),
    'large' (Qwen2.5-1.5B).

    Policy:
    - economy: everything runs on the small model. That is the point of
      the tier; the budget caps keep the cost bounded.
    - premium: everything runs on the large model, except trivially short
      factual questions, which are routed down because the large model
      adds latency there without adding quality (verified in the bench).
    - balanced: code and reasoning go to the large model, long summaries
      go to the large model, everything else stays on the small model.
      A difficulty score at or above 0.6 also escalates to large.
    """
    if tier == "economy":
        return "small", "economy tier: small model by policy"

    if tier == "premium":
        if task_type == "factual" and prompt_tokens <= 40:
            return "small", "premium route-down: trivial factual question"
        return "large", "premium tier: large model by policy"

    # balanced
    if task_type in ("code", "reasoning"):
        return "large", f"balanced: {task_type} escalates to large"
    if task_type == "summary" and prompt_tokens > 300:
        return "large", "balanced: long summary escalates to large"
    if score >= 0.6:
        return "large", "balanced: difficulty score >= 0.6 escalates"
    return "small", "balanced: small model is enough"
