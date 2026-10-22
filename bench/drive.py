"""Client-only driver: run benchmark phases against an already-running
RouteWise stack (started by run_bench.py or by hand).

Why this exists: on this VM, long-lived launcher processes get reaped
and the host itself restarts, but an orphaned stack that is already up
keeps serving. drive.py never starts or stops processes; it only makes
HTTP calls and appends to the result files, so each invocation is a
short, resumable chunk:

    python3 drive.py interop
    python3 drive.py strategy all_large 5     # at most 5 new items
    python3 drive.py strategy routed 5
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_bench  # noqa: E402  (module import has no side effects beyond env)


def main() -> None:
    import ensure_stack

    what = sys.argv[1]
    if what == "ensure":
        ensure_stack.main()
        return

    # Self-heal: start whichever services are missing before working.
    needed = ("small", "large")
    if what == "strategy":
        needed = run_bench.STRATEGY_BACKENDS[sys.argv[2]]
    if os.environ.get("ROUTEWISE_ONLY_LARGE") == "1":
        # Mirror image: only the large backend is ensured. Used when
        # every remaining routed item is large-bound and the VM cannot
        # hold both backends alongside other tenants.
        needed = ("large",)
    if os.environ.get("ROUTEWISE_SKIP_LARGE") == "1":
        # Memory emergency mode: when other tenants leave no room for
        # the 3.4 GB large backend, run anyway. Items the policy sends
        # to the large backend get a clean 502 from the router, are
        # skipped without a result row, and are retried by a later
        # chunk once the large backend fits again.
        needed = tuple(b for b in needed if b != "large")
    for backend in needed:
        ensure_stack.ensure(backend)
    ensure_stack.ensure("router")

    if what == "interop":
        run_bench.litellm_interop()
    elif what == "smoke":
        run_bench.smoke_tests()
    elif what == "strategy":
        name = sys.argv[2]
        max_new = int(sys.argv[3]) if len(sys.argv) > 3 else None
        run_bench.run_strategy(name, run_bench.STRATEGIES[name],
                               max_new=max_new)
    else:
        raise SystemExit(f"unknown phase: {what}")


if __name__ == "__main__":
    main()
