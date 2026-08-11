"""Phase-0 API baseline: does the environment have a usable dynamic range?

This answers exactly one question -- not "which model is best". We want to know
whether a capable model lands somewhere between 0/5 and 5/5, because a task set
that is always solved or never solved carries no gradient (see the
zero-advantage test in tests/test_minimal_grpo.py).

Usage
-----
    $env:OPENAI_API_KEY = "sk-..."
    .\.venv\Scripts\python.exe scripts/run_api_baseline.py --model gpt-5.4 -r 4

Any OpenAI-compatible endpoint works via --base-url.

Cost: 5 tasks x rollouts, single turn, ~4-6k prompt tokens each. At 4 rollouts
that is 20 completions -- cents, not dollars. Check your provider's pricing
before running with a large -r.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "environments" / "transformer_repair"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-var", default="OPENAI_API_KEY")
    ap.add_argument("-r", "--rollouts", type=int, default=4)
    ap.add_argument("--max-concurrent", type=int, default=5)
    ap.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "api_baseline.json"))
    args = ap.parse_args()

    api_key = os.environ.get(args.api_key_var)
    if not api_key:
        print(f"error: {args.api_key_var} is not set", file=sys.stderr)
        return 2

    from openai import OpenAI

    from transformer_repair import load_environment

    client = OpenAI(api_key=api_key, base_url=args.base_url)

    # `reward_scheme` only decides which signal is returned as THE reward; both
    # are always recorded as metrics, so one run fills in both columns.
    env = load_environment(reward_scheme="hardened")

    results = env.evaluate(
        client=client,
        model=args.model,
        rollouts_per_example=args.rollouts,
        max_concurrent=args.max_concurrent,
    )

    per_task: dict[str, list[dict]] = defaultdict(list)
    for state in results.state:
        task_id = (state.get("info") or {}).get("task_id", "?")
        metrics = state.get("metrics") or {}
        per_task[task_id].append(
            {
                "naive": metrics.get("naive_reward_metric", metrics.get("naive_reward", 0.0)),
                "hardened": metrics.get(
                    "hardened_reward_metric", metrics.get("hardened_reward", 0.0)
                ),
                "exploit_gap": metrics.get("exploit_gap_metric", 0.0),
                "trgym": state.get("trgym", {}),
            }
        )

    rows = []
    for task_id, runs in sorted(per_task.items()):
        rows.append(
            {
                "task_id": task_id,
                "n": len(runs),
                "naive_pass_at_1": statistics.mean(r["naive"] for r in runs),
                "hardened_pass_at_1": statistics.mean(r["hardened"] for r in runs),
                "exploit_gap_rate": statistics.mean(r["exploit_gap"] for r in runs),
                "gates_fired": sorted(
                    {g for r in runs for g in (r["trgym"].get("gates") or [])}
                ),
                "hidden_checks_failed": sorted(
                    {c for r in runs for c in (r["trgym"].get("hidden_failed") or [])}
                ),
            }
        )

    overall_hardened = statistics.mean(r["hardened_pass_at_1"] for r in rows) if rows else 0.0
    overall_naive = statistics.mean(r["naive_pass_at_1"] for r in rows) if rows else 0.0

    print(f"\n{'task':38s} {'naive':>7s} {'hardened':>9s} {'gap':>6s}")
    for r in rows:
        print(
            f"{r['task_id']:38s} {r['naive_pass_at_1']:7.2f} "
            f"{r['hardened_pass_at_1']:9.2f} {r['exploit_gap_rate']:6.2f}"
        )
    print(f"{'OVERALL':38s} {overall_naive:7.2f} {overall_hardened:9.2f}")

    if overall_hardened >= 0.95:
        verdict = "TOO EASY - the task set will not produce advantage signal"
    elif overall_hardened <= 0.02:
        verdict = "TOO HARD - no positive rollouts to learn from"
    else:
        verdict = "USABLE dynamic range"
    print(f"\nverdict: {verdict}")

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "model": args.model,
                "rollouts_per_example": args.rollouts,
                "per_task": rows,
                "overall_naive_pass_at_1": overall_naive,
                "overall_hardened_pass_at_1": overall_hardened,
                "verdict": verdict,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
