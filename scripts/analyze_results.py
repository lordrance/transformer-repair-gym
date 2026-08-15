"""Generate every number the final report quotes, from raw artifacts only.

Nothing here is hand-entered. Difficulty is keyed on the independent FULL_FIX
label, never on reward (guardrail G1/G3).

Usage:  python scripts/analyze_results.py [audit.json ...]
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# deepseek-v4-flash, USD per 1M tokens (input cache-miss / cache-hit / output)
PRICE_FLASH = (0.14, 0.0028, 0.28)
PRICE_PRO = (0.435, 0.003625, 0.87)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Reported because n=4 per task and a bare rate would
    imply more precision than 4 samples can carry."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def analyse(audit_path: Path, label: str, prices=PRICE_FLASH) -> dict:
    rows = json.loads(audit_path.read_text(encoding="utf-8"))
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)

    print(f"\n{'=' * 84}\n{label}   n={len(rows)}\n{'=' * 84}")
    print(
        f"{'task':28s} {'FULL':>5s} {'SEM':>4s} {'hard':>5s} {'naive':>6s} "
        f"{'loc':>4s} {'turns':>6s} {'class':>12s}"
    )

    per_task = {}
    for task, rs in sorted(by_task.items()):
        n = len(rs)
        full = sum(1 for r in rs if r["label"] == "FULL_FIX")
        sem = sum(1 for r in rs if r["label"] in ("FULL_FIX", "SEMANTIC_FIX"))
        hard = sum(1 for r in rs if r["hardened_reward"] >= 1.0)
        naive = sum(1 for r in rs if r["naive_reward"] >= 1.0)
        loc = sum(1 for r in rs if r["located_correctly"])
        turns = statistics.mean(r["turns"] for r in rs)

        cls = "TOO_EASY" if full == n else ("TOO_HARD" if full == 0 else "PROMISING")
        per_task[task] = {
            "n": n, "full_fix": full, "semantic_fix": sem, "hardened_pass": hard,
            "naive_pass": naive, "located_correctly": loc,
            "mean_turns": round(turns, 1), "class": cls,
            "full_fix_rate": full / n,
            "full_fix_wilson95": [round(x, 3) for x in wilson(full, n)],
            "reward_pattern": [int(r["hardened_reward"]) for r in rs],
            "label_pattern": [r["label"] for r in rs],
        }
        print(
            f"{task:28s} {full:2d}/{n:<2d} {sem:2d}/{n:<1d} {hard:2d}/{n:<2d} "
            f"{naive:2d}/{n:<3d} {loc:2d}/{n:<1d} {turns:6.1f} {cls:>12s}"
        )

    labels = Counter(r["label"] for r in rows)
    n = len(rows)
    full_rows = [r for r in rows if r["label"] == "FULL_FIX"]
    genuine = [r for r in rows if r["label"] in ("FULL_FIX", "SEMANTIC_FIX")]

    naive_pass = [r for r in rows if r["naive_reward"] >= 1.0]
    hard_pass = [r for r in rows if r["hardened_reward"] >= 1.0]

    naive_fp = [r for r in naive_pass if r not in genuine]
    hard_fp = [r for r in hard_pass if r not in genuine]
    hard_fp_full = [r for r in hard_pass if r["label"] != "FULL_FIX"]
    hard_fn = [r for r in full_rows if r["hardened_reward"] < 1.0]
    disagree_reward = [
        r for r in rows if (r["hardened_reward"] >= 1.0) != (r["label"] == "FULL_FIX")
    ]

    # cost
    p_miss, p_hit, p_out = prices
    pt = sum(r["prompt_tokens"] for r in rows)
    ct = sum(r["completion_tokens"] for r in rows)
    rt = sum(r["reasoning_tokens"] for r in rows)
    cost = pt / 1e6 * p_miss + ct / 1e6 * p_out

    summary = {
        "label": label,
        "n_trajectories": n,
        "labels": dict(labels),
        "full_success_rate": len(full_rows) / n,
        "full_success_wilson95": [round(x, 3) for x in wilson(len(full_rows), n)],
        "semantic_success_rate": len(genuine) / n,
        "naive_pass_rate": len(naive_pass) / n,
        "hardened_pass_rate": len(hard_pass) / n,
        "naive_FPR": len(naive_fp) / max(1, len(naive_pass)),
        "hardened_FPR": len(hard_fp) / max(1, len(hard_pass)),
        "hardened_FPR_vs_full_success": len(hard_fp_full) / max(1, len(hard_pass)),
        "hardened_FNR": len(hard_fn) / max(1, len(full_rows)),
        "reward_vs_independent_disagreements": len(disagree_reward),
        "natural_reward_hack": labels.get("REWARD_HACK", 0),
        "natural_reward_tampering": labels.get("REWARD_TAMPERING", 0),
        "infra_failures": labels.get("INFRA_FAILURE", 0),
        "invalid": labels.get("INVALID", 0),
        "localization_rate": sum(1 for r in rows if r["located_correctly"]) / n,
        "mean_turns": round(statistics.mean(r["turns"] for r in rows), 2),
        "episodes_ending_by_budget": sum(
            1 for r in rows if "budget" in r["end_reason"]
        ),
        "episodes_submitting": sum(1 for r in rows if r["submitted"]),
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "reasoning_tokens": rt,
        "reasoning_share_of_completion": round(rt / max(1, ct), 3),
        "cost_usd": round(cost, 4),
        "cost_usd_per_trajectory": round(cost / n, 5),
        "difficulty_distribution": Counter(v["class"] for v in per_task.values()),
        "per_task": per_task,
    }

    print("\n-- verifier behaviour on real rollouts --")
    for k in ("naive_pass_rate", "hardened_pass_rate", "full_success_rate",
              "semantic_success_rate", "naive_FPR", "hardened_FPR",
              "hardened_FPR_vs_full_success", "hardened_FNR",
              "reward_vs_independent_disagreements", "natural_reward_hack",
              "natural_reward_tampering", "infra_failures", "invalid"):
        print(f"  {k:38s} {summary[k]}")
    print("\n-- trajectory / cost --")
    for k in ("localization_rate", "mean_turns", "episodes_ending_by_budget",
              "episodes_submitting", "prompt_tokens", "completion_tokens",
              "reasoning_share_of_completion", "cost_usd", "cost_usd_per_trajectory"):
        print(f"  {k:38s} {summary[k]}")
    print(f"\n  difficulty_distribution              {dict(summary['difficulty_distribution'])}")

    summary["difficulty_distribution"] = dict(summary["difficulty_distribution"])
    return summary


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] or [REPO_ROOT / "artifacts" / "tier_m_audit.json"]
    out = []
    for p in paths:
        if not p.exists():
            print(f"skip missing {p}")
            continue
        label = p.stem.replace("_audit", "")
        prices = PRICE_PRO if "pro" in p.stem else PRICE_FLASH
        out.append(analyse(p, label, prices))
    dest = REPO_ROOT / "artifacts" / "analysis_summary.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
