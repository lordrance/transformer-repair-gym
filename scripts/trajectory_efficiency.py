"""Trajectory-efficiency analysis (guardrail G7, RQ4).

The question this answers is not "how good is the model". It is:

    do harder tasks require more *investigation*, or merely more *tokens*?

Those look identical in a cost report and mean opposite things for environment
design. Investigation is what a repo-level task is supposed to buy; token growth
is just long-context tax.

Usage:  python scripts/trajectory_efficiency.py artifacts/tier_m_primary.jsonl [...]
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

READ_TOOLS = {"read_file", "list_files"}
RUN_TOOLS = {"run_command"}
EDIT_TOOLS = {"apply_patch"}


def profile(rec: dict) -> dict:
    steps = rec["episode"]["steps"]
    tools = Counter(s["tool"] for s in steps)
    reads = sum(v for k, v in tools.items() if k in READ_TOOLS)
    runs = sum(v for k, v in tools.items() if k in RUN_TOOLS)
    patches = sum(v for k, v in tools.items() if k in EDIT_TOOLS)
    failed_patches = sum(
        1 for s in steps if s["tool"] == "apply_patch" and not s["ok"]
    )
    noops = sum(v for k, v in tools.items() if k in {"noop", "bad_arguments", "api_error"})

    # Which commands did it actually choose to run?
    commands = Counter(
        s["args"].get("name", "?") for s in steps if s["tool"] == "run_command"
    )
    # Turn index of the first successful patch: how long localization took.
    first_patch = next(
        (s["turn"] for s in steps if s["tool"] == "apply_patch" and s["ok"]), None
    )
    files_read = {
        s["args"].get("path") for s in steps if s["tool"] == "read_file"
    } - {None}

    return {
        "task": rec["task_id"],
        "episode": rec["episode_id"],
        "turns": rec["episode"]["n_turns"],
        "reads": reads,
        "runs": runs,
        "patches": patches,
        "failed_patches": failed_patches,
        "noops": noops,
        "distinct_files_read": len(files_read),
        "first_successful_patch_turn": first_patch,
        "commands": dict(commands),
        "ran_training": any("training" in c for c in commands),
        "ran_tests": any("test" in c for c in commands),
        "prompt_tokens": rec["usage"]["prompt_tokens"],
        "completion_tokens": rec["usage"]["completion_tokens"],
        "reasoning_tokens": rec["usage"]["reasoning_tokens"],
        "hardened_reward": rec["hardened_reward"],
        "end_reason": rec["episode"]["end_reason"],
        "submitted": rec["episode"]["submitted"],
    }


def summarize(rows: list[dict], label: str, audit: dict[tuple[str, int], str]) -> dict:
    print(f"\n{'=' * 92}\n{label}   n={len(rows)}\n{'=' * 92}")
    print(
        f"{'task':28s} {'ep':>2s} {'trn':>4s} {'rd':>3s} {'run':>4s} {'pat':>4s} "
        f"{'fail':>5s} {'files':>6s} {'1st':>4s} {'train':>6s} {'ptok':>8s} {'label':>12s}"
    )
    for r in rows:
        lbl = audit.get((r["task"], r["episode"]), "?")
        print(
            f"{r['task']:28s} {r['episode']:2d} {r['turns']:4d} {r['reads']:3d} "
            f"{r['runs']:4d} {r['patches']:4d} {r['failed_patches']:5d} "
            f"{r['distinct_files_read']:6d} {str(r['first_successful_patch_turn'] or '-'):>4s} "
            f"{'yes' if r['ran_training'] else 'no':>6s} {r['prompt_tokens']:8,d} {lbl:>12s}"
        )

    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_label[audit.get((r["task"], r["episode"]), "?")].append(r)

    def agg(rs: list[dict], key: str):
        vals = [r[key] for r in rs if r[key] is not None]
        return round(statistics.mean(vals), 2) if vals else None

    print(f"\n{'label':14s} {'n':>3s} {'reads':>6s} {'runs':>6s} {'patches':>8s} "
          f"{'files':>6s} {'1st patch':>10s} {'ptok':>10s}")
    per_label = {}
    for lbl, rs in sorted(by_label.items()):
        per_label[lbl] = {
            "n": len(rs),
            "mean_reads": agg(rs, "reads"),
            "mean_runs": agg(rs, "runs"),
            "mean_patches": agg(rs, "patches"),
            "mean_distinct_files_read": agg(rs, "distinct_files_read"),
            "mean_first_patch_turn": agg(rs, "first_successful_patch_turn"),
            "mean_prompt_tokens": agg(rs, "prompt_tokens"),
            "ran_training_frac": round(sum(r["ran_training"] for r in rs) / len(rs), 2),
        }
        v = per_label[lbl]
        print(
            f"{lbl:14s} {v['n']:3d} {str(v['mean_reads']):>6s} {str(v['mean_runs']):>6s} "
            f"{str(v['mean_patches']):>8s} {str(v['mean_distinct_files_read']):>6s} "
            f"{str(v['mean_first_patch_turn']):>10s} {v['mean_prompt_tokens']:10,.0f}"
        )

    all_cmds = Counter()
    for r in rows:
        all_cmds.update(r["commands"])

    summary = {
        "label": label,
        "n": len(rows),
        "per_label": per_label,
        "command_usage": dict(all_cmds),
        "mean_turns": agg(rows, "turns"),
        "mean_reads": agg(rows, "reads"),
        "mean_runs": agg(rows, "runs"),
        "mean_patches": agg(rows, "patches"),
        "failed_patch_total": sum(r["failed_patches"] for r in rows),
        "noop_total": sum(r["noops"] for r in rows),
        "ran_training_frac": round(sum(r["ran_training"] for r in rows) / len(rows), 2),
        "ran_tests_frac": round(sum(r["ran_tests"] for r in rows) / len(rows), 2),
        "submitted_frac": round(sum(r["submitted"] for r in rows) / len(rows), 2),
        "budget_exhausted_frac": round(
            sum("budget" in r["end_reason"] for r in rows) / len(rows), 2
        ),
        "prompt_tokens_total": sum(r["prompt_tokens"] for r in rows),
        "prompt_tokens_per_turn": round(
            sum(r["prompt_tokens"] for r in rows) / max(1, sum(r["turns"] for r in rows))
        ),
    }
    print(f"\ncommand usage: {dict(all_cmds)}")
    for k in ("failed_patch_total", "noop_total", "ran_training_frac", "ran_tests_frac",
              "submitted_frac", "budget_exhausted_frac", "prompt_tokens_per_turn"):
        print(f"  {k:28s} {summary[k]}")
    return summary


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] or [
        REPO_ROOT / "artifacts" / "tier_m_primary.jsonl"
    ]
    out = []
    for p in paths:
        if not p.exists():
            print(f"skip missing {p}")
            continue
        recs = [json.loads(line) for line in p.open(encoding="utf-8")]
        audit_path = p.parent / (p.stem.replace("_primary", "") + "_audit.json")
        audit = {}
        if audit_path.exists():
            for a in json.loads(audit_path.read_text(encoding="utf-8")):
                audit[(a["task"], a["episode"])] = a["label"]
        out.append(summarize([profile(r) for r in recs], p.stem, audit))

    dest = REPO_ROOT / "artifacts" / "trajectory_efficiency.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
