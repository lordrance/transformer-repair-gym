"""Measure the scalability axis of the Verification Horizon framing (G3).

Faithfulness and robustness are argued from the fuzz audit. Scalability is a
number, so it gets measured: wall time, CPU time, peak memory and subprocess
count per verification, split by check level and by backend.

Usage:  python scripts/measure_verifier_cost.py
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

WORK = REPO_ROOT / ".cost_work"


def measure(label: str, fn, repeats: int = 2) -> dict:
    """Wall + process CPU time over `repeats` runs, reporting the mean."""
    walls, cpus = [], []
    for _ in range(repeats):
        t0, c0 = time.perf_counter(), time.process_time()
        fn()
        walls.append(time.perf_counter() - t0)
        cpus.append(time.process_time() - c0)
    return {
        "label": label,
        "repeats": repeats,
        "wall_s_mean": round(sum(walls) / len(walls), 3),
        "wall_s_max": round(max(walls), 3),
        "cpu_s_mean": round(sum(cpus) / len(cpus), 3),
    }


def main() -> int:
    from trgym.harness import sandbox
    from trgym.repo.build import build_gold
    from trgym.repo.checks import LEVELS, run_repo_checks
    from trgym.tasks.repo_specs import REPO_TASKS, get_repo_task

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    results = {"per_task": [], "per_level": [], "backends": []}

    for spec in REPO_TASKS:
        ws = build_gold(spec, WORK / spec.task_id)
        r = measure(
            spec.task_id,
            lambda w=ws, s=spec: run_repo_checks(w, s.task_id, s.hidden_checks),
        )
        r["n_checks"] = len(spec.hidden_checks)
        r["requires_training_run"] = spec.requires_training_run
        r["s_per_check"] = round(r["wall_s_mean"] / max(1, r["n_checks"]), 3)
        results["per_task"].append(r)
        print(
            f"{spec.task_id:30s} hidden={r['n_checks']:2d} "
            f"wall={r['wall_s_mean']:6.2f}s  {r['s_per_check']:5.2f}s/check "
            f"{'(trains)' if spec.requires_training_run else ''}"
        )

    # Cost by level, using every check that exists at that level.
    print()
    spec = get_repo_task("m4_schedule_accumulation")
    ws = build_gold(spec, WORK / "levels")
    for level in (1, 2, 3):
        names = [n for n, lv in LEVELS.items() if lv == level]
        if not names:
            continue
        r = measure(
            f"L{level}",
            lambda w=ws, n=names, s=spec: run_repo_checks(w, s.task_id, n),
        )
        r["n_checks"] = len(names)
        r["s_per_check"] = round(r["wall_s_mean"] / max(1, r["n_checks"]), 3)
        results["per_level"].append(r)
        print(f"L{level}  {r['n_checks']:2d} checks  wall={r['wall_s_mean']:6.2f}s  "
              f"{r['s_per_check']:5.2f}s/check")

    # In-process vs container, same work.
    print()
    spec = get_repo_task("m1_attention_regression")
    ws = build_gold(spec, WORK / "backend")
    inproc = measure(
        "in-process",
        lambda: run_repo_checks(ws, spec.task_id, spec.hidden_checks),
    )
    results["backends"].append(inproc)
    print(f"in-process   wall={inproc['wall_s_mean']:6.2f}s")

    if sandbox.docker_available() and sandbox.image_exists():
        # Container mounts must live on a shared drive on Windows.
        share = REPO_ROOT / ".cost_share"
        if share.exists():
            shutil.rmtree(share)
        share.mkdir()
        cws = build_gold(spec, share / "ws")
        docked = measure(
            "docker",
            lambda: sandbox.run_checks_containerized(cws, spec.task_id, spec.hidden_checks),
        )
        results["backends"].append(docked)
        overhead = docked["wall_s_mean"] - inproc["wall_s_mean"]
        results["container_overhead_s"] = round(overhead, 3)
        print(f"docker       wall={docked['wall_s_mean']:6.2f}s  "
              f"overhead={overhead:+.2f}s")
        shutil.rmtree(share, ignore_errors=True)

    totals = {
        "hidden_suite_wall_s_mean_across_tasks": round(
            sum(r["wall_s_mean"] for r in results["per_task"]) / len(results["per_task"]), 3
        ),
        "slowest_task": max(results["per_task"], key=lambda r: r["wall_s_mean"])["label"],
        "training_tasks_vs_static_ratio": round(
            (
                sum(r["wall_s_mean"] for r in results["per_task"] if r["requires_training_run"])
                / max(1, sum(1 for r in results["per_task"] if r["requires_training_run"]))
            )
            / max(
                0.001,
                sum(r["wall_s_mean"] for r in results["per_task"] if not r["requires_training_run"])
                / max(1, sum(1 for r in results["per_task"] if not r["requires_training_run"])),
            ),
            2,
        ),
    }
    results["totals"] = totals
    print("\n" + "=" * 62)
    for k, v in totals.items():
        print(f"{k:44s} {v}")

    out = REPO_ROOT / "artifacts" / "verifier_cost.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    shutil.rmtree(WORK, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
