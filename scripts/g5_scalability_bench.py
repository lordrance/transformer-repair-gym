"""G5 -- grading throughput: >=30 sequential jobs, cold path vs final path.

The frozen contract offers two ways to pass:

  A. a safe persistent/batched grader that measurably reduces startup overhead, with a
     100 %-passing isolation suite;
  B. benchmark data showing the official isolated path is already acceptable, with the
     custom optimization deleted.

and it fails outright if isolation is traded for speed. So this script measures, it does not
optimise. The decision between A and B is made from the numbers plus
`scripts/g5_isolation_canaries.py`, and isolation wins ties by construction.

Three paths are timed:

  `cold`         one fresh sandbox container per grading job -- the isolated path that R14
                 made mandatory for candidate-touched trees. This is the honest cost of the
                 security property.
  `final`        the path actually shipped (currently identical to `cold`; if a persistent
                 grader is ever added under Plan A, it is timed here and must clear the same
                 canary suite).
  `in_process`   reference only, for attribution. NOT a candidate path -- R14 forbids it for
                 policy-touched trees. Timed so the report can state what the isolation
                 boundary costs rather than hand-waving it.

Every job grades a *gold* tree, so the timings measure the grader rather than the difficulty
of the code, and `in_process` is safe to time here because gold is code this repo wrote.

Writes artifacts/g5_scalability.json (consumed by scripts/final_acceptance.py).
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "artifacts" / "g5_scalability.json"
TASK_ID = "m1_attention_regression"
DEFAULT_JOBS = 30


def build_gold_tree(dest: Path) -> Path:
    from trgym.repo.build import build_gold
    from trgym.tasks.repo_specs import get_repo_task

    return build_gold(get_repo_task(TASK_ID), dest)


def checks_for() -> list[str]:
    from trgym.repo.verifier_v2 import v2_checks
    from trgym.tasks.repo_specs import get_repo_task

    return list(v2_checks(get_repo_task(TASK_ID)))


def time_sandboxed(ws: Path, checks: list[str]) -> tuple[float, bool, str]:
    from trgym.harness.sandbox import run_checks

    t0 = time.perf_counter()
    result = run_checks(ws, TASK_ID, checks, fallback=False)
    elapsed = time.perf_counter() - t0
    ok = all(o for _, o, _ in result.results)
    return elapsed, ok, result.backend


def time_in_process(ws: Path, checks: list[str]) -> tuple[float, bool, str]:
    from environments.transformer_repair.grading import grade_workspace

    t0 = time.perf_counter()
    outcome = grade_workspace(ws, TASK_ID, checks, allow_in_process=True)
    return time.perf_counter() - t0, outcome.passed, "in_process"


def stats(samples: list[float]) -> dict:
    if not samples:
        return {"n": 0, "mean_s": None, "p50_s": None, "p95_s": None,
                "min_s": None, "max_s": None}
    ordered = sorted(samples)
    # Nearest-rank p95: with n=30 this is the 29th value. Stated explicitly because
    # interpolated percentiles on 30 samples invite over-reading.
    idx95 = max(0, min(len(ordered) - 1, int(round(0.95 * len(ordered))) - 1))
    return {
        "n": len(samples),
        "mean_s": round(statistics.fmean(ordered), 4),
        "p50_s": round(statistics.median(ordered), 4),
        "p95_s": round(ordered[idx95], 4),
        "min_s": round(ordered[0], 4),
        "max_s": round(ordered[-1], 4),
        "stdev_s": round(statistics.stdev(ordered), 4) if len(ordered) > 1 else 0.0,
        "percentile_method": "nearest-rank",
    }


def run_path(label: str, runner, n_jobs: int, checks: list[str]) -> dict:
    print(f"\n-- {label}: {n_jobs} sequential grading jobs --")
    samples: list[float] = []
    failures: list[dict] = []
    backends: set[str] = set()

    for i in range(1, n_jobs + 1):
        ws = Path(tempfile.mkdtemp(prefix=f"g5_{label}_{i}_"))
        try:
            build_gold_tree(ws)
            elapsed, ok, backend = runner(ws, checks)
            samples.append(elapsed)
            backends.add(backend)
            if not ok:
                # Gold failing is a real defect, not a timing artifact -- record it.
                failures.append({"job": i, "reason": "gold did not pass"})
            if i % 5 == 0 or i == 1:
                print(f"   job {i:>3}  {elapsed:6.2f}s  backend={backend}")
        except Exception as exc:  # noqa: BLE001 - an unavailable path is data
            failures.append({"job": i, "reason": f"{type(exc).__name__}: {exc}"[:200]})
            print(f"   job {i:>3}  ERROR {type(exc).__name__}: {exc}")
        finally:
            shutil.rmtree(ws, ignore_errors=True)

    out = stats(samples)
    out["failures"] = failures
    out["n_failures"] = len(failures)
    out["backends_observed"] = sorted(backends)
    print(f"   mean {out['mean_s']}s  p50 {out['p50_s']}s  p95 {out['p95_s']}s  "
          f"failures {len(failures)}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=DEFAULT_JOBS,
                    help="sequential grading jobs per path (contract minimum 30)")
    ap.add_argument("--skip-in-process", action="store_true",
                    help="omit the reference in-process timing")
    args = ap.parse_args()

    if args.jobs < 30:
        print(f"refusing: the frozen contract requires >=30 jobs, got {args.jobs}")
        return 1

    from trgym.harness.sandbox import docker_available, image_exists

    if not (docker_available() and image_exists()):
        print("FAIL: the sandbox image is unavailable. Build it first:\n"
              "      python scripts/build_sandbox.py\n"
              "Grading candidate trees in-process is forbidden by R14, so there is no\n"
              "fallback to measure here.")
        return 1

    checks = checks_for()
    print(f"grading {TASK_ID} gold with {len(checks)} v2 checks per job")

    cold = run_path("cold", time_sandboxed, args.jobs, checks)

    # The shipped path. Identical to `cold` today: R14 requires one isolated grade per
    # candidate tree, and no persistent grader has been introduced. Timed separately so the
    # report compares like with like if that ever changes.
    final = run_path("final", time_sandboxed, args.jobs, checks)

    in_proc = None
    if not args.skip_in_process:
        in_proc = run_path("in_process (reference only, NOT a candidate path)",
                           time_in_process, args.jobs, checks)

    overhead = None
    if in_proc and in_proc["mean_s"] and final["mean_s"]:
        overhead = {
            "isolation_overhead_s": round(final["mean_s"] - in_proc["mean_s"], 4),
            "isolation_overhead_ratio": round(final["mean_s"] / in_proc["mean_s"], 2),
            "note": "the measured price of R14's inbound boundary, per grading job",
        }

    payload = {
        "task_id": TASK_ID,
        "n_jobs": args.jobs,
        "n_checks_per_job": len(checks),
        "cold": cold,
        "final": final,
        "in_process_reference": in_proc,
        "overhead": overhead,
        "decision": {
            "contract_option": "B",
            "rationale": (
                "The official isolated sandbox is used per grading job. No custom "
                "persistent grader is retained: R14 established that candidate-touched "
                "trees must not be graded in a process holding gold, and the contract "
                "fails G5 outright if isolation is traded for speed. Option A would "
                "require a persistent grader that passes the full canary suite; none is "
                "shipped, so there is no custom optimization to delete."
            ),
        },
        "isolation_evidence": "artifacts/g5_isolation_canaries.json",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {OUT.relative_to(ROOT).as_posix()}")

    if cold["n_failures"] or final["n_failures"]:
        print("FAIL: gold did not pass on every job; grading is not reliable")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
