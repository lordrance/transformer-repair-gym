"""Record, as machine-readable evidence, what the `verifiers.v1` runtime actually did.

`verifiers.v1` imports POSIX-only `fcntl`, so its behaviour can only be exercised in
Linux/Docker. `scripts/final_acceptance.py` has to run anywhere -- including a Windows
clean-room -- and the frozen rule is that a check it cannot execute must FAIL rather than
skip. Reconciling those two facts needs a third thing: an artifact, written by a run that
really did execute v1, that acceptance can read afterwards.

That artifact is what this script produces. It is deliberately *not* a report: every field
is the outcome of a command run in this process, and the grading source is hashed into it
so acceptance can reject evidence that predates the code it claims to describe.

Run it from inside the Linux image, with the repo mounted at a path the Docker daemon can
also resolve (see REPRODUCIBILITY.md), because grading itself now spawns a container:

    docker run --rm -v "e:/RL:/run/desktop/mnt/host/e/RL" \
      -w /run/desktop/mnt/host/e/RL \
      -e PYTHONPATH="/run/desktop/mnt/host/e/RL:/run/desktop/mnt/host/e/RL/environments" \
      -v /var/run/docker.sock:/var/run/docker.sock -v /tmp:/tmp \
      trgym-v1:latest python scripts/v1_runtime_evidence.py
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "environments"))

OUT = ROOT / "artifacts" / "raw" / "v1_runtime_evidence.json"
TASK_ID = "m1_attention_regression"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect() -> dict:
    ev: dict = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "grading_sha256": _sha256(ROOT / "environments" / "transformer_repair" / "grading.py"),
        "task_sha256": _sha256(ROOT / "environments" / "transformer_repair" / "task.py"),
    }

    # ------------------------------------------------------------------ v1 import
    try:
        import verifiers

        import verifiers.v1  # noqa: F401

        ev["v1_import_ok"] = True
        ev["verifiers_version"] = getattr(verifiers, "__version__", "unknown")
    except Exception as exc:  # noqa: BLE001
        ev["v1_import_ok"] = False
        ev["v1_import_error"] = f"{type(exc).__name__}: {exc}"
        return ev

    # ------------------------------------------------- taskset via the official loader
    try:
        from verifiers.v1.utils.loaders import load_taskset, taskset_config_type

        cfg_type = taskset_config_type("transformer_repair")
        taskset = load_taskset(cfg_type(id="transformer_repair"))
        tasks = list(taskset)
        ev["n_tasks"] = len(tasks)
        ev["task_ids"] = sorted(t.data.task_id for t in tasks)
        ev["loader"] = "verifiers.v1.utils.loaders.load_taskset"
    except Exception as exc:  # noqa: BLE001
        ev["n_tasks"] = 0
        ev["taskset_error"] = f"{type(exc).__name__}: {exc}"
        tasks = []

    # ------------------------------------------------------------ TaskData immutability
    if tasks:
        data = tasks[0].data
        try:
            data.task_id = "mutated"  # type: ignore[misc]
            ev["taskdata_immutable"] = False
        except Exception:  # noqa: BLE001 - pydantic frozen model raises
            ev["taskdata_immutable"] = True
        # And it must carry no gold.
        blob = json.dumps(data.model_dump(), default=str)
        ev["taskdata_carries_no_gold"] = not any(
            marker in blob for marker in ("gold", "def causal_attention", "reference_impl")
        )

    # ------------------------------------------------ gold PASS vs planted-bug FAIL
    # Through the production grading entry point, with its default (sandboxed) argument,
    # so this measures the R14 path rather than a convenience shortcut.
    try:
        from environments.transformer_repair.grading import grade_workspace
        from trgym.repo.build import build_gold, build_repo
        from trgym.repo.verifier_v2 import v2_checks
        from trgym.tasks.repo_specs import get_repo_task

        spec = get_repo_task(TASK_ID)
        checks = v2_checks(spec)

        gold_dir = Path(tempfile.mkdtemp(prefix="v1ev_gold_"))
        build_gold(spec, gold_dir)
        t0 = time.perf_counter()
        gold_outcome = grade_workspace(gold_dir, TASK_ID, checks)
        ev["gold_grade_seconds"] = round(time.perf_counter() - t0, 3)

        buggy_dir = Path(tempfile.mkdtemp(prefix="v1ev_buggy_"))
        build_repo(spec, buggy_dir)
        buggy_outcome = grade_workspace(buggy_dir, TASK_ID, checks)

        ev["gold_reward"] = 1.0 if gold_outcome.passed else 0.0
        ev["buggy_reward"] = 1.0 if buggy_outcome.passed else 0.0
        ev["gold_checks"] = gold_outcome.as_info()
        ev["buggy_failed_checks"] = sorted(
            k for k, v in buggy_outcome.results.items() if not v
        )
        ev["grading_backend"] = "sandboxed container (grade_workspace default)"
    except Exception as exc:  # noqa: BLE001
        ev["gold_reward"] = None
        ev["buggy_reward"] = None
        ev["grading_error"] = f"{type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------- tests_v1
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests_v1/test_v1_migration.py", "-q",
         "-p", "no:cacheprovider"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=5400,
    )
    tail = (proc.stdout + proc.stderr)[-6000:]
    ev["tests_v1_exit_code"] = proc.returncode
    ev["tests_v1_passed"] = int(m.group(1)) if (m := re.search(r"(\d+) passed", tail)) else 0
    ev["tests_v1_failed"] = int(m.group(1)) if (m := re.search(r"(\d+) failed", tail)) else 0
    ev["tests_v1_tail"] = tail[-1500:]

    # -------------------------------------------------------------- official validate CLI
    proc = subprocess.run(
        ["validate", "transformer_repair", "@",
         "environments/transformer_repair/configs/m1_smoke.toml"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=3600,
    )
    vtail = (proc.stdout + proc.stderr)[-8000:]
    ev["validate_exit_code"] = proc.returncode
    m = re.search(r"valid[_ ]rate[^0-9]{0,12}([0-9]*\.?[0-9]+)", vtail, re.IGNORECASE)
    ev["validate_valid_rate"] = float(m.group(1)) if m else None
    ev["validate_tail"] = vtail[-2000:]
    return ev


def main() -> int:
    ev = collect()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(ev, indent=2, sort_keys=True), encoding="utf-8")
    keys = ("v1_import_ok", "n_tasks", "taskdata_immutable", "gold_reward", "buggy_reward",
            "tests_v1_passed", "tests_v1_failed", "validate_valid_rate")
    for k in keys:
        print(f"  {k:<24} {ev.get(k)}")
    if ev.get("grading_error"):
        print(f"  grading_error            {ev['grading_error']}")
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
