"""Run candidate checks in an isolated subprocess with a wall-clock timeout.

Model-generated code is untrusted. Phase 0 uses process isolation only, which
stops runaway loops and torch monkey-patching but is NOT a security boundary: it
shares the filesystem and network with the parent. Before any unattended or
large-scale run this must be swapped for a container or a Modal sandbox with
`--network=none`, a read-only mount and CPU/memory limits. See
PHASE_0_FEASIBILITY_REPORT.md, "Sandboxing".
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from trgym.verifier._worker import MARKER

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TIMEOUT_S = 180


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class SandboxOutcome:
    results: tuple[CheckResult, ...]
    timed_out: bool
    crashed: bool
    stderr: str
    wall_time_s: float

    @property
    def all_passed(self) -> bool:
        return (
            not self.timed_out
            and not self.crashed
            and bool(self.results)
            and all(r.passed for r in self.results)
        )


def run_checks_sandboxed(
    workspace: Path,
    task_id: str,
    checks: Sequence[str],
    *,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> SandboxOutcome:
    import time

    request = json.dumps(
        {
            "repo_root": str(REPO_ROOT),
            "workspace": str(Path(workspace).resolve()),
            "task_id": task_id,
            "checks": list(checks),
        }
    )

    started = time.perf_counter()
    try:
        proc = subprocess.run(
            # `-E` ignores PYTHONPATH/PYTHONHOME so an ambient env cannot change
            # what gets imported, while still keeping site-packages (where torch
            # lives). `-I`/`-s` would drop user site-packages and break torch.
            # The worker bootstraps repo_root from the request itself.
            [sys.executable, "-E", str(Path(__file__).with_name("_worker.py"))],
            input=request,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(REPO_ROOT),
        )
    except subprocess.TimeoutExpired:
        return SandboxOutcome(
            results=tuple(CheckResult(c, False, "timed out") for c in checks),
            timed_out=True,
            crashed=False,
            stderr=f"exceeded {timeout_s}s",
            wall_time_s=float(timeout_s),
        )
    elapsed = time.perf_counter() - started

    marker_at = proc.stdout.find(MARKER)
    if marker_at < 0:
        return SandboxOutcome(
            results=tuple(CheckResult(c, False, "worker produced no result") for c in checks),
            timed_out=False,
            crashed=True,
            stderr=(proc.stderr or proc.stdout)[-2000:],
            wall_time_s=elapsed,
        )

    payload = json.loads(proc.stdout[marker_at + len(MARKER) :])
    return SandboxOutcome(
        results=tuple(
            CheckResult(r["name"], r["passed"], r["detail"]) for r in payload["results"]
        ),
        timed_out=False,
        crashed=False,
        stderr=proc.stderr[-2000:],
        wall_time_s=elapsed,
    )
