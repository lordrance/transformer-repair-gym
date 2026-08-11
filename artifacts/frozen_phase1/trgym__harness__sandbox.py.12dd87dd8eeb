"""Run candidate code inside a Docker container.

This is the security boundary; `trgym/harness/tools.py` is not. The container
gets no network, a read-only grader mount, a capped CPU and memory allowance, a
wall-clock timeout, dropped capabilities and a non-root user. See
SANDBOX_DESIGN.md for the threat model and for what is explicitly out of scope.

The Windows-host subprocess path from Phase 0 remains available as
`fallback=True` for fast local iteration. It is not a boundary and refuses to
pretend otherwise.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE = "trgym-sandbox:latest"
MARKER = "<<<TRGYM_RESULT>>>"

DEFAULT_TIMEOUT_S = 600
DEFAULT_MEMORY = "2g"
DEFAULT_CPUS = "2.0"
DEFAULT_PIDS = 256


@dataclass(frozen=True)
class SandboxResult:
    results: tuple[tuple[str, bool, str], ...]
    ok: bool
    timed_out: bool
    stderr: str
    wall_time_s: float
    backend: str


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True, text=True, timeout=20,
        )
        return proc.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def image_exists() -> bool:
    try:
        proc = subprocess.run(
            ["docker", "image", "inspect", IMAGE], capture_output=True, text=True, timeout=30
        )
        return proc.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def build_image(quiet: bool = False) -> tuple[bool, str]:
    cmd = ["docker", "build", "-t", IMAGE, "-f", str(REPO_ROOT / "docker" / "Dockerfile")]
    if quiet:
        cmd.append("--quiet")
    cmd.append(str(REPO_ROOT / "docker"))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    return proc.returncode == 0, (proc.stdout + proc.stderr)[-4000:]


WORKER = r"""
import json, sys
sys.path.insert(0, "/grader")
from trgym.repo.checks import run_repo_checks
req = json.loads(sys.stdin.read())
res = run_repo_checks(req["workspace"], req["task_id"], req["checks"])
sys.stdout.write("<<<TRGYM_RESULT>>>" + json.dumps(
    [{"name": n, "passed": ok, "detail": d} for n, ok, d in res]))
"""


def run_checks_containerized(
    workspace: Path,
    task_id: str,
    checks: Sequence[str],
    *,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    memory: str = DEFAULT_MEMORY,
    cpus: str = DEFAULT_CPUS,
) -> SandboxResult:
    """Grade `workspace` inside a locked-down container."""
    workspace = Path(workspace).resolve()
    request = json.dumps(
        {"workspace": "/workspace", "task_id": task_id, "checks": list(checks)}
    )

    cmd = [
        "docker", "run", "--rm", "-i",
        "--network=none",                       # no egress, no lateral movement
        f"--memory={memory}", f"--memory-swap={memory}",   # no swap escape hatch
        f"--cpus={cpus}",
        f"--pids-limit={DEFAULT_PIDS}",         # fork bombs cannot starve the host
        "--cap-drop=ALL",                       # no capabilities at all
        "--security-opt=no-new-privileges",     # setuid binaries cannot escalate
        "--read-only",                          # rootfs immutable...
        "--tmpfs=/tmp:rw,noexec,nosuid,size=128m",   # ...with a scratch dir
        "--tmpfs=/home/runner:rw,nosuid,size=32m",
        "-v", f"{REPO_ROOT}:/grader:ro",        # grader + protected oracle, read-only
        "-v", f"{workspace}:/workspace:rw",     # the only writable mount
        "-w", "/workspace",
        IMAGE,
        "python", "-c", WORKER,
    ]

    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd, input=request, capture_output=True, text=True, timeout=timeout_s
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(
            results=tuple((c, False, "container timed out") for c in checks),
            ok=False, timed_out=True,
            stderr=f"exceeded {timeout_s}s", wall_time_s=float(timeout_s),
            backend="docker",
        )
    elapsed = time.perf_counter() - started

    at = proc.stdout.find(MARKER)
    if at < 0:
        return SandboxResult(
            results=tuple((c, False, "container produced no result") for c in checks),
            ok=False, timed_out=False,
            stderr=(proc.stderr or proc.stdout)[-3000:], wall_time_s=elapsed,
            backend="docker",
        )

    payload = json.loads(proc.stdout[at + len(MARKER) :])
    results = tuple((r["name"], r["passed"], r["detail"]) for r in payload)
    return SandboxResult(
        results=results,
        ok=all(ok for _, ok, _ in results),
        timed_out=False,
        stderr=proc.stderr[-2000:],
        wall_time_s=elapsed,
        backend="docker",
    )


def run_checks(
    workspace: Path,
    task_id: str,
    checks: Sequence[str],
    *,
    fallback: bool = False,
    **kwargs,
) -> SandboxResult:
    """Grade in a container; optionally fall back to an in-process run.

    `fallback=True` is for local iteration only. It shares the filesystem and
    network with the host and provides no isolation whatsoever, so it must never
    be used for an unattended run or for code from an untrusted policy.
    """
    if docker_available() and image_exists():
        return run_checks_containerized(workspace, task_id, checks, **kwargs)
    if not fallback:
        missing = "docker daemon" if not docker_available() else f"image {IMAGE}"
        raise RuntimeError(
            f"{missing} unavailable. Build it with "
            f"`python scripts/build_sandbox.py`, or pass fallback=True to run "
            f"without isolation (development only)."
        )

    from trgym.repo.checks import run_repo_checks

    started = time.perf_counter()
    results = tuple(run_repo_checks(Path(workspace), task_id, checks))
    return SandboxResult(
        results=results,
        ok=all(ok for _, ok, _ in results),
        timed_out=False,
        stderr="",
        wall_time_s=time.perf_counter() - started,
        backend="in-process (NOT ISOLATED)",
    )
