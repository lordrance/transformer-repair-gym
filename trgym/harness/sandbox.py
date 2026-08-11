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
import secrets
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGE = "trgym-sandbox:latest"
MARKER = "<<<TRGYM_OBS"   # completed per job as `<<<TRGYM_OBS:{nonce}>>>`

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
    tampered: bool = False
    """True when the worker observed candidate code rewriting the grader's decision
    surface, or when the result protocol could not be authenticated. See R15."""


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


PROBE_SOURCE_PATH = REPO_ROOT / "trgym" / "repo" / "candidate_probe.py"


def _stage_probe(into: Path) -> Path:
    """Copy the candidate-side probe into an otherwise empty directory.

    This directory is the *only* thing besides the workspace that the candidate container
    mounts. Staging a lone file rather than mounting the repository is the whole fix: the
    container that executes candidate code no longer contains `trgym.repo.checks`, the
    gold tree, or the pristine template, so there is nothing for candidate code to read,
    import, or reach by walking the object graph. See PROTOCOL_CHANGELOG R16.
    """
    into.mkdir(parents=True, exist_ok=True)
    dest = into / "candidate_probe.py"
    shutil.copyfile(PROBE_SOURCE_PATH, dest)
    # `mkdtemp` yields a 0700 directory owned by whoever is running, and the sandbox runs
    # as uid 10001, so without this the container cannot even open the probe -- it exits
    # with "Permission denied" and every check reads as a candidate failure. Exactly the
    # trap the workspace helper below already documents. World-readable is correct here:
    # the probe holds no secret, it is mounted `:ro`, and the candidate is about to run
    # it anyway.
    import os

    if os.name == "posix":
        try:
            os.chmod(into, 0o755)
            os.chmod(dest, 0o644)
        except OSError:
            pass
    return dest


def _make_readable_by_the_sandbox_user(workspace: Path) -> None:
    """Open the workspace's POSIX permissions so uid 10001 in the container can use it.

    The image runs as a non-root `runner` user, and a bind mount carries the host's
    ownership straight through. When the caller is root on Linux -- which it is whenever
    grading runs inside the project's own container -- `mkdtemp` produces a 0700 directory
    owned by uid 0, and the sandbox then cannot even list the tree it is supposed to grade.
    Every check fails with an import error, which reads as "the candidate is wrong" instead
    of "grading never happened".

    The workspace is a disposable per-job copy of candidate code, mounted rw by design, so
    widening its mode costs nothing that was not already granted. No-op on Windows, where
    the mode bits do not carry.
    """
    import os

    if os.name != "posix":
        return
    try:
        os.chmod(workspace, 0o777)
        for path in workspace.rglob("*"):
            os.chmod(path, 0o777 if path.is_dir() else 0o666)
    except OSError:
        pass  # best effort: a permission failure here surfaces as a check failure


def run_checks_containerized(
    workspace: Path,
    task_id: str,
    checks: Sequence[str],
    *,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    memory: str = DEFAULT_MEMORY,
    cpus: str = DEFAULT_CPUS,
) -> SandboxResult:
    """Grade `workspace` by observing it in a container and deciding out here.

    The container computes observations and returns them. It is handed no gold, no
    oracle, no template and no reference data, so there is nothing in it worth stealing
    and no verdict in it worth forging. Every predicate runs in this process, after the
    container has exited, against data that has been validated field by field.
    """
    import tempfile

    from trgym.repo import predicates
    from trgym.repo.obs_protocol import ProtocolError, decode_observations

    workspace = Path(workspace).resolve()
    _make_readable_by_the_sandbox_user(workspace)
    checks = list(checks)
    nonce = secrets.token_hex(16)
    marker = f"{MARKER}:{nonce}>>>"

    # Gold is consulted HERE, in the trusted process, both for the public inputs the
    # candidate needs and later for the comparisons themselves.
    groups = predicates.groups_for(checks)
    try:
        inputs = predicates.build_inputs(task_id, groups)
    except Exception as exc:  # noqa: BLE001 - a gold failure is an infrastructure failure
        return SandboxResult(
            results=tuple((c, False, f"reference unavailable: {exc}") for c in checks),
            ok=False, timed_out=False, stderr=str(exc)[-2000:], wall_time_s=0.0,
            backend="docker",
        )

    request = json.dumps(
        {"workspace": "/workspace", "nonce": nonce, "groups": groups, "inputs": inputs}
    )

    staging = Path(tempfile.mkdtemp(prefix="trgym_probe_"))
    try:
        _stage_probe(staging)
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
            # NO repository mount. The probe is staged alone; /grader does not exist.
            "-v", f"{staging}:/probe:ro",
            "-v", f"{workspace}:/workspace:rw",     # the only writable mount
            "-w", "/workspace",
            IMAGE,
            "python", "/probe/candidate_probe.py",
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
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    # The nonce, and `rfind`: candidate code shares stdout with the probe and can print
    # whatever it likes, so the document is taken from the LAST block carrying this job's
    # secret marker. An unnonced block written by a candidate does not parse at all.
    try:
        observations, obs_errors = decode_observations(proc.stdout, marker=marker)
    except ProtocolError as exc:
        # Malformed, oversized or unauthenticated output is an infrastructure failure,
        # never a graded outcome: a candidate must not be able to choose its verdict by
        # choosing how to corrupt its output.
        forged = MARKER in proc.stdout
        # The container's own diagnostics ride along. Without them "no authenticated
        # block" is indistinguishable between a crashed probe, a bad mount and a
        # candidate that killed the process, which are three different bugs.
        why = (proc.stderr or proc.stdout or "").strip()[-700:]
        return SandboxResult(
            results=tuple((c, False,
                           f"observation protocol rejected: {exc} "
                           f"[exit={proc.returncode}] {why}")
                          for c in checks),
            ok=False, timed_out=False,
            stderr=(proc.stderr or proc.stdout)[-3000:], wall_time_s=elapsed,
            backend="docker", tampered=forged,
        )

    results = tuple(
        predicates.evaluate(task_id, checks, observations, obs_errors, workspace)
    )
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
