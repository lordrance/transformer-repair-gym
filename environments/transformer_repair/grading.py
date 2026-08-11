"""Host-side grading bridge: the reason the gold oracle stays unreachable.

The security property frozen for this project is that the oracle is *unreachable*, not
merely protected. This module is where that is enforced, and it takes TWO boundaries, not
one — a correction to the first version of this file, which claimed the property from the
first boundary alone (see PROTOCOL_CHANGELOG R14):

  1. **Outbound.** The candidate's code lives in a v1 `Runtime`. The hidden checks and the
     gold repo are never written into it, so an agent that greps its entire filesystem
     finds nothing. G1's `test_grading_artifacts_are_not_candidate_readable` verifies this.

  2. **Inbound.** Grading must *execute* the candidate's code to check it, and importing a
     module runs its top-level statements. Doing that in the rollout process would put
     candidate code exactly where `gold_repo()` is importable — so the grade itself runs in
     a locked-down container too. Boundary 1 constrains the agent; only boundary 2
     constrains code the grader imports on the agent's behalf.

Instead the candidate's edited sources are read *out* through `runtime.read()`, laid
down in a fresh host-side temp directory, and graded there. `trgym.repo.checks.gold_repo`
builds gold in a process-local temp dir that is likewise never mounted anywhere. A
candidate that greps its own entire filesystem finds nothing to tamper with, because
nothing is there -- which is a stronger claim than "the file is read-only".

Consequence worth stating plainly: grading trusts the *bytes read out of* the runtime,
not the runtime's own report of success. The candidate can run whatever it likes inside
the container; only its source files are graded.

  3. **Integrity.** Both boundaries above are about *reach*, and R14/R15 tried to make the
     answer honest while the candidate and the checks still shared one process inside the
     grading container. That was the wrong shape, and the G5 canaries eventually showed
     why: the repository was mounted at `/grader`, so candidate code could simply read
     `trgym/repo/checks.py`, or walk the object graph to `gold_repo()`. No gate fixes that.

     R16 removes the premise instead. The candidate container mounts no repository: it
     receives its own workspace, a standalone probe, and the public task inputs. It
     computes *observations* and returns them over a nonced, explicitly validated channel.
     Gold, the hidden checks and every predicate live out here, in a process that never
     imports candidate code. See `trgym/repo/predicates.py` and SECURITY_MODEL.md.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Sequence

from trgym.repo.checks import run_repo_checks
from trgym.tasks.repo_specs import RepoTaskSpec


class GradeOutcome:
    """Per-check outcomes for one candidate workspace."""

    def __init__(self, results: dict[str, bool], errors: dict[str, str]) -> None:
        self.results = results
        self.errors = errors

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(self.results.values())

    @property
    def n_passed(self) -> int:
        return sum(1 for v in self.results.values() if v)

    def as_info(self) -> dict:
        return {
            "checks": dict(self.results),
            "n_passed": self.n_passed,
            "n_total": len(self.results),
            # Truncated: a check failure message can carry a full tensor repr.
            "errors": {k: v[:800] for k, v in self.errors.items()},
        }


async def materialize_candidate(runtime, spec: RepoTaskSpec, workdir: str) -> Path:
    """Copy the candidate's editable sources out of the runtime into a host temp dir.

    Only `spec.editable` is read back. That is deliberate: a candidate cannot smuggle a
    `conftest.py`, a `sitecustomize.py`, or a patched `trgym_visible_checks.py` into the
    graded tree, because files outside the declared editable set are never copied out --
    the graded tree is reconstructed from the buggy template and then overwritten with
    only the candidate's versions of files it was allowed to touch.
    """
    from trgym.repo.build import build_repo

    dest = Path(tempfile.mkdtemp(prefix=f"trgym_grade_{spec.task_id}_"))
    build_repo(spec, dest)  # baseline: the buggy tree, so a deleted file still grades

    for rel in spec.editable:
        try:
            raw = await runtime.read(f"{workdir}/{rel}")
        except Exception:  # noqa: BLE001 - an unreadable file grades as unchanged
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    return dest


def grade_workspace(
    workspace: Path,
    task_id: str,
    checks: Sequence[str],
    *,
    allow_in_process: bool = False,
) -> GradeOutcome:
    """Grade a host-side workspace, in a sandbox container by default.

    **Why not in-process.** Grading has to *execute* the candidate's code to check it:
    `RepoModules` puts the workspace on `sys.path[0]` and calls
    `importlib.import_module("tinygpt")`, which runs every module-level statement the
    candidate wrote. Doing that in the rollout process puts candidate code in the one
    process where `trgym.repo.checks.gold_repo()` is importable and the gold tree is on
    disk. The container boundary G1 verifies is real, but it constrains the *agent*; it
    does not constrain code that the grader itself imports afterwards.

    So "the oracle is unreachable" holds only if grading is also sandboxed. It is a
    narrower claim than the first version of this module implied, and the fix is to route
    grading through the locked-down container (`--network=none --cap-drop=ALL --read-only`,
    non-root, tmpfs scratch) that Phase 1 already used. See PROTOCOL_CHANGELOG R14.

    `allow_in_process=True` exists only for host-side tests that grade *gold* or a
    template-built tree -- code this repo wrote. It must never be used on a tree a policy
    has touched, and `validate()`'s gold preflight is the one legitimate caller.
    """
    from trgym.harness.sandbox import run_checks

    workspace = Path(workspace)
    if not allow_in_process:
        # R15's layer 1 -- a static refusal for sources that named the grader -- used to
        # stand here. It is gone (R16). Two reasons, and the second is the important one:
        #
        #   * It is no longer load-bearing. The candidate container no longer contains the
        #     grader, gold, the template or the checks source, so a tree that "reaches for
        #     the grading machinery" reaches an empty filesystem. Refusing such a tree
        #     defended nothing that absence does not already defend.
        #   * It actively hid the boundary from measurement. Every canary aimed at the
        #     oracle was refused before executing, so the suite reported "contained" for
        #     probes that never ran -- the vacuous-green failure this project keeps
        #     rediscovering. Deleting the gate is what let those probes finally measure
        #     the real thing.
        #
        # A pattern denylist was always an arms race against the next spelling. The fix
        # is that there is nothing in the room to steal.
        result = run_checks(workspace, task_id, list(checks), fallback=False)
        results: dict[str, bool] = {}
        errors: dict[str, str] = {}
        # Layer 2 (R15): the worker discards its own verdict if the tolerances or the
        # check registry moved while grading ran. A tampered run is a zero, never a pass.
        if result.tampered:
            return GradeOutcome(
                {name: False for name in checks},
                {name: "grader state was modified during grading; verdict discarded"
                 for name in checks},
            )
        for name, ok, message in result.results:
            results[name] = bool(ok)
            if not ok:
                errors[name] = message or "check failed without a message"
        for name in checks:
            if name not in results:
                results[name] = False
                errors[name] = "check did not run (unknown name?)"
        return GradeOutcome(results, errors)

    return _grade_in_process(workspace, task_id, checks)


def _grade_in_process(
    workspace: Path, task_id: str, checks: Sequence[str]
) -> GradeOutcome:
    """Import and check the workspace in THIS process. Trusted trees only."""
    results: dict[str, bool] = {}
    errors: dict[str, str] = {}

    # `run_repo_checks` RETURNS (name, ok, message) tuples -- it does not raise on a
    # failing check. An earlier version of this function wrapped it in try/except and
    # recorded True whenever no exception escaped, which made every check pass and the
    # whole verifier vacuously green: gold and the planted bug both scored 1.0.
    # Consume the return value. See PROTOCOL_CHANGELOG R11.
    outcomes = run_repo_checks(Path(workspace), task_id, list(checks))

    seen = set()
    for name, ok, message in outcomes:
        results[name] = bool(ok)
        seen.add(name)
        if not ok:
            errors[name] = message or "check failed without a message"

    # A check that silently vanished must not read as a pass. Missing names are
    # recorded as failures so a typo shrinks the score rather than the suite.
    for name in checks:
        if name not in seen:
            results[name] = False
            errors[name] = "check did not run (unknown name?)"

    return GradeOutcome(results, errors)


def discard(workspace: Path) -> None:
    shutil.rmtree(workspace, ignore_errors=True)
