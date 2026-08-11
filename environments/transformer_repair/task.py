"""`TransformerRepairTask` -- a native `verifiers.v1` Task.

Shape taken from `verifiers.v1.tasksets.lean.taskset.LeanTask`, the shipped reference
for a container-graded task:

  * `NEEDS_CONTAINER = True` so the rollout provisions a Runtime
  * `setup(runtime)` plants the buggy repo with `runtime.write()`
  * `@reward` / `@metric` methods declare `task`/`trace`/`runtime` **by name** --
    `rollout.py` injects them (`invoke(self.task.setup, {"trace":…, "runtime":…})`),
    which is why omitting `trace` from `setup` is legal
  * `validate(runtime)` preflights gold

`Task.score` is NOT overridden. The base implementation already discovers decorated
rewards/metrics, invokes them, and writes through `Trace.record_reward(key, value,
weight)`. Overriding it would bypass the aggregation and weighting that `@reward`'s
`_vf_weight` exists to feed.
"""

from __future__ import annotations

import shlex

from verifiers.v1.runtimes import Runtime
from verifiers.v1.state import State
from verifiers.v1.task import Task, TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.utils.decorators import metric, reward

from .configs import TransformerRepairTaskConfig
from .grading import (
    discard,
    grade_workspace,
    materialize_candidate,
)


class TransformerRepairData(TaskData):
    """Task payload. Deliberately carries no gold and no expected values.

    `TaskData` is serialized into run records, so anything here is effectively public.
    The hidden check *names* are safe to carry (they are labels); the check
    implementations and the gold repo live only in the rollout process.
    """

    task_id: str
    tier: str
    family: str
    provenance: str
    hidden_checks: tuple[str, ...] = ()
    visible_checks: tuple[str, ...] = ()
    editable: tuple[str, ...] = ()
    requires_training_run: bool = False


class TransformerRepairTask(
    Task[TransformerRepairData, State, TransformerRepairTaskConfig]
):
    NEEDS_CONTAINER = True

    # ------------------------------------------------------------------ lifecycle
    async def setup(self, runtime: Runtime) -> None:
        """Plant the buggy repo inside the runtime, one `runtime.write` per file.

        Built into a host temp dir first, then written in. Nothing is bind-mounted:
        a mount would make the host tree writable from the candidate and put the
        grading side of the boundary one `..` away.
        """
        import tempfile
        from pathlib import Path

        from trgym.repo.build import build_repo
        from trgym.tasks.repo_specs import get_repo_task

        spec = get_repo_task(self.data.task_id)
        staging = Path(tempfile.mkdtemp(prefix=f"trgym_plant_{spec.task_id}_"))
        try:
            build_repo(spec, staging)
            workdir = self.config.workdir
            for path in sorted(staging.rglob("*")):
                if not path.is_file():
                    continue
                rel = path.relative_to(staging).as_posix()
                await runtime.write(f"{workdir}/{rel}", path.read_bytes())
        finally:
            discard(staging)

    async def validate(self, runtime: Runtime) -> bool:
        """Preflight: gold must pass the hidden suite and the planted repo must import.

        Runs entirely host-side for the gold half -- gold is never written into the
        runtime, so `validate` cannot be the hole that leaks it.
        """
        import tempfile
        from pathlib import Path

        from trgym.repo.build import build_gold
        from trgym.tasks.repo_specs import get_repo_task

        spec = get_repo_task(self.data.task_id)
        gold_dir = Path(tempfile.mkdtemp(prefix=f"trgym_gold_{spec.task_id}_"))
        try:
            build_gold(spec, gold_dir)
            # `allow_in_process=True` is legitimate here and only here: this tree was
            # built by `build_gold` from our own template, so no policy has touched it.
            # Every candidate-facing grade goes through the sandbox (R14).
            outcome = grade_workspace(
                gold_dir, spec.task_id, self._checks(), allow_in_process=True
            )
            if not outcome.passed:
                return False
        finally:
            discard(gold_dir)

        # And the planted buggy tree must at least be importable in the runtime,
        # or the candidate is debugging our packaging rather than the defect.
        probe = await runtime.run(
            ["bash", "-lc",
             f"cd {shlex.quote(self.config.workdir)} && python -c 'import tinygpt'"],
            {},
        )
        return probe.exit_code == 0

    # -------------------------------------------------------------------- signals
    @reward(weight=1.0)
    async def semantic_repair(self, trace: Trace, runtime: Runtime) -> float:
        """1.0 iff every hidden check passes on the candidate's extracted sources.

        Binary by design. A partial-credit reward here would be a reward-hacking
        surface: the fuzz suite's class-C probes are partial semantic repairs, and
        the frozen protocol grades them WRONG rather than 0.6.

        Deliberately NOT gated on `trace.has_error`, which is what the shipped
        `LeanTask` reference does. `Trace.ok` defaults False and `has_error` is
        `not ok`, so an early return there scores every incomplete rollout 0.0 --
        which silently reclassifies an INFRA/PROTOCOL failure as a capability zero.
        That is exactly the conflation that made `h2` look 0/4 hard when its episodes
        had actually been killed by an adapter bug (PROTOCOL_CHANGELOG R5, R10).
        The workspace is graded on whatever state it is in, and the error status is
        recorded separately as a metric so the two stay separable in analysis.
        """
        from trgym.tasks.repo_specs import get_repo_task

        spec = get_repo_task(self.data.task_id)
        graded = await materialize_candidate(runtime, spec, self.config.workdir)
        try:
            outcome = grade_workspace(graded, spec.task_id, self._checks())
            trace.info["hidden_grading"] = outcome.as_info()
            trace.info["grading_side"] = "host"
            trace.info["suite"] = "v2" if self.config.use_contract_layer else "v1"
            return 1.0 if outcome.passed else 0.0
        finally:
            discard(graded)

    @metric
    async def infra_error(self, trace: Trace) -> float:
        """1.0 when the rollout recorded an error.

        This is the channel that keeps INFRA/PROTOCOL failure separable from a
        capability zero, now that `semantic_repair` no longer swallows it (R10). An
        analysis that averages reward without conditioning on this metric is reporting a
        mix of "the model could not fix it" and "the provider hiccuped".

        Reads `trace.errors`, NOT `trace.has_error`. `has_error` is `not trace.ok`, and
        `ok` is still False while scoring runs -- so the first version of this metric
        returned 1.0 for every rollout including a clean one, which made it useless as
        a discriminator. `trace.errors` is already populated at scoring time.
        See PROTOCOL_CHANGELOG R12.
        """
        return 1.0 if trace.errors else 0.0

    @metric
    async def hit_turn_limit(self, trace: Trace) -> float:
        """1.0 when the episode ended by exhausting its turn budget.

        G3 established this environment is BUDGET-SENSITIVE: at 14 turns voluntary
        submission was 0/20 and budget exhaustion 20/20, while at 24 turns that inverted.
        Recording the stop condition per episode means that confound is visible in the
        raw trace instead of having to be reconstructed by a separate ablation.
        """
        return 1.0 if trace.stop_condition == "max_turns" else 0.0

    @metric
    async def files_changed(self, runtime: Runtime) -> float:
        """How many editable files the candidate actually modified.

        A metric, never a reward: patch size is diagnostic, and rewarding or
        penalising it would teach patch-shape rather than repair. Frozen in Phase 0.
        """
        import tempfile
        from pathlib import Path

        from trgym.repo.build import build_repo
        from trgym.tasks.repo_specs import get_repo_task

        spec = get_repo_task(self.data.task_id)
        baseline = Path(tempfile.mkdtemp(prefix=f"trgym_base_{spec.task_id}_"))
        try:
            build_repo(spec, baseline)
            changed = 0
            for rel in spec.editable:
                original = (baseline / rel).read_bytes() if (baseline / rel).exists() else b""
                try:
                    current = await runtime.read(f"{self.config.workdir}/{rel}")
                except Exception:  # noqa: BLE001 - unreadable counts as unchanged
                    continue
                if current != original:
                    changed += 1
            return float(changed)
        finally:
            discard(baseline)

    @metric
    async def touched_the_defective_files(self, runtime: Runtime) -> float:
        """Localization: fraction of the truly-defective files the candidate edited.

        This is the metric G4's Tier S localization claim will rest on, so it is
        defined here once rather than recomputed per experiment.
        """
        import tempfile
        from pathlib import Path

        from trgym.repo.build import build_repo
        from trgym.tasks.repo_specs import get_repo_task

        spec = get_repo_task(self.data.task_id)
        defective = tuple(spec.mutations)
        if not defective:
            return 0.0

        baseline = Path(tempfile.mkdtemp(prefix=f"trgym_loc_{spec.task_id}_"))
        try:
            build_repo(spec, baseline)
            hit = 0
            for rel in defective:
                original = (baseline / rel).read_bytes() if (baseline / rel).exists() else b""
                try:
                    current = await runtime.read(f"{self.config.workdir}/{rel}")
                except Exception:  # noqa: BLE001
                    continue
                if current != original:
                    hit += 1
            return hit / len(defective)
        finally:
            discard(baseline)

    # --------------------------------------------------------------------- helpers
    def _checks(self) -> tuple[str, ...]:
        from trgym.repo.verifier_v2 import v1_checks, v2_checks
        from trgym.tasks.repo_specs import get_repo_task

        spec = get_repo_task(self.data.task_id)
        return v2_checks(spec) if self.config.use_contract_layer else v1_checks(spec)


__all__ = ["TransformerRepairData", "TransformerRepairTask"]
