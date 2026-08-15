"""v0.2-C: a thin UK AISI Inspect adapter over the native transformer-repair tasks.

Deliberately thin. The native environment stays the source of truth: this module adds no
task definitions, no checks and no grading logic of its own. It maps the existing pieces
onto Inspect's dataset / solver / scorer interface and nothing more.

  * **dataset**  -- one `Sample` per native `RepoTaskSpec`, carrying the same
    symptom text the native harness shows the agent.
  * **solver**   -- materializes the workspace. The default `mode="gold"` applies the
    reference patch deterministically, which is what makes the smoke evaluation runnable
    with no model and no credentials. `mode="noop"` leaves the planted defect in place.
  * **scorer**   -- calls `trgym.harness.sandbox.run_checks`, i.e. the *same* R16
    trusted-comparator path production grading uses. The candidate's code runs in a
    container with no oracle; the verdict is decided outside it.

Two consequences worth stating. Scoring here inherits the real isolation boundary rather
than approximating it — an Inspect run cannot accidentally grade more permissively than the
native harness. And because the checks come from `spec.hidden_checks`, the adapter cannot
drift into scoring a different suite; `tests/test_inspect_adapter.py` fails if it does.

Run the deterministic smoke (no API key, no model call):

    inspect eval inspect_adapter/transformer_repair_inspect.py --model mockllm/model
    inspect eval inspect_adapter/transformer_repair_inspect.py -T mode=noop --model mockllm/model

The first must score 1.0 (gold passes), the second 0.0 (the planted defect fails). Both
directions matter: a scorer that cannot fail is not a scorer.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer
from inspect_ai.solver import Generate, TaskState, solver

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_TASKS = "m1_attention_regression"


def _spec(task_id: str):
    from trgym.tasks.repo_specs import get_repo_task

    return get_repo_task(task_id)


def build_dataset(task_ids: list[str]) -> MemoryDataset:
    """One Sample per native task, using the native symptom as the prompt."""
    from trgym.repo.build import SYMPTOM

    samples = []
    for task_id in task_ids:
        spec = _spec(task_id)
        samples.append(
            Sample(
                input=SYMPTOM.format(symptom=spec.symptom),
                target="hidden suite passes",
                id=task_id,
                metadata={
                    "task_id": task_id,
                    "tier": spec.tier,
                    "family": spec.family,
                    # Recorded so an Inspect log is self-describing about what was graded.
                    "hidden_checks": list(spec.hidden_checks),
                    "editable_files": len(spec.editable),
                },
            )
        )
    return MemoryDataset(samples)


@solver
def materialize_repo(mode: str = "gold"):
    """Lay down the workspace this sample will be graded on.

    `gold` gives the deterministic smoke its reference tree; `noop` gives it the negative
    control. Neither calls the model -- a real agent solver would go here, and the scorer
    below would be unchanged, which is the point of keeping grading separate.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        from trgym.repo.build import build_gold, build_repo

        task_id = state.metadata["task_id"]
        spec = _spec(task_id)
        workspace = Path(tempfile.mkdtemp(prefix=f"inspect_{task_id}_"))
        tree = workspace / "repo"
        if mode == "gold":
            build_gold(spec, tree)
        elif mode == "noop":
            build_repo(spec, tree)
        else:
            raise ValueError(f"unknown mode {mode!r}; expected 'gold' or 'noop'")
        state.metadata["workspace"] = str(tree)
        state.metadata["mode"] = mode
        return state

    return solve


@scorer(metrics=[accuracy()])
def trgym_hidden_suite():
    """Grade through the native R16 trusted comparator. No grading logic lives here."""

    async def score(state: TaskState, target: Target) -> Score:
        from trgym.harness.sandbox import run_checks

        task_id = state.metadata["task_id"]
        tree = Path(state.metadata["workspace"])
        spec = _spec(task_id)
        try:
            # fallback=False: never silently degrade to an unisolated in-process grade.
            result = run_checks(tree, task_id, list(spec.hidden_checks), fallback=False)
            passed = all(ok for _, ok, _ in result.results)
            failed = [n for n, ok, _ in result.results if not ok]
            return Score(
                value=CORRECT if passed else INCORRECT,
                answer=f"{len(result.results) - len(failed)}/{len(result.results)} checks",
                explanation=(
                    f"backend={result.backend}; "
                    + ("all hidden checks passed" if passed else f"failed: {failed}")
                ),
                metadata={"backend": result.backend, "failed_checks": failed},
            )
        finally:
            shutil.rmtree(tree.parent, ignore_errors=True)

    return score


@task
def transformer_repair(mode: str = "gold", tasks: str = DEFAULT_TASKS) -> Task:
    """Inspect entry point. `-T mode=gold|noop`, `-T tasks=id1,id2`."""
    task_ids = [t.strip() for t in tasks.split(",") if t.strip()]
    return Task(
        dataset=build_dataset(task_ids),
        solver=materialize_repo(mode),
        scorer=trgym_hidden_suite(),
    )
