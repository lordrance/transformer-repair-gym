"""The Inspect adapter must stay a thin mapping, not a second implementation.

The risk with any compatibility shim is silent drift: the adapter keeps working while
quietly scoring a different suite, grading through a weaker path, or falling back to an
unisolated in-process run when Docker is missing. Any of those would make an Inspect log
disagree with the native harness while both look green.

These tests are static plus a cheap construction check, so they need no Docker and no
model. They skip cleanly when `inspect-ai` is not installed, because it is an optional
extra and the native environment must remain runnable without it.
"""

from __future__ import annotations

import ast
import inspect as pyinspect
from pathlib import Path

import pytest

inspect_ai = pytest.importorskip(
    "inspect_ai", reason="optional extra; install with `uv sync --extra inspect`"
)

ADAPTER = Path(__file__).resolve().parents[1] / "inspect_adapter" / "transformer_repair_inspect.py"


@pytest.fixture(scope="module")
def adapter():
    from inspect_adapter import transformer_repair_inspect as mod

    return mod


# --------------------------------------------------------------------------- #
# It is a mapping, not a reimplementation
# --------------------------------------------------------------------------- #
def test_adapter_scores_exactly_the_native_hidden_suite(adapter) -> None:
    """The graded checks must come from the spec, never from a list held here."""
    from trgym.tasks.repo_specs import get_repo_task

    task = adapter.transformer_repair(tasks="m1_attention_regression")
    sample = task.dataset[0]
    spec = get_repo_task("m1_attention_regression")
    assert sample.metadata["hidden_checks"] == list(spec.hidden_checks)


def test_adapter_defines_no_checks_of_its_own(adapter) -> None:
    src = ADAPTER.read_text(encoding="utf-8")
    assert "def check_repo_" not in src, "the adapter must not define checks"
    assert "CheckFailure" not in src, "the adapter must not decide check outcomes"
    assert "hidden_checks" in src, "it must read the suite from the spec"


def test_scorer_grades_through_the_isolated_path(adapter) -> None:
    """`fallback=False`, or a missing Docker daemon silently downgrades the boundary."""
    src = pyinspect.getsource(adapter.trgym_hidden_suite)
    assert "run_checks" in src, "scoring must go through the native sandbox entry point"
    assert "fallback=False" in src, (
        "the scorer must refuse to grade unisolated; fallback=True would let an Inspect "
        "run score candidate code in the process holding gold"
    )

    tree = ast.parse(ADAPTER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            if name == "run_checks":
                kwargs = {k.arg: k.value for k in node.keywords}
                assert "fallback" in kwargs, "run_checks called without an explicit fallback"
                assert kwargs["fallback"].value is False


def test_adapter_never_grades_in_process(adapter) -> None:
    src = ADAPTER.read_text(encoding="utf-8")
    assert "allow_in_process" not in src
    assert "run_repo_checks" not in src, (
        "run_repo_checks is the in-process surface; the adapter must use the sandboxed one"
    )


# --------------------------------------------------------------------------- #
# It constructs, and both directions are reachable
# --------------------------------------------------------------------------- #
def test_task_constructs_with_solver_and_scorer(adapter) -> None:
    task = adapter.transformer_repair()
    assert len(task.dataset) == 1
    assert task.solver is not None
    assert task.scorer is not None


def test_multiple_tasks_can_be_selected(adapter) -> None:
    task = adapter.transformer_repair(tasks="m1_attention_regression,m2_position_encoding")
    assert [s.id for s in task.dataset] == [
        "m1_attention_regression",
        "m2_position_encoding",
    ]


def test_both_modes_are_supported_and_a_bad_mode_is_rejected(adapter) -> None:
    """`noop` is the negative control; without it the smoke cannot fail."""
    for mode in ("gold", "noop"):
        adapter.transformer_repair(mode=mode)
    with pytest.raises(ValueError, match="unknown mode"):
        import asyncio

        from inspect_ai.solver import TaskState

        solve = adapter.materialize_repo("banana")
        state = TaskState(
            model="mockllm/model", sample_id="x", epoch=1, input="", messages=[],
            metadata={"task_id": "m1_attention_regression"},
        )
        asyncio.run(solve(state, None))


def test_prompt_is_the_native_symptom(adapter) -> None:
    """The adapter must not paraphrase the task; the symptom is part of the measurement."""
    from trgym.tasks.repo_specs import get_repo_task

    task = adapter.transformer_repair(tasks="m1_attention_regression")
    spec = get_repo_task("m1_attention_regression")
    assert spec.symptom in task.dataset[0].input
