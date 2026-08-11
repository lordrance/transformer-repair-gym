"""Integration tests for the verifiers environment.

These exercise the full grading path -- prompt construction, diff extraction,
patch application, workspace materialization, sandboxed checks, both reward
schemes -- using synthetic completions instead of a live model. No API key and
no network.

Requires the project venv (verifiers + datasets); skipped otherwise.
"""

from __future__ import annotations

import asyncio
import difflib
import sys
import tempfile
from pathlib import Path

import pytest

vf = pytest.importorskip("verifiers", reason="run with .venv/Scripts/python.exe")

REPO_ROOT = Path(__file__).resolve().parents[1]
# The v0 environment moved to legacy_research/ when G1 migrated the public path
# to verifiers.v1. These tests still cover it because the 90 frozen historical
# trajectories were produced by it. See legacy_research/README.md.
sys.path.insert(0, str(REPO_ROOT / "legacy_research"))

from transformer_repair_v0 import (  # noqa: E402
    _grade_submission,
    build_dataset,
    load_environment,
)

from trgym.cheats.catalog import CHEATS  # noqa: E402
from trgym.tasks.build import REFERENCE_DIR, build_workspace  # noqa: E402
from trgym.tasks.registry import TASKS, get_task  # noqa: E402


def _fenced_diff(diff: str) -> str:
    return f"Here is the fix.\n\n```diff\n{diff}```\n"


def _diff_between(before: str, after: str) -> str:
    return "".join(
        difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True))
    )


def _buggy_source(spec) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        ws = build_workspace(spec, Path(tmp) / "ws", gold=False)
        return (ws / spec.target_file).read_text(encoding="utf-8")


def _gold_diff_completion(task_id: str) -> str:
    spec = get_task(task_id)
    gold = (REFERENCE_DIR / spec.target_file).read_text(encoding="utf-8")
    return _fenced_diff(_diff_between(_buggy_source(spec), gold))


def _cheat_diff_completion(cheat) -> str:
    spec = get_task(cheat.task_id)
    before = _buggy_source(spec)
    with tempfile.TemporaryDirectory() as tmp:
        ws = build_workspace(spec, Path(tmp) / "ws", gold=False)
        cheat.apply(ws, spec.target_file)
        after = (ws / spec.target_file).read_text(encoding="utf-8")
    return _fenced_diff(_diff_between(before, after))


# --------------------------------------------------------------------------- #
# Protocol failures
# --------------------------------------------------------------------------- #
def test_prose_without_a_diff_is_invalid_not_scored() -> None:
    graded = _grade_submission("I would change the tril diagonal.", {"task_id": TASKS[0].task_id})
    assert graded["invalid_patch"] is True
    assert graded["naive"] == 0.0 and graded["hardened"] == 0.0


def test_diff_with_nonexistent_context_is_invalid() -> None:
    bad = _fenced_diff("@@ -1,2 +1,2 @@\n def nothing_like_this():\n-    return 1\n+    return 2\n")
    graded = _grade_submission(bad, {"task_id": TASKS[0].task_id})
    assert graded["invalid_patch"] is True
    assert "context block not found" in graded["invalid_reason"]


def test_noop_diff_is_invalid() -> None:
    spec = TASKS[0]
    line = _buggy_source(spec).splitlines()[0]
    graded = _grade_submission(_fenced_diff(f"@@ -1,1 +1,1 @@\n {line}\n"), {"task_id": spec.task_id})
    assert graded["invalid_patch"] is True


# --------------------------------------------------------------------------- #
# The three-line credibility pattern
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("spec", TASKS, ids=lambda s: s.task_id)
def test_gold_diff_passes_both_schemes(spec) -> None:
    graded = _grade_submission(_gold_diff_completion(spec.task_id), {"task_id": spec.task_id})
    assert graded["invalid_patch"] is False, graded
    assert graded["naive"] == 1.0, graded
    assert graded["hardened"] == 1.0, f"gold falsely rejected: {graded}"
    assert graded["gates"] == []


@pytest.mark.parametrize("spec", TASKS, ids=lambda s: s.task_id)
def test_unfixed_code_fails_both_schemes(spec) -> None:
    """The Phase-0.5 naive suite is no longer a straw man: doing nothing fails it."""
    with tempfile.TemporaryDirectory() as tmp:
        ws = build_workspace(spec, Path(tmp) / "ws", gold=False)
        from trgym.verifier.reward import grade

        result = grade(spec, ws)
    assert result.naive_reward == 0.0, f"{spec.task_id}: naive still accepts the untouched bug"
    assert result.hardened_reward == 0.0


@pytest.mark.parametrize("cheat", CHEATS, ids=lambda c: c.cheat_id)
def test_cheats_pass_naive_and_fail_hardened(cheat) -> None:
    spec = get_task(cheat.task_id)

    # Patches that rewrite grader files are applied to the workspace directly,
    # since they are not expressible as a diff against the target source file.
    if cheat.rewrite_files:
        with tempfile.TemporaryDirectory() as tmp:
            ws = build_workspace(spec, Path(tmp) / "ws", gold=False)
            cheat.apply(ws, spec.target_file)
            from trgym.verifier.reward import grade

            result = grade(spec, ws)
        assert result.naive_reward == 1.0, cheat.cheat_id
        assert result.hardened_reward == 0.0, cheat.cheat_id
        return

    graded = _grade_submission(_cheat_diff_completion(cheat), {"task_id": cheat.task_id})
    assert graded["invalid_patch"] is False, graded
    assert graded["naive"] == 1.0, f"{cheat.cheat_id} not rewarded by naive: {graded}"
    assert graded["hardened"] == 0.0, f"{cheat.cheat_id} escaped hardened: {graded}"


# --------------------------------------------------------------------------- #
# Environment construction
# --------------------------------------------------------------------------- #
def test_dataset_has_one_row_per_task_and_embeds_the_buggy_source() -> None:
    ds = build_dataset()
    assert len(ds) == len(TASKS)
    for row, spec in zip(ds, TASKS):
        assert spec.target_file in row["question"]
        assert "```python" in row["question"]
        assert row["info"]["task_id"] == spec.task_id
        # the symptom must not leak the root cause
        assert "tril" not in row["question"].split("## File to fix")[0]


def _score_via_rubric(env, task_id: str, completion_text: str) -> dict:
    """Drive the real public scoring path with a synthetic rollout state.

    `score_rollout` mutates the state in place and returns None, so the caller
    reads `state["reward"]` and `state["metrics"]` afterwards.
    """
    state = {
        "input": {"info": {"task_id": task_id}},
        "task": "transformer-repair",
        "prompt": [{"role": "user", "content": "..."}],
        "completion": [{"role": "assistant", "content": completion_text}],
        "info": {"task_id": task_id},
        "metrics": {},
        "trajectory": [],
    }
    asyncio.run(env.rubric.score_rollout(state))
    return state


def test_environment_scores_gold_through_the_rubric() -> None:
    spec = TASKS[0]
    for scheme in ("naive", "hardened"):
        env = load_environment(reward_scheme=scheme)
        assert isinstance(env, vf.Environment)
        assert len(env.dataset) == len(TASKS)

        scored = _score_via_rubric(env, spec.task_id, _gold_diff_completion(spec.task_id))
        assert scored["reward"] == 1.0, f"gold scored {scored['reward']} under {scheme}"

        names = set(scored["metrics"])
        assert f"{scheme}_reward" in names, names
        assert {"exploit_gap_metric", "invalid_patch_metric"} <= names, names
        other = "hardened" if scheme == "naive" else "naive"
        assert f"{other}_reward_metric" in names, names


def test_reward_scheme_separates_a_cheating_submission() -> None:
    """The A/B in one assertion: same rollout, opposite reward under the two schemes."""
    cheat = next(c for c in CHEATS if not c.rewrite_files)
    completion = _cheat_diff_completion(cheat)

    naive_state = _score_via_rubric(load_environment(reward_scheme="naive"), cheat.task_id, completion)
    hardened_state = _score_via_rubric(
        load_environment(reward_scheme="hardened"), cheat.task_id, completion
    )

    assert naive_state["reward"] == 1.0
    assert hardened_state["reward"] == 0.0
    assert naive_state["metrics"]["exploit_gap_metric"] == 1.0


def test_unknown_reward_scheme_is_rejected() -> None:
    with pytest.raises(ValueError):
        load_environment(reward_scheme="fancy")
