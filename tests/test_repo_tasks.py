"""Discrimination and hygiene tests for every repo-level task, Tier M and Tier H.

These run on every test invocation, which is the point: guardrail G5 (SWE-Universe)
warns against validating a generated task once at construction and never again. A
later edit that silently breaks gold/buggy discrimination fails here.
"""

from __future__ import annotations

import pytest

from trgym.repo.build import build_gold, build_repo, repo_fingerprint
from trgym.repo.checks import run_repo_checks
from trgym.tasks.repo_specs import REPO_TASKS
from trgym.tasks.repo_specs_h import HELD_OUT_TASK_IDS, REPO_TASKS_H

ALL_TASKS = REPO_TASKS + REPO_TASKS_H

# Anything that would turn a repo-level task back into a single-hop one.
LEAKS = (
    "attention.py", "positional.py", "optim.py", "data.py", "train.py",
    "model.py", "norm.py", "config.py",
    "tril", "rotate_half", "zero_grad", "ignore_index", "weight_decay",
    "rope_theta", "padding_mask", "scheduler.step",
)


@pytest.mark.parametrize("spec", ALL_TASKS, ids=lambda s: s.task_id)
def test_gold_passes_every_check(spec, tmp_path) -> None:
    ws = build_gold(spec, tmp_path / "gold")
    results = run_repo_checks(ws, spec.task_id, spec.visible_checks + spec.hidden_checks)
    failed = [(n, d) for n, ok, d in results if not ok]
    assert not failed, f"gold fails: {failed}"


@pytest.mark.parametrize("spec", ALL_TASKS, ids=lambda s: s.task_id)
def test_buggy_fails_hidden_but_passes_visible(spec, tmp_path) -> None:
    """The exploit gap: the bug must be invisible to the visible suite."""
    ws = build_repo(spec, tmp_path / "bug", gold=False)

    hidden = run_repo_checks(ws, spec.task_id, spec.hidden_checks)
    assert any(not ok for _, ok, _ in hidden), "buggy repo passes the hidden suite"

    visible = run_repo_checks(ws, spec.task_id, spec.visible_checks)
    failed = [n for n, ok, _ in visible if not ok]
    assert not failed, f"visible suite gives the bug away via {failed}"


@pytest.mark.parametrize("spec", ALL_TASKS, ids=lambda s: s.task_id)
def test_symptom_does_not_leak_location_or_cause(spec, tmp_path) -> None:
    ws = build_repo(spec, tmp_path / "bug", gold=False)
    symptom = (ws / "SYMPTOM.md").read_text(encoding="utf-8")
    # Only the part above the fixed boilerplate is task-authored prose.
    prose = symptom.split("## What you have")[0]
    for leak in LEAKS:
        assert leak not in prose, f"{spec.task_id} symptom leaks {leak!r}"


@pytest.mark.parametrize("spec", ALL_TASKS, ids=lambda s: s.task_id)
def test_mutations_actually_change_the_declared_files(spec, tmp_path) -> None:
    gold = build_gold(spec, tmp_path / "gold")
    bug = build_repo(spec, tmp_path / "bug", gold=False)
    g, b = repo_fingerprint(gold), repo_fingerprint(bug)
    changed = {k for k in g if g[k] != b.get(k)}
    assert changed == set(spec.mutations), (
        f"{spec.task_id}: changed {sorted(changed)}, declared {sorted(spec.mutations)}"
    )


@pytest.mark.parametrize("spec", REPO_TASKS_H, ids=lambda s: s.task_id)
def test_tier_h_has_at_least_two_defects(spec) -> None:
    """H is defined by interacting defects; a single-mutation H task is mislabelled."""
    total = sum(len(muts) for muts in spec.mutations.values())
    assert total >= 2, f"{spec.task_id} has only {total} mutation(s)"


def test_held_out_task_is_registered_and_real() -> None:
    ids = {s.task_id for s in REPO_TASKS_H}
    assert HELD_OUT_TASK_IDS, "no held-out task reserved"
    for held in HELD_OUT_TASK_IDS:
        assert held in ids, f"held-out id {held!r} is not a real task"


def test_task_count_is_within_the_session_cap() -> None:
    """Guardrail G6: structure over volume. Hard cap of 15 accepted tasks."""
    from trgym.tasks.registry import TASKS

    total = len(TASKS) + len(REPO_TASKS) + len(REPO_TASKS_H)
    assert total <= 15, f"{total} tasks exceeds the 15-task cap"


def test_partial_fix_of_a_tier_h_task_does_not_pass(tmp_path) -> None:
    """The property that defines Tier H: one defect fixed is not enough.

    Uses h2, whose two defects live in different files, and reverts only one.
    """
    from trgym.tasks.repo_specs_h import H2

    ws = build_repo(H2, tmp_path / "partial", gold=False)
    cfg = ws / "tinygpt" / "config.py"
    cfg.write_text(
        cfg.read_text(encoding="utf-8").replace(
            "    rope_theta: float = 500.0", "    rope_theta: float = 10000.0"
        ),
        encoding="utf-8",
    )
    results = run_repo_checks(ws, H2.task_id, H2.hidden_checks)
    assert any(not ok for _, ok, _ in results), (
        "reverting only one of two defects passed the hidden suite"
    )
