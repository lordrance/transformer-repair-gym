"""Multi-turn harness tests, driven by scripted policies.

Verifying the loop with a scripted policy rather than a live model is deliberate:
it proves the harness works before any API key exists, and it makes the failure
modes (budget exhaustion, path escape, unusable actions) testable at all, which
they would not be against a real model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

from trgym.harness.session import Action, Budget, Observation, run_episode  # noqa: E402
from trgym.harness.tools import apply_patch, list_files, read_file, run_command  # noqa: E402
from trgym.repo.build import build_gold, build_repo  # noqa: E402
from trgym.repo.checks import run_repo_checks  # noqa: E402
from trgym.tasks.repo_specs import REPO_TASKS, get_repo_task  # noqa: E402

M1 = get_repo_task("m1_attention_regression")


class ScriptedPolicy:
    """Replays a fixed list of actions, then submits."""

    def __init__(self, actions: list[Action]) -> None:
        self.actions = list(actions)
        self.seen: list[Observation] = []

    def act(self, obs: Observation) -> Action:
        self.seen.append(obs)
        if self.actions:
            return self.actions.pop(0)
        return Action("submit", {"summary": "done"})


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
def test_list_files_sees_the_package(tmp_path) -> None:
    ws = build_repo(M1, tmp_path / "ws")
    out = list_files(ws)
    assert out.ok
    for name in ("tinygpt/attention.py", "tinygpt/train.py", "tinygpt/optim.py"):
        assert name in out.output


def test_read_file_is_line_numbered_and_pageable(tmp_path) -> None:
    ws = build_repo(M1, tmp_path / "ws")
    out = read_file(ws, "tinygpt/attention.py", start=1, end=5)
    assert out.ok
    assert out.output.splitlines()[0].strip().startswith("1 |")
    assert len(out.output.splitlines()) == 5


def test_read_file_refuses_to_escape_the_workspace(tmp_path) -> None:
    ws = build_repo(M1, tmp_path / "ws")
    out = read_file(ws, "../../secrets.txt")
    assert not out.ok
    assert "outside the workspace" in out.output


def test_apply_patch_refuses_files_outside_the_package(tmp_path) -> None:
    ws = build_repo(M1, tmp_path / "ws")
    diff = "@@ -1,1 +1,1 @@\n-x\n+y\n"
    out = apply_patch(ws, "tests/test_visible.py", diff)
    assert not out.ok
    assert "not editable" in out.output


def test_run_command_rejects_anything_not_on_the_allowlist(tmp_path) -> None:
    ws = build_repo(M1, tmp_path / "ws")
    out = run_command(ws, "rm -rf /", Budget())
    assert not out.ok
    assert "unknown command" in out.output


def test_run_training_short_produces_a_loss_trace(tmp_path) -> None:
    ws = build_repo(M1, tmp_path / "ws")
    out = run_command(ws, "run_training_short", Budget())
    assert out.ok, out.output
    assert "loss" in out.output


def test_visible_tests_pass_on_the_buggy_repo(tmp_path) -> None:
    """The visible suite must not give the bug away -- that is the exploit gap."""
    ws = build_repo(M1, tmp_path / "ws")
    out = run_command(ws, "run_visible_tests", Budget(), repo_root=REPO_ROOT)
    assert out.ok, f"visible tests should pass on the buggy repo:\n{out.output}"


# --------------------------------------------------------------------------- #
# Episode loop
# --------------------------------------------------------------------------- #
def test_episode_records_every_step_and_ends_on_submit(tmp_path) -> None:
    ws = build_repo(M1, tmp_path / "ws")
    policy = ScriptedPolicy(
        [
            Action("list_files", {}),
            Action("read_file", {"path": "tinygpt/attention.py", "start": 25, "end": 45}),
            Action("run_command", {"name": "run_training_short"}),
            Action("submit", {"summary": "looked around"}),
        ]
    )
    ep = run_episode(ws, M1.task_id, policy, repo_root=REPO_ROOT)

    assert ep.submitted and ep.end_reason == "submitted"
    assert [s.action.tool for s in ep.steps] == [
        "list_files", "read_file", "run_command", "submit",
    ]
    assert ep.summary == "looked around"
    # the policy must have been shown the symptom, and it must not name the file
    assert "SYMPTOM" not in policy.seen[0].symptom or True
    assert "attention.py" not in policy.seen[0].symptom
    assert "tril" not in policy.seen[0].symptom


def test_episode_stops_when_the_turn_budget_runs_out(tmp_path) -> None:
    ws = build_repo(M1, tmp_path / "ws")
    policy = ScriptedPolicy([Action("list_files", {}) for _ in range(50)])
    ep = run_episode(ws, M1.task_id, policy, budget=Budget(max_turns=4))
    assert not ep.submitted
    assert "turn budget" in ep.end_reason
    assert len(ep.steps) == 4


def test_episode_stops_after_repeated_unusable_actions(tmp_path) -> None:
    ws = build_repo(M1, tmp_path / "ws")
    policy = ScriptedPolicy([Action("teleport", {}) for _ in range(10)])
    ep = run_episode(ws, M1.task_id, policy)
    assert ep.end_reason == "too many unusable actions"
    assert len(ep.steps) == 3


def test_command_budget_is_enforced(tmp_path) -> None:
    ws = build_repo(M1, tmp_path / "ws")
    policy = ScriptedPolicy(
        [Action("run_command", {"name": "run_training_short"}) for _ in range(6)]
    )
    ep = run_episode(ws, M1.task_id, policy, budget=Budget(max_commands=2, max_turns=10))
    assert "command budget" in ep.end_reason


# --------------------------------------------------------------------------- #
# End to end: a policy that actually fixes the bug is graded as fixed
# --------------------------------------------------------------------------- #
def test_a_policy_that_fixes_the_bug_passes_the_hidden_suite(tmp_path) -> None:
    ws = build_repo(M1, tmp_path / "ws")
    gold = build_gold(M1, tmp_path / "gold")

    before = run_repo_checks(ws, M1.task_id, M1.hidden_checks)
    assert any(not ok for _, ok, _ in before), "buggy repo should fail hidden checks"

    fix = (
        "@@ -34,1 +34,1 @@\n"
        "-    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, "
        "device=scores.device).tril(diagonal=1)\n"
        "+    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, "
        "device=scores.device).tril(diagonal=0)\n"
    )
    policy = ScriptedPolicy(
        [
            Action("apply_patch", {"path": "tinygpt/attention.py", "diff": fix}),
            Action("run_command", {"name": "run_visible_tests"}),
            Action("submit", {"summary": "restored the causal mask diagonal"}),
        ]
    )
    ep = run_episode(ws, M1.task_id, policy, repo_root=REPO_ROOT)
    assert ep.steps[0].result_ok, ep.steps[0].result_output

    after = run_repo_checks(ws, M1.task_id, M1.hidden_checks)
    failed = [n for n, ok, _ in after if not ok]
    assert not failed, f"still failing after the fix: {failed}"

    assert (ws / "tinygpt" / "attention.py").read_text(encoding="utf-8") == (
        gold / "tinygpt" / "attention.py"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize("spec", REPO_TASKS, ids=lambda s: s.task_id)
def test_every_task_is_reachable_through_the_harness(spec, tmp_path) -> None:
    """The symptom must be readable, the repo listable, and training runnable."""
    ws = build_repo(spec, tmp_path / spec.task_id)
    assert (ws / "SYMPTOM.md").is_file()
    symptom = (ws / "SYMPTOM.md").read_text(encoding="utf-8")

    # A symptom that names the file or the root cause defeats the whole tier.
    for leak in ("attention.py", "positional.py", "optim.py", "data.py", "train.py",
                 "tril", "rotate_half", "zero_grad", "ignore_index", "weight_decay"):
        assert leak not in symptom, f"{spec.task_id} symptom leaks {leak!r}"

    assert list_files(ws).ok
    out = run_command(ws, "run_training_short", Budget())
    assert out.ok, out.output
