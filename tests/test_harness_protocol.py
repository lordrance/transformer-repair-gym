"""Regression tests for the two foundation defects found after the Tier H run.

Both are harness/protocol defects that produced a *scientific* misreading:
`h2_position_double_defect` was labelled 0/4 TOO_HARD when its episodes had in
fact been terminated by an adapter bug. These tests exist so that cannot recur
silently.

See PROTOCOL_CHANGELOG R5.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trgym.harness.session import (
    ADAPTER_NONACTIONS,
    Action,
    Observation,
    run_episode,
)
from trgym.harness.tools import Budget
from trgym.repo.build import build_repo
from trgym.tasks.repo_specs import get_repo_task

M1 = get_repo_task("m1_attention_regression")


class EmptyResponsePolicy:
    """Emits the adapter's empty-response non-action forever, then would submit.

    Models the observed DeepSeek failure: the provider returns a message with
    neither content nor a tool call.
    """

    def __init__(self, n_empty: int) -> None:
        self.n_empty = n_empty
        self.emitted = 0

    def act(self, obs: Observation) -> Action:
        if self.emitted < self.n_empty:
            self.emitted += 1
            return Action("empty_response", {"consecutive": self.emitted})
        return Action("list_files", {})


class InventedToolPolicy:
    """Emits tool names that do not exist -- a genuine model-side protocol failure."""

    def act(self, obs: Observation) -> Action:
        return Action("teleport", {})


# --------------------------------------------------------------------------- #
# Defect 1: adapter non-actions must not terminate the episode
# --------------------------------------------------------------------------- #
def test_adapter_nonactions_are_declared() -> None:
    """The set must cover every synthetic name the DeepSeek adapter can emit."""
    for name in ("api_error", "empty_response", "no_tool_call", "noop"):
        assert name in ADAPTER_NONACTIONS


def test_empty_responses_do_not_kill_the_episode(tmp_path) -> None:
    """Three consecutive empty provider responses used to end the run.

    That is what terminated both h2 episodes at turns 17 and 19 with nothing
    edited, and it was then misread as the task being too hard.
    """
    ws = build_repo(M1, tmp_path / "ws")
    policy = EmptyResponsePolicy(n_empty=5)
    ep = run_episode(ws, M1.task_id, policy, budget=Budget(max_turns=10))

    assert ep.end_reason != "too many unusable actions", (
        "adapter non-actions must not consume the unusable-action allowance"
    )
    assert "budget" in ep.end_reason
    # The empties are recorded in the trace rather than hidden.
    assert sum(1 for s in ep.steps if s.action.tool == "empty_response") == 5


def test_empty_responses_then_recovery_still_makes_progress(tmp_path) -> None:
    """After transient empties the policy must still be able to act."""
    ws = build_repo(M1, tmp_path / "ws")
    policy = EmptyResponsePolicy(n_empty=3)
    ep = run_episode(ws, M1.task_id, policy, budget=Budget(max_turns=8))

    tools = [s.action.tool for s in ep.steps]
    assert tools[:3] == ["empty_response"] * 3
    assert "list_files" in tools, "the episode must survive the empty run and continue"
    assert any(s.result_ok for s in ep.steps if s.action.tool == "list_files")


def test_genuinely_invented_tool_names_still_end_the_episode(tmp_path) -> None:
    """The unusable-action guard must still work for real model-side failures."""
    ws = build_repo(M1, tmp_path / "ws")
    ep = run_episode(ws, M1.task_id, InventedToolPolicy(), budget=Budget(max_turns=10))
    assert ep.end_reason == "too many unusable actions"
    assert len(ep.steps) == 3


# --------------------------------------------------------------------------- #
# Defect 2: evidence directories must never be used as scratch
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[1]

EVIDENCE_DIRS = (".sandbox_work", ".sandbox_work_h", ".sandbox_work_pro", ".sandbox_work_h24")


def _rmtree_targets(tree, source: str) -> set[str]:
    """String literals reachable as the first argument of a `rmtree` call.

    Resolves one level of module-level `NAME = <expr>` indirection, which is how
    every script in this repo names its scratch directory. Merely *reading* an
    evidence path elsewhere in the same file is not flagged -- only paths that are
    actually handed to rmtree.
    """
    import ast

    assigned: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                literals = [
                    n.value for n in ast.walk(node.value)
                    if isinstance(n, ast.Constant) and isinstance(n.value, str)
                ]
                if literals:
                    assigned[target.id] = " ".join(literals)

    targets: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
        if name != "rmtree" or not node.args:
            continue
        arg = node.args[0]
        for sub in ast.walk(arg):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                targets.add(sub.value)
            elif isinstance(sub, ast.Name) and sub.id in assigned:
                targets.add(assigned[sub.id])
    return targets


def test_no_script_passes_an_evidence_dir_to_rmtree() -> None:
    """`build_sandbox.py` rmtree'd `.sandbox_work` and destroyed Tier M's workspaces.

    The ban is on *deleting* an evidence directory, not on referencing one: the
    replay legitimately reads `.sandbox_work` while rmtree-ing its own scratch dir.
    """
    import ast

    offenders = []
    for script in sorted((REPO_ROOT / "scripts").glob("*.py")):
        text = script.read_text(encoding="utf-8")
        if "rmtree" not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for target in _rmtree_targets(tree, text):
            for ev in EVIDENCE_DIRS:
                if ev in target.split():
                    offenders.append(f"{script.name} passes {ev!r} to rmtree")
    assert not offenders, "; ".join(sorted(set(offenders)))


def test_the_rmtree_detector_actually_detects() -> None:
    """A guard is worthless if it cannot fail; prove it catches the original defect."""
    import ast

    bad = 'import shutil\nwork = Path("x") / ".sandbox_work"\nshutil.rmtree(work)\n'
    targets = _rmtree_targets(ast.parse(bad), bad)
    assert any(".sandbox_work" in t.split() for t in targets)

    good = (
        'import shutil\n'
        'POPULATION = [(".sandbox_work", "M")]\n'
        'scratch = Path("x") / ".my_scratch"\n'
        'shutil.rmtree(scratch)\n'
    )
    targets = _rmtree_targets(ast.parse(good), good)
    assert not any(".sandbox_work" in t.split() for t in targets), (
        "reading an evidence path must not be flagged"
    )


def test_evidence_dirs_are_in_the_g0_manifest() -> None:
    """G0 must index the graded workspaces, not only the JSONL and audits."""
    import json

    manifest_path = REPO_ROOT / "FROZEN_PHASE1_MANIFEST.json"
    if not manifest_path.exists():
        pytest.skip("G0 manifest not built yet")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    indexed = " ".join(e["path"] for e in manifest["artifacts"])
    assert "final_sources" in indexed, (
        "the preserved per-trajectory sources are the reconstruction source of "
        "record and must be indexed by G0"
    )
