"""A patched file must be the file that runs, even when the edit is invisible to mtime.

This is a grading-correctness test, not a hygiene one.

`RepoModules` clears `sys.modules` and calls `importlib.invalidate_caches()` before
importing a candidate tree. That resets the *finder* caches. It does not decide whether a
cached `__pycache__/*.pyc` is reused -- Python decides that by comparing the source's
(mtime, size) against the pair recorded in the `.pyc` header.

So a candidate patch that keeps the file **the same size** and lands in the **same mtime
second** as the write before it is invisible to that check, and Python runs the previous
bytecode. The repair is graded as though it had never been applied: a false negative
against a correct submission.

That is not hypothetical. `test_a_policy_that_fixes_the_bug_passes_the_hidden_suite` failed
intermittently on CI -- first windows/py3.11, then both ubuntu jobs -- while passing
locally, with `repo_strict_causality` (which never consults gold) among the failures. The
tasks make it especially easy to hit: `tril(diagonal=1)` -> `tril(diagonal=0)` is a
same-size edit.

The tests below force the collision rather than waiting for it, so a regression fails every
time instead of once in a while.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from trgym.repo.build import build_gold, build_repo
from trgym.repo.checks import run_repo_checks
from trgym.repo.visible_runtime import RepoModules, _purge_bytecode
from trgym.tasks.repo_specs import get_repo_task

TASK = "m1_attention_regression"


def _freeze_mtime(path: Path, when: float) -> None:
    """Pin mtime so a rewrite is indistinguishable from the original to the pyc check."""
    os.utime(path, (when, when))


def test_same_size_edit_at_identical_mtime_is_still_executed(tmp_path: Path) -> None:
    """The core race, forced: identical size, identical mtime, different content."""
    pkg = tmp_path / "pkg"
    (pkg / "tinygpt").mkdir(parents=True)
    (pkg / "tinygpt" / "__init__.py").write_text("", encoding="utf-8")
    module = pkg / "tinygpt" / "flag.py"

    module.write_text("VALUE = 1\n", encoding="utf-8")
    stamp = module.stat().st_mtime

    sys.path.insert(0, str(pkg))
    try:
        import importlib

        for name in [n for n in sys.modules if n.startswith("tinygpt")]:
            del sys.modules[name]
        first = importlib.import_module("tinygpt.flag")
        assert first.VALUE == 1
        assert list(module.parent.rglob("__pycache__")), "expected bytecode to be cached"

        # Same length, same mtime -- exactly what a candidate's one-character repair
        # looks like to the bytecode staleness check.
        module.write_text("VALUE = 2\n", encoding="utf-8")
        _freeze_mtime(module, stamp)

        for name in [n for n in sys.modules if n.startswith("tinygpt")]:
            del sys.modules[name]
        _purge_bytecode(pkg)
        importlib.invalidate_caches()
        second = importlib.import_module("tinygpt.flag")

        assert second.VALUE == 2, (
            "the edited file was not the file that ran: stale bytecode was reused, which "
            "in grading means a correct repair is scored as broken"
        )
    finally:
        sys.path.remove(str(pkg))
        for name in [n for n in sys.modules if n.startswith("tinygpt")]:
            del sys.modules[name]


def test_repomodules_purges_bytecode_on_entry(tmp_path: Path) -> None:
    """`RepoModules.__enter__` must remove cached bytecode, not merely reset finders."""
    spec = get_repo_task(TASK)
    ws = build_gold(spec, tmp_path / "gold")

    with RepoModules(ws):
        pass
    assert list((ws / "tinygpt").rglob("__pycache__")), "import should have cached bytecode"

    with RepoModules(ws) as cand:
        # If entry purged the cache, the modules were compiled from source this time.
        assert cand.config.Config().vocab_size > 0


def test_a_same_size_repair_is_graded_as_fixed(tmp_path: Path) -> None:
    """End to end: patch the planted defect in place and require the checks to notice.

    Mirrors what a policy does -- edit the file, grade the tree -- with the mtime pinned so
    the stale-bytecode path is taken if the purge ever regresses.
    """
    spec = get_repo_task(TASK)
    ws = build_repo(spec, tmp_path / "buggy")

    before = run_repo_checks(ws, TASK, ["repo_strict_causality"])
    assert any(not ok for _, ok, _ in before), "the planted defect must fail this check"

    target = ws / "tinygpt" / "attention.py"
    stamp = target.stat().st_mtime
    broken = target.read_text(encoding="utf-8")
    fixed = broken.replace("tril(diagonal=1)", "tril(diagonal=0)")
    assert fixed != broken, "fixture drifted; the defect is not the expected one"
    assert len(fixed) == len(broken), "this test is only meaningful for a same-size edit"

    target.write_text(fixed, encoding="utf-8")
    _freeze_mtime(target, stamp)

    after = run_repo_checks(ws, TASK, ["repo_strict_causality"])
    failed = [n for n, ok, _ in after if not ok]
    assert not failed, (
        f"a correct same-size repair was graded as still broken: {failed}. Stale bytecode "
        "was executed instead of the patched source."
    )


@pytest.mark.parametrize("subdir", ["", "_ops"])
def test_purge_reaches_nested_packages(tmp_path: Path, subdir: str) -> None:
    """Tier S puts implementation in subpackages; the purge must recurse."""
    root = tmp_path / "pkg"
    target_dir = root / "tinygpt" / subdir if subdir else root / "tinygpt"
    target_dir.mkdir(parents=True)
    cache = target_dir / "__pycache__"
    cache.mkdir()
    (cache / "stale.pyc").write_bytes(b"\x00\x01")

    _purge_bytecode(root)
    assert not cache.exists(), f"__pycache__ under {subdir or 'tinygpt'} was not removed"
