"""Tests for the unified-diff parser and applier.

The applier decides whether a rollout is INVALID, so its failure modes matter as
much as its successes: silently landing a hunk in the wrong place would corrupt
the reward signal worse than an honest rejection.
"""

from __future__ import annotations

import pytest

from trgym.patching import (
    PatchError,
    apply_unified_diff,
    extract_diff,
    last_fuzz,
    parse_unified_diff,
)
from trgym.tasks.build import REFERENCE_DIR, build_workspace
from trgym.tasks.registry import TASKS, get_task

ORIGINAL = "alpha\nbravo\ncharlie\ndelta\necho\n"


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def test_extract_prefers_a_block_containing_hunks() -> None:
    text = "```python\nprint(1)\n```\n```diff\n@@ -1,1 +1,1 @@\n-a\n+b\n```"
    assert "@@" in extract_diff(text)


def test_extract_falls_back_to_raw_text() -> None:
    raw = "@@ -1,1 +1,1 @@\n-alpha\n+ALPHA"
    assert extract_diff(raw) == raw


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def test_parse_reads_context_and_changes() -> None:
    diff = "@@ -2,3 +2,3 @@\n bravo\n-charlie\n+CHARLIE\n delta\n"
    (hunk,) = parse_unified_diff(diff)
    assert hunk.before == ("bravo", "charlie", "delta")
    assert hunk.after == ("bravo", "CHARLIE", "delta")


def test_parse_rejects_text_without_hunks() -> None:
    with pytest.raises(PatchError, match="no @@ hunk headers"):
        parse_unified_diff("I would change the tril diagonal to 0.")


def test_parse_rejects_garbage_inside_a_hunk() -> None:
    with pytest.raises(PatchError, match="unrecognised line"):
        parse_unified_diff("@@ -1,1 +1,1 @@\n-alpha\nthis is prose\n")


def test_parse_tolerates_no_newline_marker() -> None:
    diff = "@@ -5,1 +5,1 @@\n-echo\n+ECHO\n\\ No newline at end of file\n"
    (hunk,) = parse_unified_diff(diff)
    assert hunk.after == ("ECHO",)


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #
def test_apply_replaces_the_targeted_line() -> None:
    diff = "@@ -2,3 +2,3 @@\n bravo\n-charlie\n+CHARLIE\n delta\n"
    assert apply_unified_diff(ORIGINAL, diff) == "alpha\nbravo\nCHARLIE\ndelta\necho\n"


def test_apply_tolerates_wrong_line_numbers() -> None:
    """Models miscount lines constantly; correct context should still apply."""
    diff = "@@ -999,3 +999,3 @@\n bravo\n-charlie\n+CHARLIE\n delta\n"
    assert apply_unified_diff(ORIGINAL, diff) == "alpha\nbravo\nCHARLIE\ndelta\necho\n"


def test_apply_handles_multiple_hunks() -> None:
    diff = (
        "@@ -1,1 +1,1 @@\n-alpha\n+ALPHA\n"
        "@@ -5,1 +5,1 @@\n-echo\n+ECHO\n"
    )
    assert apply_unified_diff(ORIGINAL, diff) == "ALPHA\nbravo\ncharlie\ndelta\nECHO\n"


def test_apply_supports_pure_insertion() -> None:
    diff = "@@ -2,1 +2,2 @@\n bravo\n+bravo2\n"
    assert apply_unified_diff(ORIGINAL, diff) == "alpha\nbravo\nbravo2\ncharlie\ndelta\necho\n"


# --------------------------------------------------------------------------- #
# Fuzz: models paraphrase context, and refusing those patches would measure
# transcription accuracy rather than whether they understood the bug.
# --------------------------------------------------------------------------- #
def test_fuzz_tolerates_a_wrong_leading_context_line() -> None:
    """The model misquotes the docstring above the lines it is changing."""
    original = 'def f():\n    """Says "one" thing."""\n    return 1\n'
    diff = (
        "@@ -1,3 +1,3 @@\n"
        " def f():\n"
        "     \"\"\"Says 'one' thing.\"\"\"\n"   # paraphrased: wrong quote characters
        "-    return 1\n"
        "+    return 2\n"
    )
    assert apply_unified_diff(original, diff) == 'def f():\n    """Says "one" thing."""\n    return 2\n'
    assert last_fuzz() > 0


def test_fuzz_tolerates_omitted_context_lines() -> None:
    """The model skips the docstring entirely when quoting context."""
    original = 'def g():\n    """Doc."""\n    x = 1\n    return x\n'
    diff = "@@ -1,3 +1,3 @@\n def g():\n     x = 1\n-    return x\n+    return x + 1\n"
    out = apply_unified_diff(original, diff)
    assert out == 'def g():\n    """Doc."""\n    x = 1\n    return x + 1\n'


def test_exact_match_uses_no_fuzz() -> None:
    diff = "@@ -2,3 +2,3 @@\n bravo\n-charlie\n+CHARLIE\n delta\n"
    apply_unified_diff(ORIGINAL, diff)
    assert last_fuzz() == 0


def test_fuzz_does_not_rescue_a_wrong_removed_line() -> None:
    """Context may be paraphrased; the line being deleted may not."""
    diff = "@@ -2,3 +2,3 @@\n bravo\n-this line is not in the file\n+CHARLIE\n delta\n"
    with pytest.raises(PatchError):
        apply_unified_diff(ORIGINAL, diff)


def test_apply_rejects_context_that_does_not_exist() -> None:
    diff = "@@ -2,3 +2,3 @@\n foxtrot\n-golf\n+GOLF\n hotel\n"
    with pytest.raises(PatchError, match="context block not found"):
        apply_unified_diff(ORIGINAL, diff)


def test_apply_rejects_a_noop_patch() -> None:
    diff = "@@ -2,1 +2,1 @@\n bravo\n"
    with pytest.raises(PatchError, match="changes nothing"):
        apply_unified_diff(ORIGINAL, diff)


def test_ambiguous_context_resolves_to_the_declared_line() -> None:
    original = "x\nrepeat\ny\nrepeat\nz\n"
    diff = "@@ -4,1 +4,1 @@\n-repeat\n+REPEAT\n"
    assert apply_unified_diff(original, diff) == "x\nrepeat\ny\nREPEAT\nz\n"


def test_apply_preserves_absence_of_trailing_newline() -> None:
    original = "alpha\nbravo"
    diff = "@@ -1,1 +1,1 @@\n-alpha\n+ALPHA\n"
    assert apply_unified_diff(original, diff) == "ALPHA\nbravo"


# --------------------------------------------------------------------------- #
# Round trip against the real tasks: the gold patch must apply cleanly
# --------------------------------------------------------------------------- #
def _gold_diff(spec, tmp_path) -> str:
    """Build the true buggy->gold diff with difflib, as a model ideally would."""
    import difflib

    ws = build_workspace(spec, tmp_path / "ws", gold=False)
    buggy = (ws / spec.target_file).read_text(encoding="utf-8")
    gold = (REFERENCE_DIR / spec.target_file).read_text(encoding="utf-8")
    return "".join(
        difflib.unified_diff(
            buggy.splitlines(keepends=True),
            gold.splitlines(keepends=True),
            fromfile=spec.target_file,
            tofile=spec.target_file,
        )
    )


@pytest.mark.parametrize("spec", TASKS, ids=lambda s: s.task_id)
def test_gold_diff_applies_and_reproduces_the_reference(spec, tmp_path) -> None:
    ws = build_workspace(spec, tmp_path / "ws2", gold=False)
    buggy = (ws / spec.target_file).read_text(encoding="utf-8")
    gold = (REFERENCE_DIR / spec.target_file).read_text(encoding="utf-8")

    patched = apply_unified_diff(buggy, _gold_diff(spec, tmp_path))
    assert patched == gold, f"gold diff did not reproduce the reference for {spec.task_id}"


def test_gold_diff_is_far_smaller_than_the_whole_file(tmp_path) -> None:
    """The reason for the format change, asserted rather than assumed."""
    spec = get_task("t1_causal_mask_off_by_one")
    gold = (REFERENCE_DIR / spec.target_file).read_text(encoding="utf-8")
    diff = _gold_diff(spec, tmp_path)
    assert len(diff) < 0.2 * len(gold), (
        f"diff {len(diff)} chars vs whole file {len(gold)} chars"
    )
