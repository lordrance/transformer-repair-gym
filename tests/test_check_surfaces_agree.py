"""The two check surfaces must not drift apart.

R16 left this project with two implementations of the same decision:

  * `trgym/repo/checks.py`      -- in-process. Used by the gold preflight, the
    `fallback=True` development path, and most of the host test suite.
  * `trgym/repo/predicates.py`  -- the trusted comparator. This is what production
    grading actually decides with, because every real grade goes through the sandboxed
    path.

Duplication was the price of the boundary: the predicate has to live where gold lives,
and the in-process surface predates it and still has legitimate callers. The danger is
that they diverge, because then the host suite can be green while production grades
differently -- the same shape as R11, where the verifier was vacuously green and only a
gold-vs-no-op separation assertion caught it.

These tests do not check that the two implementations compute identical answers; that
would require running both, and the whole point of the boundary is that only one of them
may see candidate code. They check the cheaper invariant that actually catches drift:
the two surfaces must offer the same set of named checks.
"""

from __future__ import annotations

import pytest

from trgym.repo import predicates
from trgym.repo.checks import CHECKS, LEVELS


def test_every_named_check_has_a_predicate() -> None:
    """A check the in-process surface knows about must be decidable in the sandbox.

    A name in `LEVELS` with no predicate means a task can request a check that the
    production grader silently cannot evaluate.
    """
    missing = sorted(set(LEVELS) - set(predicates.PREDICATES))
    assert not missing, (
        f"checks with no trusted-side predicate: {missing}. Production grading would "
        "report these as 'unknown check' while the host suite still passes."
    )


def test_every_predicate_is_a_known_check() -> None:
    """And the reverse: a predicate for a check nothing declares is dead weight."""
    extra = sorted(set(predicates.PREDICATES) - set(LEVELS))
    assert not extra, f"predicates for undeclared checks: {extra}"


def test_visible_checks_are_covered_too() -> None:
    """The visible suite is graded through the same path and must not be forgotten."""
    visible = {n for n in LEVELS if n.startswith("repo_visible_")}
    assert visible, "the visible suite disappeared from LEVELS"
    missing = sorted(visible - set(predicates.PREDICATES))
    assert not missing, f"visible checks with no predicate: {missing}"


def test_in_process_surface_still_implements_everything() -> None:
    """`CHECKS` backs the gold preflight; a gap there breaks gold verification."""
    missing = sorted(set(LEVELS) - set(CHECKS))
    assert not missing, f"declared checks absent from the in-process registry: {missing}"


@pytest.mark.parametrize("name", sorted(LEVELS))
def test_each_check_is_reachable_from_both_surfaces(name: str) -> None:
    """Per-check, so a failure names the offender rather than a set difference."""
    assert name in CHECKS, f"{name} missing from checks.CHECKS"
    assert name in predicates.PREDICATES, f"{name} missing from predicates.PREDICATES"


def test_every_predicate_is_classified_exactly_once() -> None:
    """`FORGEABLE` and `GOLD_ANCHORED` must partition the predicate set.

    Both sets are quoted in SECURITY_MODEL.md. An unclassified predicate makes that
    document incomplete; a doubly-classified one makes it contradictory. Overstating the
    guarantee is the worse direction, so this fails loudly either way.
    """
    both = sorted(predicates.FORGEABLE & predicates.GOLD_ANCHORED)
    assert not both, f"checks claimed as both forgeable and gold-anchored: {both}"

    classified = predicates.FORGEABLE | predicates.GOLD_ANCHORED
    unclassified = sorted(set(predicates.PREDICATES) - classified)
    assert not unclassified, (
        f"unclassified predicates: {unclassified}. Every check must be declared either "
        "forgeable or gold-anchored, or SECURITY_MODEL.md's claim is incomplete."
    )
    stale = sorted(classified - set(predicates.PREDICATES))
    assert not stale, f"classified names that are not predicates: {stale}"


def test_the_four_original_gold_checks_remain_anchored() -> None:
    """The checks that were unforgeable before v0.2-B must not have regressed."""
    original = {
        "repo_matches_gold_logits",
        "repo_supervised_token_count",
        "repo_lr_schedule_matches_gold",
        "repo_contract_public_api",
    }
    lost = sorted(original - predicates.GOLD_ANCHORED)
    assert not lost, f"previously gold-backed checks are no longer anchored: {lost}"


def test_v02b_conversions_are_anchored_and_have_a_gold_fixture() -> None:
    """The five checks v0.2-B converted must be anchored AND actually compare to gold.

    Declaring a check gold-anchored without the predicate consulting gold would be a
    documentation claim rather than a defence -- exactly the failure R15's denylist and
    the dead contract checks both represented.
    """
    import inspect

    converted = {
        "repo_padding_keys_masked": "_pred_padding_keys_masked",
        "repo_rope_relative_property": "_pred_rope_relative",
        "repo_rope_norm_preserved": "_pred_rope_norm",
        "repo_visible_single_token_attention": "_visible_single",
        "repo_visible_rope_position_zero": "_visible_rope0",
    }
    for check, fn_name in converted.items():
        assert check in predicates.GOLD_ANCHORED, f"{check} is not declared anchored"
        assert check not in predicates.FORGEABLE, f"{check} is still declared forgeable"
        src = inspect.getsource(getattr(predicates, fn_name))
        assert "_gold_pure_outputs" in src, (
            f"{fn_name} claims to be gold-anchored but never consults gold"
        )
