"""Task specification for the transformer-repair environment.

A task is *generated*, never hand-maintained: we take the pristine reference
file, apply one declared mutation, and ship the result as the workspace. The
gold patch is therefore always exactly the reference file, which removes a whole
class of "the gold answer drifted from the tests" bugs.

`provenance` records how close the injected bug is to a bug that actually
happened in the wild. See REAL_BUG_EVIDENCE.md.

    REAL          reproduces a specific documented bug essentially verbatim
    REAL-DERIVED  re-instantiates a bug pattern documented in real issues/PRs
                  inside our tiny model
    SYNTHETIC     invented mutation with no specific real-world referent
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

Provenance = Literal["REAL", "REAL-DERIVED", "SYNTHETIC"]


@dataclass(frozen=True)
class Mutation:
    """One exact, auditable source edit.

    Using literal string replacement rather than an AST rewrite keeps the diff
    reviewable by a human and makes it trivial to assert the mutation actually
    landed. `count` guards against a pattern silently matching twice.
    """

    find: str
    replace: str
    count: int = 1

    def apply(self, source: str) -> str:
        occurrences = source.count(self.find)
        if occurrences != self.count:
            raise ValueError(
                f"mutation expected {self.count} occurrence(s) of "
                f"{self.find[:60]!r}, found {occurrences}"
            )
        return source.replace(self.find, self.replace)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    family: str
    family_id: str
    provenance: Provenance
    target_file: str
    """Which file in the workspace the model must edit."""
    symptom: str
    """What a user would observe. Deliberately does not name the root cause."""
    evidence: tuple[str, ...]
    """Anchors into REAL_BUG_EVIDENCE.md / upstream issue URLs."""
    mutations: tuple[Mutation, ...]
    hidden_checks: tuple[str, ...]
    """Names of checks in trgym.verifier.hidden that must pass."""
    visible_checks: tuple[str, ...]
    """Names in trgym.verifier.visible. The 'reasonable but insufficient' suite.

    These are NOT a subset of hidden_checks. They test the same properties but at
    a single fixed public configuration -- see trgym/verifier/visible.py for why.
    """
    support_files: tuple[str, ...] = field(default_factory=tuple)
    """Extra reference files copied unmutated into the workspace."""

    def __post_init__(self) -> None:
        if not self.visible_checks:
            raise ValueError(f"{self.task_id}: needs at least one visible check")
        if not all(c.startswith("visible_") for c in self.visible_checks):
            raise ValueError(
                f"{self.task_id}: visible checks must come from trgym.verifier.visible"
            )
        if set(self.visible_checks) & set(self.hidden_checks):
            raise ValueError(
                f"{self.task_id}: visible and hidden suites must be distinct functions"
            )
