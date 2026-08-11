"""Verifier v2 check sets: v1 hidden checks plus the L1 contract layer.

Kept as a separate module so v1 remains byte-identical and frozen; the replay
compares the two rather than mutating one into the other.
"""

from __future__ import annotations

CONTRACT_CHECKS = ("repo_contract_return_types", "repo_contract_public_api")


def v1_checks(spec) -> tuple[str, ...]:
    return tuple(spec.hidden_checks)


def v2_checks(spec) -> tuple[str, ...]:
    return tuple(spec.hidden_checks) + CONTRACT_CHECKS
