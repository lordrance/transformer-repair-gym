"""The synthetic symbol cycles.

A fixed set of disjoint cycles. Every symbol belongs to exactly one cycle and therefore
has exactly one legal successor, which makes next-token prediction a deterministic
function of the current token and thus learnable in a few dozen CPU steps.
"""

from __future__ import annotations

CYCLES: tuple[tuple[int, ...], ...] = (
    (11, 12, 13),
    (21, 22, 23, 24),
    (31, 32),
    (41, 42, 43, 44, 45),
)


def successor(symbol: int) -> int | None:
    """The single legal next symbol, or None if the symbol is not in the corpus."""
    for cycle in CYCLES:
        if symbol in cycle:
            return cycle[(cycle.index(symbol) + 1) % len(cycle)]
    return None
