"""Loss-curve diagnostics."""

from __future__ import annotations


def loss_slope(losses: list[float], window: int = 5) -> float:
    """Mean change per step over the last `window` steps. Negative means learning."""
    tail = [float(v) for v in losses[-window:]]
    if len(tail) < 2:
        return 0.0
    return (tail[-1] - tail[0]) / (len(tail) - 1)
