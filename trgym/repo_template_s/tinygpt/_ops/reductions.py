"""Reductions used by the metrics layer."""

from __future__ import annotations

import torch


def global_norm(tensors) -> float:
    total = 0.0
    for t in tensors:
        if t is not None:
            total += float(t.detach().norm() ** 2)
    return total ** 0.5
