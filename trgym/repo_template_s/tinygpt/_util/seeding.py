"""Deterministic generators.

Every fixture in this package draws from an explicitly seeded generator rather than
global RNG state, so a training run is reproducible regardless of import order.
"""

from __future__ import annotations

import torch


def seeded_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g
