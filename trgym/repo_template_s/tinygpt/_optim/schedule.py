"""Learning-rate schedule.

Linear warmup for `warmup_steps`, then cosine decay to zero over the remaining
budget. Expressed as a multiplier on the base lr so it composes with LambdaLR.
"""

from __future__ import annotations

import math

import torch

from .._core.settings import Config


def warmup_cosine(step: int, warmup_steps: int, total_steps: int) -> float:
    """The multiplier for `step`."""
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def make_scheduler(
    optimizer: torch.optim.Optimizer, cfg: Config
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        return warmup_cosine(step, cfg.warmup_steps, cfg.total_steps)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
