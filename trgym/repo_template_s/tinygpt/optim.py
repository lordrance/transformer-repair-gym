"""Optimizer and learning-rate schedule."""

from __future__ import annotations

import torch

from ._core.settings import Config


def make_optimizer(model: torch.nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW with weight decay applied only to matmul weights."""
    from ._optim.factory import make_optimizer as _impl

    return _impl(model, cfg)


def make_scheduler(
    optimizer: torch.optim.Optimizer, cfg: Config
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay."""
    from ._optim.schedule import make_scheduler as _impl

    return _impl(optimizer, cfg)
