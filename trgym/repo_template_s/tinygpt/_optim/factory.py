"""Optimizer construction."""

from __future__ import annotations

import torch

from .._core.registry import register
from .._core.settings import Config


@register("optimizer", "adamw")
def make_optimizer(model: torch.nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW with weight decay applied only to matmul weights.

    Norm gains and any 1-D parameter are excluded: decaying them pulls the
    normalization scale towards zero, which is not what weight decay is for.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay.append(param)
        else:
            no_decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
