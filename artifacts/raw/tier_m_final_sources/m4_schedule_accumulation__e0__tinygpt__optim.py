"""Optimizer and learning-rate schedule."""

from __future__ import annotations

import math

import torch

from .config import Config


def make_optimizer(model: torch.nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW with weight decay applied only to matmul weights.

    Norm gains and any 1-D parameter are excluded: decaying them pulls the
    normalization scale towards zero, which is not what weight decay is for.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
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


def make_scheduler(
    optimizer: torch.optim.Optimizer, cfg: Config
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay, expressed as a multiplier on the base lr."""

    def lr_lambda(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
