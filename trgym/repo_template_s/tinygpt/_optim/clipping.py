"""Gradient clipping."""

from __future__ import annotations

import torch


def clip_gradients(model: torch.nn.Module, max_norm: float) -> float:
    """Clip in place and return the pre-clip global norm."""
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))
