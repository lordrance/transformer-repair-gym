"""Normalization layers."""

from __future__ import annotations

import torch

from ._layers.rmsnorm import RMSNorm as RMSNorm


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Root-mean-square layer norm."""
    from ._layers.rmsnorm import rms_norm as _impl

    return _impl(x, weight, eps)


__all__ = ["RMSNorm", "rms_norm"]
