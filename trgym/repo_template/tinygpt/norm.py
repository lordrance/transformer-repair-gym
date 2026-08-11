"""Normalization layers."""

from __future__ import annotations

import torch
import torch.nn as nn


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Root-mean-square layer norm.

    The variance is accumulated in float32 regardless of the input dtype and only
    then cast back. Under reduced precision the squares can otherwise overflow.
    """
    input_dtype = x.dtype
    x32 = x.to(torch.float32)
    variance = x32.pow(2).mean(dim=-1, keepdim=True)
    x32 = x32 * torch.rsqrt(variance + eps)
    return weight.to(input_dtype) * x32.to(input_dtype)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)
