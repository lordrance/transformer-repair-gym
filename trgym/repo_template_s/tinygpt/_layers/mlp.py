"""Position-wise feed-forward network."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._core.settings import Config


class MLP(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        hidden = 4 * cfg.d_model
        self.up_proj = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.up_proj(x)))
