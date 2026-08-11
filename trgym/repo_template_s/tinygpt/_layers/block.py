"""One pre-norm transformer block."""

from __future__ import annotations

import torch
import torch.nn as nn

from .._core.settings import Config
from .attention import Attention
from .mlp import MLP
from .rmsnorm import RMSNorm


class Block(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, padding_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x
