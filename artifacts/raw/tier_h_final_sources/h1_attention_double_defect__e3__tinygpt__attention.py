"""Causal self-attention."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import Config
from .positional import apply_rope


def causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention with an explicit causal mask.

    q, k, v      : (B, H, S, head_dim)
    padding_mask : (B, S) bool, True = real token. Padded *keys* must be
                   unattendable.

    A row that ends up fully masked (a padded query) would softmax over all
    -inf, so a large finite penalty is used rather than -inf: -inf rows produce
    NaN in the backward pass even when the forward value is later discarded.
    """
    head_dim = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

    seq_len = scores.shape[-1]
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device).tril()
    keep = causal.unsqueeze(0).unsqueeze(0)

    if padding_mask is not None:
        keep = keep & padding_mask[:, None, None, :]

    penalty = -1e4 if scores.dtype in (torch.float16, torch.bfloat16) else -1e9
    scores = scores.masked_fill(~keep, penalty)
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


class Attention(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        h, hd = self.cfg.n_head, self.cfg.head_dim

        q = self.q_proj(x).view(b, s, h, hd).transpose(1, 2)
        k = self.k_proj(x).view(b, s, h, hd).transpose(1, 2)
        v = self.v_proj(x).view(b, s, h, hd).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)
        out = causal_attention(q, k, v, padding_mask)
        out = out.transpose(1, 2).contiguous().view(b, s, self.cfg.d_model)
        return self.o_proj(out)
