"""Causal self-attention, assembled from the primitives in `_ops`."""

from __future__ import annotations

import torch
import torch.nn as nn

from .._core.settings import Config
from .._ops.masking import build_causal_mask, combine_masks
from .._ops.rope import apply_rope
from .._ops.scaling import attention_scale, masked_fill_penalty


def causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention with an explicit causal mask.

    q, k, v      : (B, H, S, head_dim)
    padding_mask : (B, S) bool, True = real token.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) * attention_scale(q.shape[-1])

    causal = build_causal_mask(scores.shape[-1], device=scores.device)
    keep = combine_masks(causal, padding_mask)

    scores = scores.masked_fill(~keep, masked_fill_penalty(scores.dtype))
    weights = torch.softmax(scores, dim=-1)
    if padding_mask is not None:
        weights = weights * padding_mask[:, None, :, None]
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
