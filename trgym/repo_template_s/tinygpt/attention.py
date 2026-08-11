"""Causal self-attention."""

from __future__ import annotations

import torch

from ._layers.attention import Attention as Attention


def causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention with an explicit causal mask."""
    from ._layers.attention import causal_attention as _impl

    return _impl(q, k, v, padding_mask)


__all__ = ["Attention", "causal_attention"]
