"""Attention masks.

`build_causal_mask` decides which key positions each query may attend to. A query at
position i may attend to keys 0..i inclusive and no further -- `tril(diagonal=0)`.
"""

from __future__ import annotations

import torch


def build_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """(S, S) bool. True where attention is permitted."""
    ones = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    return ones.tril(diagonal=0)


def combine_masks(
    causal: torch.Tensor, padding_mask: torch.Tensor | None
) -> torch.Tensor:
    """Broadcast the causal mask to (B, H, S, S) and intersect with padded keys."""
    keep = causal.unsqueeze(0).unsqueeze(0)
    if padding_mask is not None:
        keep = keep & padding_mask[:, None, None, :]
    return keep
