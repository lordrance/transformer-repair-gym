"""Rotary position embeddings."""

from __future__ import annotations

import torch


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin), each of shape (seq_len, head_dim)."""
    from ._ops.rope import build_rope_cache as _impl

    return _impl(seq_len, head_dim, theta, device, dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate by pairing dim i with dim i + head_dim/2."""
    from ._ops.rope import rotate_half as _impl

    return _impl(x)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q and k of shape (B, H, S, head_dim)."""
    from ._ops.rope import apply_rope as _impl

    return _impl(q, k, cos, sin)
