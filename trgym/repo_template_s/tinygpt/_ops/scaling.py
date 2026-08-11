"""Score scaling and the masking penalty."""

from __future__ import annotations

import math

import torch


def attention_scale(head_dim: int) -> float:
    return 1.0 / math.sqrt(head_dim)


def masked_fill_penalty(dtype: torch.dtype) -> float:
    """A large finite penalty rather than -inf.

    A fully masked row (a padded query) would softmax over all -inf and produce NaN in
    the backward pass even when the forward value is discarded downstream.
    """
    return -1e4 if dtype in (torch.float16, torch.bfloat16) else -1e9
