"""Tensor-level primitives with no module state."""

from .masking import build_causal_mask, combine_masks
from .rope import apply_rope, build_rope_cache, rotate_half
from .scaling import attention_scale, masked_fill_penalty

__all__ = [
    "build_causal_mask",
    "combine_masks",
    "apply_rope",
    "build_rope_cache",
    "rotate_half",
    "attention_scale",
    "masked_fill_penalty",
]
