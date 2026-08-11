"""The model and its objective."""

from __future__ import annotations

import torch

from ._layers.block import Block as Block
from ._layers.mlp import MLP as MLP
from ._model.gpt import TinyGPT as TinyGPT


def shifted_cross_entropy_sum(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Next-token cross entropy as a SUM, plus the supervised token count."""
    from ._model.objective import shifted_cross_entropy_sum as _impl

    return _impl(logits, labels, ignore_index)


__all__ = ["TinyGPT", "Block", "MLP", "shifted_cross_entropy_sum"]
