"""Padding and label construction.

Sequences have different lengths on purpose: uneven padding is what exposes
loss-normalization and masking bugs, and a fixed-length corpus hides them.

Padded positions must carry `ignore_index` in `labels`, never `pad_token`. The
cross-entropy counts every label that is not `ignore_index`, so a padded position
written as a real token would be supervised as if the model were meant to predict
padding.
"""

from __future__ import annotations

import torch

from .._core.settings import Config
from .._core.types import Batch


def collate(cfg: Config, sequences: list[list[int]]) -> Batch:
    """Right-pad to the longest sequence in the batch."""
    width = max(len(s) for s in sequences)
    input_ids = torch.full((len(sequences), width), cfg.pad_token, dtype=torch.long)
    labels = torch.full((len(sequences), width), cfg.ignore_index, dtype=torch.long)
    mask = torch.zeros(len(sequences), width, dtype=torch.bool)

    for row, seq in enumerate(sequences):
        n = len(seq)
        input_ids[row, :n] = torch.tensor(seq, dtype=torch.long)
        labels[row, :n] = torch.tensor(seq, dtype=torch.long)
        mask[row, :n] = True

    return Batch(input_ids=input_ids, labels=labels, padding_mask=mask)
