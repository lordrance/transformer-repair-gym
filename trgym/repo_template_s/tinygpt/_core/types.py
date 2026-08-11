"""Shared data containers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Batch:
    input_ids: torch.Tensor      # (B, S) int64
    labels: torch.Tensor         # (B, S) int64, ignore_index at padding
    padding_mask: torch.Tensor   # (B, S) bool, True = real token

    @property
    def n_supervised(self) -> int:
        return int((self.labels[:, 1:] != -100).sum())
