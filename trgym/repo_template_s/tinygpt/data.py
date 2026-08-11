"""Synthetic dataset, padding and label construction."""

from __future__ import annotations

import torch

from ._core.types import Batch as Batch
from ._data.vocab import CYCLES as CYCLES
from ._core.settings import Config


def make_sequence(cfg: Config, length: int, generator: torch.Generator) -> list[int]:
    """A repeating cycle, entered at a random phase."""
    from ._data.sequence import make_sequence as _impl

    return _impl(cfg, length, generator)


def collate(cfg: Config, sequences: list[list[int]]) -> Batch:
    """Right-pad to the longest sequence in the batch."""
    from ._data.collate import collate as _impl

    return _impl(cfg, sequences)


def make_batches(cfg: Config, n_batches: int, seed: int) -> list[Batch]:
    """Deterministic list of batches with uneven sequence lengths."""
    from ._data.batching import make_batches as _impl

    return _impl(cfg, n_batches, seed)
