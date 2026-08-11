"""Deterministic batch construction."""

from __future__ import annotations

import torch

from .._core.settings import Config
from .._core.types import Batch
from .._util.seeding import seeded_generator
from .collate import collate
from .sequence import make_sequence


def make_batches(cfg: Config, n_batches: int, seed: int) -> list[Batch]:
    """Deterministic list of batches with uneven sequence lengths."""
    generator = seeded_generator(seed)
    batches = []
    for _ in range(n_batches):
        lengths = torch.randint(
            8, cfg.max_seq_len // 2, (cfg.micro_batch_size,), generator=generator
        ).tolist()
        batches.append(collate(cfg, [make_sequence(cfg, n, generator) for n in lengths]))
    return batches
