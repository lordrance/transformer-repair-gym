"""Synthetic dataset, padding and label construction.

The task is deliberately learnable in a few dozen CPU steps: each sequence is a
repeating short cycle, so a working model drops well below ln(vocab_size) quickly
and a broken one does not.

Sequences have *different* lengths on purpose. Uneven padding is what exposes
loss-normalization and masking bugs; a fixed-length dataset hides them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import Config


@dataclass
class Batch:
    input_ids: torch.Tensor      # (B, S) int64
    labels: torch.Tensor         # (B, S) int64, ignore_index at padding
    padding_mask: torch.Tensor   # (B, S) bool, True = real token

    @property
    def n_supervised(self) -> int:
        return int((self.labels[:, 1:] != -100).sum())


# A fixed set of disjoint cycles. Every symbol belongs to exactly one cycle and
# therefore has exactly one legal successor, which makes next-token prediction a
# deterministic function of the current token and thus learnable in a few dozen
# CPU steps. With random symbols per sequence there is nothing consistent to
# learn, the loss sits at ln(vocab_size), and the L3 behavioural check loses all
# of its signal.
CYCLES: tuple[tuple[int, ...], ...] = (
    (11, 12, 13),
    (21, 22, 23, 24),
    (31, 32),
    (41, 42, 43, 44, 45),
)


def make_sequence(cfg: Config, length: int, generator: torch.Generator) -> list[int]:
    """A repeating cycle, entered at a random phase."""
    which = int(torch.randint(0, len(CYCLES), (1,), generator=generator).item())
    cycle = CYCLES[which]
    phase = int(torch.randint(0, len(cycle), (1,), generator=generator).item())
    return [cycle[(phase + i) % len(cycle)] for i in range(length)]


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


def make_batches(cfg: Config, n_batches: int, seed: int) -> list[Batch]:
    """Deterministic list of batches with uneven sequence lengths."""
    generator = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n_batches):
        lengths = torch.randint(
            8, cfg.max_seq_len // 2, (cfg.micro_batch_size,), generator=generator
        ).tolist()
        batches.append(collate(cfg, [make_sequence(cfg, n, generator) for n in lengths]))
    return batches
