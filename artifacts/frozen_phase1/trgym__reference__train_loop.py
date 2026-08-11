"""Pristine reference training step with gradient accumulation.

This is the gold implementation for task family F4. The invariant it must
satisfy is exact and cheaply checkable on CPU:

    accumulating gradients over K micro-batches
      ==  one backward pass over the concatenated full batch

That equivalence only holds if the cross-entropy is normalized by the *total*
number of supervised tokens across all micro-batches, not by re-averaging each
micro-batch independently. See REAL_BUG_EVIDENCE.md, family F4.
"""

from __future__ import annotations

from typing import Sequence

import torch

__all__ = ["MicroBatch", "accumulate_gradients", "full_batch_gradients"]


class MicroBatch:
    """A padded micro-batch. `labels` uses ignore_index for padding positions."""

    __slots__ = ("input_ids", "labels", "padding_mask")

    def __init__(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> None:
        self.input_ids = input_ids
        self.labels = labels
        self.padding_mask = padding_mask


def accumulate_gradients(model, micro_batches: Sequence[MicroBatch]) -> float:
    """Accumulate gradients over micro-batches and return the mean loss.

    Two passes are required. The first only counts supervised tokens so that the
    denominator is known before any backward call; the second does the actual
    backward with that fixed denominator.
    """
    model.zero_grad(set_to_none=True)

    total_tokens = 0
    with torch.no_grad():
        for mb in micro_batches:
            shift_labels = mb.labels[:, 1:]
            total_tokens += int((shift_labels != model.cfg.ignore_index).sum())

    if total_tokens == 0:
        return 0.0

    total_loss = 0.0
    for mb in micro_batches:
        loss_sum, _ = model.loss_sum(mb.input_ids, mb.labels, mb.padding_mask)
        # Normalize by the GLOBAL token count, not this micro-batch's count.
        (loss_sum / total_tokens).backward()
        total_loss += float(loss_sum.detach())

    return total_loss / total_tokens


def full_batch_gradients(model, micro_batches: Sequence[MicroBatch]) -> float:
    """Ground truth: one backward pass over everything at once.

    Used only by the verifier. Requires all micro-batches to share a sequence
    length, which the task's fixtures guarantee.
    """
    model.zero_grad(set_to_none=True)

    input_ids = torch.cat([mb.input_ids for mb in micro_batches], dim=0)
    labels = torch.cat([mb.labels for mb in micro_batches], dim=0)
    if any(mb.padding_mask is not None for mb in micro_batches):
        padding_mask = torch.cat(
            [
                mb.padding_mask
                if mb.padding_mask is not None
                else torch.ones_like(mb.input_ids, dtype=torch.bool)
                for mb in micro_batches
            ],
            dim=0,
        )
    else:
        padding_mask = None

    loss_sum, n_tokens = model.loss_sum(input_ids, labels, padding_mask)
    if int(n_tokens) == 0:
        return 0.0
    loss = loss_sum / n_tokens
    loss.backward()
    return float(loss.detach())
