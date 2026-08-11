"""Gradient accumulation."""

from __future__ import annotations

from .._core.types import Batch


def accumulate_gradients(model, micro_batches: list[Batch]) -> float:
    """Accumulate over micro-batches; returns the mean loss per supervised token.

    Two passes. The first only counts supervised tokens so the denominator is fixed
    before any backward call; the second does the backward with that global
    denominator. Normalizing per micro-batch instead would make the result depend on
    how the data happened to be split.
    """
    model.zero_grad(set_to_none=True)

    total_tokens = sum(mb.n_supervised for mb in micro_batches)
    if total_tokens == 0:
        return 0.0

    total_loss = 0.0
    for mb in micro_batches:
        loss_sum, _ = model.loss_sum(mb.input_ids, mb.labels, mb.padding_mask)
        (loss_sum / total_tokens).backward()
        total_loss += float(loss_sum.detach())
    return total_loss / total_tokens
