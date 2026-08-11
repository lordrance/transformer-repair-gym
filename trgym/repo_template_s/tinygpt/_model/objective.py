"""The training objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def shifted_cross_entropy_sum(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Next-token cross entropy as a SUM, plus the supervised token count.

    Returning (sum, count) rather than a mean is what makes gradient accumulation
    exactly equivalent to full-batch training: the caller divides once, by the
    global count.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss_sum = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="sum",
    )
    n_tokens = (shift_labels != ignore_index).sum()
    return loss_sum, n_tokens
