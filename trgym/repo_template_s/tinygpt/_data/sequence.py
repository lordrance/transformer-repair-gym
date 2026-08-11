"""Single-sequence construction."""

from __future__ import annotations

import torch

from .._core.settings import Config
from .vocab import CYCLES


def make_sequence(cfg: Config, length: int, generator: torch.Generator) -> list[int]:
    """A repeating cycle, entered at a random phase."""
    which = int(torch.randint(0, len(CYCLES), (1,), generator=generator).item())
    cycle = CYCLES[which]
    phase = int(torch.randint(0, len(cycle), (1,), generator=generator).item())
    return [cycle[(phase + i) % len(cycle)] for i in range(length)]
