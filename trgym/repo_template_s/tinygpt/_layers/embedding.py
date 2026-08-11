"""Token embedding table."""

from __future__ import annotations

import torch
import torch.nn as nn

from .._core.settings import Config


class TokenEmbedding(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)
