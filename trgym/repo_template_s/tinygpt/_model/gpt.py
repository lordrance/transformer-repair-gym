"""TinyGPT."""

from __future__ import annotations

import torch
import torch.nn as nn

from .._core.settings import Config
from .._layers.block import Block
from .._layers.embedding import TokenEmbedding
from .._layers.rmsnorm import RMSNorm
from .._ops.rope import build_rope_cache
from .objective import shifted_cross_entropy_sum


class TinyGPT(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = TokenEmbedding(cfg).embed
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(
        self, input_ids: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, s = input_ids.shape
        assert s <= self.cfg.max_seq_len, f"sequence length {s} exceeds max_seq_len"
        x = self.embed(input_ids)
        cos, sin = build_rope_cache(
            s, self.cfg.head_dim, self.cfg.rope_theta, device=x.device, dtype=x.dtype
        )
        for block in self.blocks:
            x = block(x, cos, sin, padding_mask)
        x = self.final_norm(x)
        return self.lm_head(x)

    def loss_sum(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(input_ids, padding_mask)
        return shifted_cross_entropy_sum(logits, labels, self.cfg.ignore_index)
