"""The model and its objective."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention
from .config import Config
from .norm import RMSNorm
from .positional import build_rope_cache


class MLP(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        hidden = 4 * cfg.d_model
        self.up_proj = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.up_proj(x)))


class Block(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, padding_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x


def shifted_cross_entropy_sum(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Next-token cross entropy as a SUM, plus the supervised token count.

    Returning (sum, count) rather than a mean is what makes gradient
    accumulation exactly equivalent to full-batch training: the caller divides
    once, by the global count.
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


class TinyGPT(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
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
