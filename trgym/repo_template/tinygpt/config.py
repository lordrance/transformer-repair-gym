"""Model and training configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # model
    vocab_size: int = 256
    d_model: int = 64
    n_head: int = 4
    n_layer: int = 2
    max_seq_len: int = 64
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6

    # data
    pad_token: int = 0
    ignore_index: int = -100

    # training
    lr: float = 3e-3
    weight_decay: float = 0.01
    warmup_steps: int = 5
    total_steps: int = 40
    grad_clip: float = 1.0
    micro_batch_size: int = 4
    grad_accum_steps: int = 2
    seed: int = 0

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"
        return self.d_model // self.n_head
