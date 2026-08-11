"""Pristine reference implementation of a tiny decoder-only Transformer.

This file is the *gold* implementation for the transformer-repair environment.
Every task in `trgym.tasks` is produced by applying one documented mutation to a
copy of this file (or to `train_loop.py`), and "correct" is defined as
"numerically matches this file within tolerance".

Scope is deliberately limited to the parts that become RL tasks:

    token embedding -> [RMSNorm -> causal self-attention (RoPE) -> residual
                        -> RMSNorm -> MLP -> residual] * n_layer
                    -> RMSNorm -> LM head -> shifted cross-entropy

There is deliberately NO KV cache: that is inference machinery, not training
machinery, and it is reserved as a held-out task family for the flagship phase.

Everything runs on CPU in float32 (with an explicit bfloat16 path used by the
normalization-stability task).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "TinyGPTConfig",
    "rms_norm",
    "build_rope_cache",
    "rotate_half",
    "apply_rope",
    "causal_attention",
    "TinyGPT",
    "shifted_cross_entropy_sum",
]


@dataclass(frozen=True)
class TinyGPTConfig:
    vocab_size: int = 256
    d_model: int = 64
    n_head: int = 4
    n_layer: int = 2
    max_seq_len: int = 64
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    ignore_index: int = -100

    @property
    def head_dim(self) -> int:
        assert self.d_model % self.n_head == 0, "d_model must be divisible by n_head"
        return self.d_model // self.n_head


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Root-mean-square layer norm.

    The variance accumulation is performed in float32 regardless of the input
    dtype and only then cast back. This mirrors the reference LLaMA
    implementation in HuggingFace `transformers` and is what keeps the op stable
    under reduced precision -- see REAL_BUG_EVIDENCE.md, family F3.
    """
    input_dtype = x.dtype
    x32 = x.to(torch.float32)
    variance = x32.pow(2).mean(dim=-1, keepdim=True)
    x32 = x32 * torch.rsqrt(variance + eps)
    # Cast BOTH operands back, otherwise fp32 weight * bf16 activations silently
    # promotes the whole residual stream back to fp32 (HF issue #35945).
    return weight.to(input_dtype) * x32.to(input_dtype)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


# --------------------------------------------------------------------------- #
# Rotary position embeddings (GPT-NeoX / "rotate_half" convention)
# --------------------------------------------------------------------------- #
def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin), each of shape (seq_len, head_dim).

    Frequencies are computed for head_dim/2 pairs and then *duplicated* --
    `cat([freqs, freqs])` -- because `rotate_half` pairs dimension i with
    dimension i + head_dim/2.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)          # (S, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)           # (S, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate by pairing dim i with dim i + head_dim/2 (the "halves" convention)."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q and k of shape (B, H, S, head_dim)."""
    cos = cos.unsqueeze(0).unsqueeze(0)               # (1, 1, S, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out, k_out


# --------------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------------- #
def causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention with an explicit causal mask.

    q, k, v      : (B, H, S, head_dim)
    padding_mask : (B, S) bool, True = real token, False = padding. Padded
                   *keys* must be unattendable.

    Returns (B, H, S, head_dim).
    """
    head_dim = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)  # (B,H,S,S)

    seq_len = scores.shape[-1]
    # Lower-triangular INCLUDING the diagonal: position i may attend to j <= i.
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device).tril(diagonal=0)
    keep = causal.unsqueeze(0).unsqueeze(0)                               # (1,1,S,S)

    if padding_mask is not None:
        keep = keep & padding_mask[:, None, None, :]                     # mask keys

    neg_inf = torch.finfo(scores.dtype).min
    scores = scores.masked_fill(~keep, neg_inf)
    weights = torch.softmax(scores, dim=-1)
    # A fully-masked row (a padded query) yields a uniform row after softmax over
    # -inf; zero it so it cannot leak into the residual stream.
    if padding_mask is not None:
        weights = weights * padding_mask[:, None, :, None]
    return torch.matmul(weights, v)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class Attention(nn.Module):
    def __init__(self, cfg: TinyGPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        h, hd = self.cfg.n_head, self.cfg.head_dim

        q = self.q_proj(x).view(b, s, h, hd).transpose(1, 2)
        k = self.k_proj(x).view(b, s, h, hd).transpose(1, 2)
        v = self.v_proj(x).view(b, s, h, hd).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)
        out = causal_attention(q, k, v, padding_mask)
        out = out.transpose(1, 2).contiguous().view(b, s, self.cfg.d_model)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, cfg: TinyGPTConfig) -> None:
        super().__init__()
        hidden = 4 * cfg.d_model
        self.up_proj = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.up_proj(x)))


class Block(nn.Module):
    def __init__(self, cfg: TinyGPTConfig) -> None:
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
    """Next-token cross entropy, returned as a SUM plus the token count.

    Returning (sum, n_tokens) instead of a mean is what makes gradient
    accumulation exactly equivalent to full-batch training -- see
    REAL_BUG_EVIDENCE.md, family F4.

    Position t predicts token t+1, so logits are truncated on the right and
    labels on the left.
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
    def __init__(self, cfg: TinyGPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
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
