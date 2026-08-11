"""Generate the Tier S repo template: same public API, far larger search space.

G4 asks whether an agent can *localize* a defect in a repository it cannot read
exhaustively inside its turn budget. Tier M gives it 8 files; Tier S gives it 40+.

The design constraint that matters is the contract's `forbidden: dummy_001.py padding`.
Every file here does real work and is imported by something -- `scripts/tier_s_freeze.py`
verifies that from the import graph rather than taking this docstring's word for it.

The trick that keeps this affordable: `tinygpt`'s eight public modules stay exactly where
they were, as thin delegating facades over subpackages that hold the implementation. So

  * the existing hidden checks grade Tier S unchanged -- gold-PASS / no-op-FAIL is
    inherited from a suite that has already been verified, not re-derived; and
  * a planted defect can live in `tinygpt/_ops/masking.py`, four levels from the symptom,
    while `tinygpt/attention.py` reads as innocent boilerplate.

Run: python scripts/build_tier_s_template.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "trgym" / "repo_template_s" / "tinygpt"

H = '"""'

FILES: dict[str, str] = {}


def add(rel: str, body: str) -> None:
    FILES[rel] = body.lstrip("\n")


# --------------------------------------------------------------------------- #
# _core
# --------------------------------------------------------------------------- #
add("_core/__init__.py", '''
"""Configuration, shared types and the component registry."""

from .settings import Config
from .types import Batch

__all__ = ["Config", "Batch"]
''')

add("_core/settings.py", '''
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
''')

add("_core/types.py", '''
"""Shared data containers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Batch:
    input_ids: torch.Tensor      # (B, S) int64
    labels: torch.Tensor         # (B, S) int64, ignore_index at padding
    padding_mask: torch.Tensor   # (B, S) bool, True = real token

    @property
    def n_supervised(self) -> int:
        return int((self.labels[:, 1:] != -100).sum())
''')

add("_core/registry.py", '''
"""A tiny name -> factory registry.

Used by the CLI to select a component without importing every implementation at
start-up. Small, but real: `_train.loop` resolves the optimizer factory through it.
"""

from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")

_REGISTRY: dict[str, dict[str, Callable]] = {}


def register(kind: str, name: str) -> Callable[[T], T]:
    def deco(fn: T) -> T:
        _REGISTRY.setdefault(kind, {})[name] = fn
        return fn

    return deco


def resolve(kind: str, name: str) -> Callable:
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        raise KeyError(f"no {kind!r} registered under {name!r}")


def available(kind: str) -> list[str]:
    return sorted(_REGISTRY.get(kind, {}))
''')

# --------------------------------------------------------------------------- #
# _util
# --------------------------------------------------------------------------- #
add("_util/__init__.py", '''
"""Cross-cutting helpers."""

from .seeding import seeded_generator

__all__ = ["seeded_generator"]
''')

add("_util/seeding.py", '''
"""Deterministic generators.

Every fixture in this package draws from an explicitly seeded generator rather than
global RNG state, so a training run is reproducible regardless of import order.
"""

from __future__ import annotations

import torch


def seeded_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g
''')

add("_util/reporting.py", '''
"""Console formatting for the training loop."""

from __future__ import annotations


def format_step(step: int, loss: float, lr: float, grad_norm: float) -> str:
    return f"step {step:3d}  loss {loss:.4f}  lr {lr:.2e}  grad_norm {grad_norm:.3f}"
''')

# --------------------------------------------------------------------------- #
# _ops
# --------------------------------------------------------------------------- #
add("_ops/__init__.py", '''
"""Tensor-level primitives with no module state."""

from .masking import build_causal_mask, combine_masks
from .rope import apply_rope, build_rope_cache, rotate_half
from .scaling import attention_scale, masked_fill_penalty

__all__ = [
    "build_causal_mask",
    "combine_masks",
    "apply_rope",
    "build_rope_cache",
    "rotate_half",
    "attention_scale",
    "masked_fill_penalty",
]
''')

add("_ops/masking.py", '''
"""Attention masks.

`build_causal_mask` decides which key positions each query may attend to. A query at
position i may attend to keys 0..i inclusive and no further -- `tril(diagonal=0)`.
"""

from __future__ import annotations

import torch


def build_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """(S, S) bool. True where attention is permitted."""
    ones = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
    return ones.tril(diagonal=0)


def combine_masks(
    causal: torch.Tensor, padding_mask: torch.Tensor | None
) -> torch.Tensor:
    """Broadcast the causal mask to (B, H, S, S) and intersect with padded keys."""
    keep = causal.unsqueeze(0).unsqueeze(0)
    if padding_mask is not None:
        keep = keep & padding_mask[:, None, None, :]
    return keep
''')

add("_ops/rope.py", '''
"""Rotary position embeddings (GPT-NeoX / "rotate_half" convention)."""

from __future__ import annotations

import torch


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin), each of shape (seq_len, head_dim)."""
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate by pairing dim i with dim i + head_dim/2."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q and k of shape (B, H, S, head_dim)."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
''')

add("_ops/scaling.py", '''
"""Score scaling and the masking penalty."""

from __future__ import annotations

import math

import torch


def attention_scale(head_dim: int) -> float:
    return 1.0 / math.sqrt(head_dim)


def masked_fill_penalty(dtype: torch.dtype) -> float:
    """A large finite penalty rather than -inf.

    A fully masked row (a padded query) would softmax over all -inf and produce NaN in
    the backward pass even when the forward value is discarded downstream.
    """
    return -1e4 if dtype in (torch.float16, torch.bfloat16) else -1e9
''')

add("_ops/reductions.py", '''
"""Reductions used by the metrics layer."""

from __future__ import annotations

import torch


def global_norm(tensors) -> float:
    total = 0.0
    for t in tensors:
        if t is not None:
            total += float(t.detach().norm() ** 2)
    return total ** 0.5
''')

# --------------------------------------------------------------------------- #
# _layers
# --------------------------------------------------------------------------- #
add("_layers/__init__.py", '''
"""Neural network modules."""

from .attention import Attention, causal_attention
from .block import Block
from .embedding import TokenEmbedding
from .mlp import MLP
from .rmsnorm import RMSNorm, rms_norm

__all__ = [
    "Attention",
    "causal_attention",
    "Block",
    "TokenEmbedding",
    "MLP",
    "RMSNorm",
    "rms_norm",
]
''')

add("_layers/rmsnorm.py", '''
"""Normalization layers."""

from __future__ import annotations

import torch
import torch.nn as nn


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Root-mean-square layer norm, accumulated in float32."""
    input_dtype = x.dtype
    x32 = x.to(torch.float32)
    variance = x32.pow(2).mean(dim=-1, keepdim=True)
    x32 = x32 * torch.rsqrt(variance + eps)
    return weight.to(input_dtype) * x32.to(input_dtype)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)
''')

add("_layers/attention.py", '''
"""Causal self-attention, assembled from the primitives in `_ops`."""

from __future__ import annotations

import torch
import torch.nn as nn

from .._core.settings import Config
from .._ops.masking import build_causal_mask, combine_masks
from .._ops.rope import apply_rope
from .._ops.scaling import attention_scale, masked_fill_penalty


def causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention with an explicit causal mask.

    q, k, v      : (B, H, S, head_dim)
    padding_mask : (B, S) bool, True = real token.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) * attention_scale(q.shape[-1])

    causal = build_causal_mask(scores.shape[-1], device=scores.device)
    keep = combine_masks(causal, padding_mask)

    scores = scores.masked_fill(~keep, masked_fill_penalty(scores.dtype))
    weights = torch.softmax(scores, dim=-1)
    if padding_mask is not None:
        weights = weights * padding_mask[:, None, :, None]
    return torch.matmul(weights, v)


class Attention(nn.Module):
    def __init__(self, cfg: Config) -> None:
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
''')

add("_layers/mlp.py", '''
"""Position-wise feed-forward network."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._core.settings import Config


class MLP(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        hidden = 4 * cfg.d_model
        self.up_proj = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.up_proj(x)))
''')

add("_layers/block.py", '''
"""One pre-norm transformer block."""

from __future__ import annotations

import torch
import torch.nn as nn

from .._core.settings import Config
from .attention import Attention
from .mlp import MLP
from .rmsnorm import RMSNorm


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
''')

add("_layers/embedding.py", '''
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
''')

# --------------------------------------------------------------------------- #
# _model
# --------------------------------------------------------------------------- #
add("_model/__init__.py", '''
"""The assembled language model."""

from .gpt import TinyGPT
from .objective import shifted_cross_entropy_sum

__all__ = ["TinyGPT", "shifted_cross_entropy_sum"]
''')

add("_model/gpt.py", '''
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
''')

add("_model/objective.py", '''
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
''')

# --------------------------------------------------------------------------- #
# _data
# --------------------------------------------------------------------------- #
add("_data/__init__.py", '''
"""Synthetic corpus, padding and batching."""

from .batching import make_batches
from .collate import collate
from .sequence import make_sequence
from .vocab import CYCLES

__all__ = ["make_batches", "collate", "make_sequence", "CYCLES"]
''')

add("_data/vocab.py", '''
"""The synthetic symbol cycles.

A fixed set of disjoint cycles. Every symbol belongs to exactly one cycle and therefore
has exactly one legal successor, which makes next-token prediction a deterministic
function of the current token and thus learnable in a few dozen CPU steps.
"""

from __future__ import annotations

CYCLES: tuple[tuple[int, ...], ...] = (
    (11, 12, 13),
    (21, 22, 23, 24),
    (31, 32),
    (41, 42, 43, 44, 45),
)


def successor(symbol: int) -> int | None:
    """The single legal next symbol, or None if the symbol is not in the corpus."""
    for cycle in CYCLES:
        if symbol in cycle:
            return cycle[(cycle.index(symbol) + 1) % len(cycle)]
    return None
''')

add("_data/sequence.py", '''
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
''')

add("_data/collate.py", '''
"""Padding and label construction.

Sequences have different lengths on purpose: uneven padding is what exposes
loss-normalization and masking bugs, and a fixed-length corpus hides them.

Padded positions must carry `ignore_index` in `labels`, never `pad_token`. The
cross-entropy counts every label that is not `ignore_index`, so a padded position
written as a real token would be supervised as if the model were meant to predict
padding.
"""

from __future__ import annotations

import torch

from .._core.settings import Config
from .._core.types import Batch


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
''')

add("_data/batching.py", '''
"""Deterministic batch construction."""

from __future__ import annotations

import torch

from .._core.settings import Config
from .._core.types import Batch
from .._util.seeding import seeded_generator
from .collate import collate
from .sequence import make_sequence


def make_batches(cfg: Config, n_batches: int, seed: int) -> list[Batch]:
    """Deterministic list of batches with uneven sequence lengths."""
    generator = seeded_generator(seed)
    batches = []
    for _ in range(n_batches):
        lengths = torch.randint(
            8, cfg.max_seq_len // 2, (cfg.micro_batch_size,), generator=generator
        ).tolist()
        batches.append(collate(cfg, [make_sequence(cfg, n, generator) for n in lengths]))
    return batches
''')

# --------------------------------------------------------------------------- #
# _optim
# --------------------------------------------------------------------------- #
add("_optim/__init__.py", '''
"""Optimizer construction, schedule and clipping."""

from .clipping import clip_gradients
from .factory import make_optimizer
from .schedule import make_scheduler, warmup_cosine

__all__ = ["clip_gradients", "make_optimizer", "make_scheduler", "warmup_cosine"]
''')

add("_optim/factory.py", '''
"""Optimizer construction."""

from __future__ import annotations

import torch

from .._core.registry import register
from .._core.settings import Config


@register("optimizer", "adamw")
def make_optimizer(model: torch.nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW with weight decay applied only to matmul weights.

    Norm gains and any 1-D parameter are excluded: decaying them pulls the
    normalization scale towards zero, which is not what weight decay is for.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2:
            decay.append(param)
        else:
            no_decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
''')

add("_optim/schedule.py", '''
"""Learning-rate schedule.

Linear warmup for `warmup_steps`, then cosine decay to zero over the remaining
budget. Expressed as a multiplier on the base lr so it composes with LambdaLR.
"""

from __future__ import annotations

import math

import torch

from .._core.settings import Config


def warmup_cosine(step: int, warmup_steps: int, total_steps: int) -> float:
    """The multiplier for `step`."""
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def make_scheduler(
    optimizer: torch.optim.Optimizer, cfg: Config
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        return warmup_cosine(step, cfg.warmup_steps, cfg.total_steps)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
''')

add("_optim/clipping.py", '''
"""Gradient clipping."""

from __future__ import annotations

import torch


def clip_gradients(model: torch.nn.Module, max_norm: float) -> float:
    """Clip in place and return the pre-clip global norm."""
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))
''')

# --------------------------------------------------------------------------- #
# _train
# --------------------------------------------------------------------------- #
add("_train/__init__.py", '''
"""The training loop."""

from .accumulate import accumulate_gradients
from .loop import train

__all__ = ["accumulate_gradients", "train"]
''')

add("_train/accumulate.py", '''
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
''')

add("_train/history.py", '''
"""Training history bookkeeping.

Every value is coerced to a Python float: the documented contract is that the history
is JSON-serialisable, and a numpy scalar looks identical until `json.dumps` refuses it.
"""

from __future__ import annotations


def new_history() -> dict:
    return {"loss": [], "lr": [], "grad_norm": []}


def record(history: dict, loss: float, lr: float, grad_norm: float) -> None:
    history["loss"].append(float(loss))
    history["lr"].append(float(lr))
    history["grad_norm"].append(float(grad_norm))


def finalize(history: dict) -> dict:
    finite = all(
        v == v and abs(v) != float("inf") for v in history["loss"] + history["grad_norm"]
    )
    history["all_finite"] = finite
    history["final_loss"] = history["loss"][-1] if history["loss"] else float("nan")
    return history
''')

add("_train/loop.py", '''
"""The training loop proper."""

from __future__ import annotations

import torch

from .._core.registry import resolve
from .._core.settings import Config
from .._data.batching import make_batches
from .._model.gpt import TinyGPT
from .._optim.clipping import clip_gradients
from .._optim.schedule import make_scheduler
from .._util.reporting import format_step
from .accumulate import accumulate_gradients
from .history import finalize, new_history, record


def train(cfg: Config, steps: int | None = None, verbose: bool = True) -> dict:
    """Run a short training loop and return its history."""
    torch.manual_seed(cfg.seed)
    steps = steps if steps is not None else cfg.total_steps

    model = TinyGPT(cfg)
    optimizer = resolve("optimizer", "adamw")(model, cfg)
    scheduler = make_scheduler(optimizer, cfg)

    batches = make_batches(cfg, steps * cfg.grad_accum_steps, seed=cfg.seed + 1)
    history = new_history()

    for step in range(steps):
        window = batches[step * cfg.grad_accum_steps : (step + 1) * cfg.grad_accum_steps]
        loss = accumulate_gradients(model, window)
        grad_norm = clip_gradients(model, cfg.grad_clip)

        # optimizer.step() must come before scheduler.step(); the reverse order
        # silently skips the first entry of the schedule.
        optimizer.step()
        scheduler.step()

        record(history, loss, optimizer.param_groups[0]["lr"], grad_norm)
        if verbose:
            print(format_step(step, loss, history["lr"][-1], history["grad_norm"][-1]),
                  flush=True)

    return finalize(history)
''')

# --------------------------------------------------------------------------- #
# _metrics
# --------------------------------------------------------------------------- #
add("_metrics/__init__.py", '''
"""Diagnostics computed from a finished run."""

from .curves import loss_slope
from .summary import summarize

__all__ = ["loss_slope", "summarize"]
''')

add("_metrics/curves.py", '''
"""Loss-curve diagnostics."""

from __future__ import annotations


def loss_slope(losses: list[float], window: int = 5) -> float:
    """Mean change per step over the last `window` steps. Negative means learning."""
    tail = [float(v) for v in losses[-window:]]
    if len(tail) < 2:
        return 0.0
    return (tail[-1] - tail[0]) / (len(tail) - 1)
''')

add("_metrics/summary.py", '''
"""One-line run summary, used by the CLI."""

from __future__ import annotations

from .._ops.reductions import global_norm
from .curves import loss_slope


def summarize(history: dict) -> dict:
    return {
        "final_loss": float(history.get("final_loss", float("nan"))),
        "slope": loss_slope(history.get("loss", [])),
        "all_finite": bool(history.get("all_finite", False)),
        "grad_norm_scale": global_norm(
            [__import__("torch").tensor(history.get("grad_norm", [0.0]))]
        ),
    }
''')

# --------------------------------------------------------------------------- #
# _io
# --------------------------------------------------------------------------- #
add("_io/__init__.py", '''
"""Serialization helpers."""

from .serialize import history_to_json

__all__ = ["history_to_json"]
''')

add("_io/serialize.py", '''
"""JSON serialization of a run."""

from __future__ import annotations

import json
from dataclasses import asdict

from .._core.settings import Config


def history_to_json(cfg: Config, history: dict) -> str:
    return json.dumps({"config": asdict(cfg), "history": history})
''')

# --------------------------------------------------------------------------- #
# Public facades -- unchanged API, implementation moved beneath
# --------------------------------------------------------------------------- #
add("__init__.py", '"""A small GPT-style language model."""\n')

add("config.py", '''
"""Model and training configuration."""

from ._core.settings import Config

__all__ = ["Config"]
''')

add("norm.py", '''
"""Normalization layers."""

from __future__ import annotations

import torch

from ._layers.rmsnorm import RMSNorm as RMSNorm


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Root-mean-square layer norm."""
    from ._layers.rmsnorm import rms_norm as _impl

    return _impl(x, weight, eps)


__all__ = ["RMSNorm", "rms_norm"]
''')

add("positional.py", '''
"""Rotary position embeddings."""

from __future__ import annotations

import torch


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cos, sin), each of shape (seq_len, head_dim)."""
    from ._ops.rope import build_rope_cache as _impl

    return _impl(seq_len, head_dim, theta, device, dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate by pairing dim i with dim i + head_dim/2."""
    from ._ops.rope import rotate_half as _impl

    return _impl(x)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q and k of shape (B, H, S, head_dim)."""
    from ._ops.rope import apply_rope as _impl

    return _impl(q, k, cos, sin)
''')

add("attention.py", '''
"""Causal self-attention."""

from __future__ import annotations

import torch

from ._layers.attention import Attention as Attention


def causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention with an explicit causal mask."""
    from ._layers.attention import causal_attention as _impl

    return _impl(q, k, v, padding_mask)


__all__ = ["Attention", "causal_attention"]
''')

add("model.py", '''
"""The model and its objective."""

from __future__ import annotations

import torch

from ._layers.block import Block as Block
from ._layers.mlp import MLP as MLP
from ._model.gpt import TinyGPT as TinyGPT


def shifted_cross_entropy_sum(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Next-token cross entropy as a SUM, plus the supervised token count."""
    from ._model.objective import shifted_cross_entropy_sum as _impl

    return _impl(logits, labels, ignore_index)


__all__ = ["TinyGPT", "Block", "MLP", "shifted_cross_entropy_sum"]
''')

add("data.py", '''
"""Synthetic dataset, padding and label construction."""

from __future__ import annotations

import torch

from ._core.types import Batch as Batch
from ._data.vocab import CYCLES as CYCLES
from ._core.settings import Config


def make_sequence(cfg: Config, length: int, generator: torch.Generator) -> list[int]:
    """A repeating cycle, entered at a random phase."""
    from ._data.sequence import make_sequence as _impl

    return _impl(cfg, length, generator)


def collate(cfg: Config, sequences: list[list[int]]) -> Batch:
    """Right-pad to the longest sequence in the batch."""
    from ._data.collate import collate as _impl

    return _impl(cfg, sequences)


def make_batches(cfg: Config, n_batches: int, seed: int) -> list[Batch]:
    """Deterministic list of batches with uneven sequence lengths."""
    from ._data.batching import make_batches as _impl

    return _impl(cfg, n_batches, seed)
''')

add("optim.py", '''
"""Optimizer and learning-rate schedule."""

from __future__ import annotations

import torch

from ._core.settings import Config


def make_optimizer(model: torch.nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW with weight decay applied only to matmul weights."""
    from ._optim.factory import make_optimizer as _impl

    return _impl(model, cfg)


def make_scheduler(
    optimizer: torch.optim.Optimizer, cfg: Config
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then cosine decay."""
    from ._optim.schedule import make_scheduler as _impl

    return _impl(optimizer, cfg)
''')

add("train.py", '''
"""Training loop with gradient accumulation.

Runnable directly:

    python -m tinygpt.train --steps 40
"""

from __future__ import annotations

import argparse

from ._core.settings import Config
from ._core.types import Batch
from ._io.serialize import history_to_json
from ._metrics.summary import summarize


def accumulate_gradients(model, micro_batches: list[Batch]) -> float:
    """Accumulate over micro-batches; returns the mean loss per supervised token."""
    from ._train.accumulate import accumulate_gradients as _impl

    return _impl(model, micro_batches)


def train(cfg: Config, steps: int | None = None, verbose: bool = True) -> dict:
    """Run a short training loop and return its history."""
    from ._train.loop import train as _impl

    return _impl(cfg, steps, verbose)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--json", action="store_true", help="print the history as JSON")
    args = ap.parse_args()

    cfg = Config()
    history = train(cfg, steps=args.steps, verbose=not args.json)
    if args.json:
        print(history_to_json(cfg, history))
    else:
        stats = summarize(history)
        print(
            f"\\nfinal loss {stats['final_loss']:.4f}  all_finite={stats['all_finite']}"
        )


if __name__ == "__main__":
    main()
''')


def main() -> int:
    if DEST.exists():
        import shutil

        shutil.rmtree(DEST)
    for rel, body in FILES.items():
        path = DEST / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    print(f"wrote {len(FILES)} files to {DEST.relative_to(ROOT).as_posix()}")
    for rel in sorted(FILES):
        print(f"  tinygpt/{rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
