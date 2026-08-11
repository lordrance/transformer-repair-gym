"""Sanity tests for the pristine reference implementation.

If these fail, every task in the environment is meaningless, so they run first.
"""

import math

import pytest
import torch

from trgym.reference.tiny_gpt import (
    TinyGPT,
    TinyGPTConfig,
    apply_rope,
    build_rope_cache,
    causal_attention,
    rms_norm,
)
from trgym.reference.train_loop import (
    MicroBatch,
    accumulate_gradients,
    full_batch_gradients,
)

CFG = TinyGPTConfig()


def make_model(seed: int = 0) -> TinyGPT:
    torch.manual_seed(seed)
    return TinyGPT(CFG)


# --------------------------------------------------------------------------- #
# Shapes / basic wiring
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("batch,seq", [(1, 1), (2, 8), (3, 31), (2, 64)])
def test_forward_shapes(batch: int, seq: int) -> None:
    model = make_model()
    ids = torch.randint(0, CFG.vocab_size, (batch, seq))
    logits = model(ids)
    assert logits.shape == (batch, seq, CFG.vocab_size)
    assert torch.isfinite(logits).all()


# --------------------------------------------------------------------------- #
# F1: causality
# --------------------------------------------------------------------------- #
def test_attention_is_strictly_causal() -> None:
    """Perturbing a future token must not change earlier positions' outputs."""
    model = make_model()
    ids = torch.randint(0, CFG.vocab_size, (1, 12))
    base = model(ids)

    ids2 = ids.clone()
    ids2[0, 7] = (ids2[0, 7] + 1) % CFG.vocab_size
    perturbed = model(ids2)

    # positions 0..6 must be untouched
    assert torch.allclose(base[:, :7], perturbed[:, :7], atol=1e-6, rtol=0)
    # position 7 onwards must actually change (otherwise the test is vacuous)
    assert not torch.allclose(base[:, 7], perturbed[:, 7], atol=1e-6, rtol=0)


def test_attention_diagonal_is_attendable() -> None:
    """A single-token sequence must produce a finite, non-degenerate output."""
    q = torch.randn(1, 1, 1, 16)
    out = causal_attention(q, q, q)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, q, atol=1e-6)


def test_padding_keys_are_masked() -> None:
    """Changing the value vectors at padded positions must not move the output."""
    torch.manual_seed(3)
    b, h, s, hd = 2, 2, 6, 8
    q, k, v = torch.randn(3, b, h, s, hd)
    pad = torch.ones(b, s, dtype=torch.bool)
    pad[:, 4:] = False

    out1 = causal_attention(q, k, v, pad)
    v2 = v.clone()
    v2[:, :, 4:] = torch.randn_like(v2[:, :, 4:])
    out2 = causal_attention(q, k, v2, pad)

    assert torch.allclose(out1[:, :, :4], out2[:, :, :4], atol=1e-6, rtol=0)


# --------------------------------------------------------------------------- #
# F2: RoPE
# --------------------------------------------------------------------------- #
def test_rope_preserves_norm() -> None:
    """RoPE is a rotation, so per-head vector norms must be invariant."""
    cos, sin = build_rope_cache(16, 16, CFG.rope_theta)
    q = torch.randn(2, 3, 16, 16)
    q_rot, _ = apply_rope(q, q, cos, sin)
    assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5)


def test_rope_is_relative() -> None:
    """<RoPE(q,m), RoPE(k,n)> must depend only on (m - n)."""
    cos, sin = build_rope_cache(32, 16, CFG.rope_theta)
    torch.manual_seed(5)
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)

    def dot(m: int, n: int) -> float:
        qm = q * cos[m] + _rot_half(q) * sin[m]
        kn = k * cos[n] + _rot_half(k) * sin[n]
        return float((qm * kn).sum())

    assert dot(9, 4) == pytest.approx(dot(12, 7), abs=1e-4)
    assert dot(9, 4) != pytest.approx(dot(12, 8), abs=1e-4)


def _rot_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def test_rope_position_zero_is_identity() -> None:
    cos, sin = build_rope_cache(4, 16, CFG.rope_theta)
    assert torch.allclose(cos[0], torch.ones(16), atol=1e-6)
    assert torch.allclose(sin[0], torch.zeros(16), atol=1e-6)


# --------------------------------------------------------------------------- #
# F3: normalization stability
# --------------------------------------------------------------------------- #
def test_rms_norm_matches_manual_float32() -> None:
    x = torch.randn(4, 16, dtype=torch.float32)
    w = torch.randn(16)
    got = rms_norm(x, w, 1e-6)
    want = w * (x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))
    assert torch.allclose(got, want, atol=1e-6)


def test_rms_norm_survives_float16_overflow() -> None:
    """Large fp16 activations overflow x**2 (max 65504); the fp32 upcast prevents it.

    Without the upcast the variance becomes inf, rsqrt(inf) == 0, and the whole
    residual stream silently collapses to zero. This is the exact failure mode
    task F3 reproduces.
    """
    x = (torch.randn(4, 64) * 300.0).to(torch.float16)
    out = rms_norm(x, torch.ones(64), 1e-6)
    assert out.dtype == torch.float16
    rms = out.float().pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=0.02), f"rms collapsed to {rms}"


def test_rms_norm_survives_large_bfloat16_activations() -> None:
    """Large bf16 activations must not overflow, because variance runs in fp32."""
    x = (torch.randn(4, 64) * 300.0).to(torch.bfloat16)
    w = torch.ones(64)
    out = rms_norm(x, w, 1e-6)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()
    # unit RMS up to bf16 resolution
    rms = out.float().pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=0.05)


# --------------------------------------------------------------------------- #
# F4: gradient accumulation equivalence
# --------------------------------------------------------------------------- #
def _uneven_micro_batches(seed: int = 1) -> list[MicroBatch]:
    torch.manual_seed(seed)
    out = []
    for n_pad in (0, 3, 5):
        ids = torch.randint(0, CFG.vocab_size, (2, 10))
        labels = ids.clone()
        if n_pad:
            labels[:, -n_pad:] = CFG.ignore_index
        out.append(MicroBatch(ids, labels))
    return out


def test_grad_accumulation_matches_full_batch() -> None:
    model = make_model()
    mbs = _uneven_micro_batches()

    loss_accum = accumulate_gradients(model, mbs)
    g_accum = [p.grad.detach().clone() for p in model.parameters()]

    loss_full = full_batch_gradients(model, mbs)
    g_full = [p.grad.detach().clone() for p in model.parameters()]

    assert loss_accum == pytest.approx(loss_full, rel=1e-6)
    max_diff = max(float((a - b).abs().max()) for a, b in zip(g_accum, g_full))
    assert max_diff < 1e-6, f"gradient mismatch {max_diff}"


# --------------------------------------------------------------------------- #
# F5: loss construction
# --------------------------------------------------------------------------- #
def test_loss_is_next_token_not_identity() -> None:
    """Reference loss must supervise t -> t+1, not t -> t."""
    model = make_model()
    ids = torch.randint(0, CFG.vocab_size, (2, 9))
    loss_sum, n_tokens = model.loss_sum(ids, ids.clone())
    assert int(n_tokens) == 2 * (9 - 1)
    # random init => loss per token near ln(vocab)
    per_token = float(loss_sum.detach()) / int(n_tokens)
    assert abs(per_token - math.log(CFG.vocab_size)) < 1.0


def test_ignore_index_excluded_from_token_count() -> None:
    model = make_model()
    ids = torch.randint(0, CFG.vocab_size, (2, 10))
    labels = ids.clone()
    labels[:, 6:] = CFG.ignore_index
    _, n_tokens = model.loss_sum(ids, labels)
    # shifted labels are positions 1..9; positions 6..9 are ignored -> 5 per row
    assert int(n_tokens) == 2 * 5


def test_padding_does_not_change_supervised_loss() -> None:
    """Appending padding (masked in both attention and labels) must not move the loss."""
    model = make_model()
    torch.manual_seed(11)
    core = torch.randint(0, CFG.vocab_size, (2, 8))

    loss_a, n_a = model.loss_sum(core, core.clone())

    pad_len = 4
    ids_b = torch.cat([core, torch.zeros(2, pad_len, dtype=torch.long)], dim=1)
    labels_b = ids_b.clone()
    labels_b[:, 8:] = CFG.ignore_index
    mask_b = torch.ones(2, 8 + pad_len, dtype=torch.bool)
    mask_b[:, 8:] = False

    loss_b, n_b = model.loss_sum(ids_b, labels_b, mask_b)

    assert int(n_a) == int(n_b)
    assert float(loss_a.detach()) == pytest.approx(float(loss_b.detach()), rel=1e-5)
