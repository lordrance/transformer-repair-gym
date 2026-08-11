"""The graded check suite.

Three levels, per the verifier contract in PHASE_0_FEASIBILITY_REPORT.md:

    L1 structural  - imports, public API present, shapes, dtypes, finiteness
    L2 numerical   - multiple hidden inputs vs the reference, with atol/rtol,
                     forward and gradient
    L3 behavioural - short controlled training, compared as a tolerance band and
                     an invariance property, never as a fragile exact threshold

Two rules make the suite hard to game:

1. Ground truth always comes from `trgym.reference`, which lives outside the
   workspace and is hashed before grading. The copy of `full_batch_gradients`
   sitting in the T4 workspace is deliberately NOT used for grading.
2. Hidden inputs use different seeds, sequence lengths and batch sizes than the
   visible ones, so memorizing the visible fixtures does not transfer.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Iterable

import torch

from trgym.reference import tiny_gpt as ref_gpt
from trgym.reference import train_loop as ref_train
from trgym.verifier.loader import CandidateLoadError, load_module, require

# Sequence lengths the visible tests never use.
HIDDEN_SEQ_LENS = (1, 3, 5, 16, 31, 64)
VISIBLE_SEQ_LEN = 8

ATOL = 1e-5
RTOL = 1e-4


class CheckFailure(AssertionError):
    """A graded check failed. The message is surfaced to the trace."""


def _seeded(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _cfg(candidate) -> object:
    return candidate.TinyGPTConfig()


def _paired_models(candidate, seed: int = 20260809):
    """Build a candidate model and a reference model sharing identical weights."""
    torch.manual_seed(seed)
    ref = ref_gpt.TinyGPT(ref_gpt.TinyGPTConfig())
    torch.manual_seed(seed)
    cand = candidate.TinyGPT(candidate.TinyGPTConfig())
    try:
        cand.load_state_dict(ref.state_dict(), strict=True)
    except Exception as exc:  # noqa: BLE001
        raise CheckFailure(f"parameter layout changed vs reference: {exc}")
    ref.eval()
    cand.eval()
    return cand, ref


# --------------------------------------------------------------------------- #
# L1 - structural
# --------------------------------------------------------------------------- #
def check_forward_shapes(ws: Path, candidate, _support) -> None:
    require(candidate, "TinyGPT", "TinyGPTConfig", "rms_norm", "rotate_half", "causal_attention")
    cfg = _cfg(candidate)
    torch.manual_seed(0)
    model = candidate.TinyGPT(cfg)
    for batch, seq in ((1, 1), (2, VISIBLE_SEQ_LEN), (3, 5), (2, 33)):
        ids = torch.randint(0, cfg.vocab_size, (batch, seq), generator=_seeded(seq))
        out = model(ids)
        if tuple(out.shape) != (batch, seq, cfg.vocab_size):
            raise CheckFailure(
                f"logits shape {tuple(out.shape)} != {(batch, seq, cfg.vocab_size)} "
                f"for input {(batch, seq)}"
            )
        if out.dtype != torch.float32:
            raise CheckFailure(f"logits dtype {out.dtype} != float32")
        if not torch.isfinite(out).all():
            raise CheckFailure(f"non-finite logits for input {(batch, seq)}")


def check_loss_runs_and_is_finite(ws: Path, candidate, _support) -> None:
    cfg = _cfg(candidate)
    torch.manual_seed(0)
    model = candidate.TinyGPT(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 9), generator=_seeded(7))
    loss_sum, n_tokens = model.loss_sum(ids, ids.clone())
    if not torch.isfinite(loss_sum):
        raise CheckFailure("loss_sum is not finite")
    if int(n_tokens) <= 0:
        raise CheckFailure(f"token count {int(n_tokens)} must be positive")


def check_grad_accum_runs(ws: Path, candidate, support) -> None:
    require(candidate, "MicroBatch", "accumulate_gradients")
    model = _build_support_model(support)
    mbs = _micro_batches(support, candidate, n_pads=(0, 0), seed=3)
    loss = candidate.accumulate_gradients(model, mbs)
    if not isinstance(loss, float) or not math.isfinite(loss):
        raise CheckFailure(f"accumulate_gradients returned {loss!r}, expected a finite float")
    if all(p.grad is None for p in model.parameters()):
        raise CheckFailure("no gradients were populated")


# --------------------------------------------------------------------------- #
# L2 - numerical, attention (F1)
# --------------------------------------------------------------------------- #
def check_strict_causality(ws: Path, candidate, _support) -> None:
    cfg = _cfg(candidate)
    torch.manual_seed(1)
    model = candidate.TinyGPT(cfg)
    model.eval()
    for seq in (3, 5, 16, 31):
        ids = torch.randint(0, cfg.vocab_size, (1, seq), generator=_seeded(100 + seq))
        with torch.no_grad():
            base = model(ids)
        for pos in {1, seq // 2, seq - 1}:
            if pos <= 0:
                continue
            bumped = ids.clone()
            bumped[0, pos] = (bumped[0, pos] + 1) % cfg.vocab_size
            with torch.no_grad():
                other = model(bumped)
            if not torch.allclose(base[:, :pos], other[:, :pos], atol=ATOL, rtol=RTOL):
                delta = (base[:, :pos] - other[:, :pos]).abs().max()
                raise CheckFailure(
                    f"information leaked backwards: changing token {pos} of a "
                    f"length-{seq} sequence moved positions 0..{pos - 1} by {float(delta):.3e}"
                )


def check_single_token_identity(ws: Path, candidate, _support) -> None:
    q = torch.randn(1, 1, 1, 16, generator=_seeded(11))
    out = candidate.causal_attention(q, q, q)
    if not torch.allclose(out, q, atol=1e-6):
        raise CheckFailure("attention over a single token must return that token's value")


def check_padding_keys_masked(ws: Path, candidate, _support) -> None:
    b, h, s, hd = 2, 2, 9, 8
    g = _seeded(13)
    q = torch.randn(b, h, s, hd, generator=g)
    k = torch.randn(b, h, s, hd, generator=g)
    v = torch.randn(b, h, s, hd, generator=g)
    pad = torch.ones(b, s, dtype=torch.bool)
    pad[:, 6:] = False

    out1 = candidate.causal_attention(q, k, v, pad)
    v2 = v.clone()
    v2[:, :, 6:] = torch.randn(b, h, 3, hd, generator=g) * 50.0
    out2 = candidate.causal_attention(q, k, v2, pad)

    if not torch.allclose(out1[:, :, :6], out2[:, :, :6], atol=ATOL, rtol=RTOL):
        raise CheckFailure("padded key positions still influence unpadded outputs")


# --------------------------------------------------------------------------- #
# L2 - numerical, RoPE (F2)
# --------------------------------------------------------------------------- #
def check_rope_norm_preserved(ws: Path, candidate, _support) -> None:
    cos, sin = candidate.build_rope_cache(16, 16, 10000.0)
    q = torch.randn(2, 3, 16, 16, generator=_seeded(17))
    q_rot, _ = candidate.apply_rope(q, q, cos, sin)
    if not torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5):
        raise CheckFailure("RoPE must be norm-preserving")


def check_rope_position_zero_identity(ws: Path, candidate, _support) -> None:
    cos, sin = candidate.build_rope_cache(4, 16, 10000.0)
    if not torch.allclose(cos[0], torch.ones(16), atol=1e-6):
        raise CheckFailure("cos at position 0 must be all ones")
    if not torch.allclose(sin[0], torch.zeros(16), atol=1e-6):
        raise CheckFailure("sin at position 0 must be all zeros")


def check_rope_relative_property(ws: Path, candidate, _support) -> None:
    """<RoPE(q, m), RoPE(k, n)> must depend only on m - n."""
    cos, sin = candidate.build_rope_cache(40, 16, 10000.0)
    g = _seeded(19)
    q = torch.randn(1, 1, 1, 16, generator=g)
    k = torch.randn(1, 1, 1, 16, generator=g)

    def dot(m: int, n: int) -> float:
        qm, _ = candidate.apply_rope(q, q, cos[m : m + 1], sin[m : m + 1])
        kn, _ = candidate.apply_rope(k, k, cos[n : n + 1], sin[n : n + 1])
        return float((qm * kn).sum())

    for (m1, n1), (m2, n2) in (((9, 4), (14, 9)), ((20, 3), (33, 16))):
        a, b = dot(m1, n1), dot(m2, n2)
        if abs(a - b) > 1e-3 * max(1.0, abs(a)):
            raise CheckFailure(
                f"RoPE is not translation invariant: <q@{m1},k@{n1}>={a:.5f} but "
                f"<q@{m2},k@{n2}>={b:.5f} (same relative distance)"
            )


# --------------------------------------------------------------------------- #
# L2 - numerical, normalization (F3)
# --------------------------------------------------------------------------- #
def check_rms_norm_matches_float32_math(ws: Path, candidate, _support) -> None:
    g = _seeded(23)
    x = torch.randn(4, 16, generator=g)
    w = torch.randn(16, generator=g)
    got = candidate.rms_norm(x, w, 1e-6)
    want = w * (x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))
    if not torch.allclose(got, want, atol=1e-6, rtol=1e-5):
        raise CheckFailure("rms_norm does not match the float32 definition")


def check_rms_norm_dtype_preserved(ws: Path, candidate, _support) -> None:
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        x = torch.randn(4, 32, generator=_seeded(29)).to(dtype)
        out = candidate.rms_norm(x, torch.ones(32), 1e-6)
        if out.dtype != dtype:
            raise CheckFailure(f"rms_norm({dtype}) returned {out.dtype}; dtype must be preserved")


def check_rms_norm_float16_no_collapse(ws: Path, candidate, _support) -> None:
    """fp16 squares overflow above ~256; the variance must be accumulated in fp32.

    Sweeps batch shape as well as scale: a fix that special-cases the one
    (4, 64) fixture from the visible suite has to fail somewhere here.
    """
    for scale in (100.0, 300.0, 800.0):
        for batch, dim in ((2, 64), (4, 32), (8, 128), (3, 64)):
            x = (torch.randn(batch, dim, generator=_seeded(31)) * scale).to(torch.float16)
            out = candidate.rms_norm(x, torch.ones(dim), 1e-6).float()
            if not torch.isfinite(out).all():
                raise CheckFailure(
                    f"non-finite output at scale {scale}, shape {(batch, dim)}"
                )
            rms = out.pow(2).mean(-1).sqrt()
            if not torch.allclose(rms, torch.ones_like(rms), atol=0.05):
                raise CheckFailure(
                    f"output RMS is {rms.tolist()} at scale {scale}, shape "
                    f"{(batch, dim)}; expected ~1.0 (fp16 variance overflowed)"
                )


def check_rms_norm_matches_reference_fp16(ws: Path, candidate, _support) -> None:
    """Element-wise agreement with the reference under fp16, not just unit RMS.

    Unit RMS is self-normalizing: clamping the input, rescaling it, or otherwise
    distorting it still yields RMS 1. Only an element-wise comparison against the
    reference distinguishes "stable" from "correct".
    """
    for scale in (50.0, 300.0):
        for batch, dim in ((4, 64), (2, 32)):
            g = _seeded(37)
            x = (torch.randn(batch, dim, generator=g) * scale).to(torch.float16)
            w = torch.randn(dim, generator=g).to(torch.float16)
            got = candidate.rms_norm(x, w, 1e-6).float()
            want = ref_gpt.rms_norm(x, w, 1e-6).float()
            # fp16 has ~3 decimal digits; compare relative to the reference scale.
            denom = max(1e-6, float(want.abs().max()))
            rel = float((got - want).abs().max()) / denom
            if rel > 5e-3:
                raise CheckFailure(
                    f"rms_norm differs from reference by {rel:.3%} at scale {scale}, "
                    f"shape {(batch, dim)}; output is stable but not the same function"
                )


# --------------------------------------------------------------------------- #
# L2 - numerical, loss construction (F5)
# --------------------------------------------------------------------------- #
def check_loss_is_next_token(ws: Path, candidate, _support) -> None:
    cfg = _cfg(candidate)
    torch.manual_seed(2)
    model = candidate.TinyGPT(cfg)
    for seq in (4, 9, 17):
        ids = torch.randint(0, cfg.vocab_size, (2, seq), generator=_seeded(200 + seq))
        loss_sum, n_tokens = model.loss_sum(ids, ids.clone())
        if int(n_tokens) != 2 * (seq - 1):
            raise CheckFailure(
                f"supervised token count {int(n_tokens)} != {2 * (seq - 1)} for seq={seq}; "
                "next-token training supervises S-1 positions per row"
            )
        per_token = float(loss_sum.detach()) / int(n_tokens)
        if abs(per_token - math.log(cfg.vocab_size)) > 1.0:
            raise CheckFailure(
                f"per-token loss {per_token:.3f} at init is implausible; "
                f"expected ~ln(vocab)={math.log(cfg.vocab_size):.3f}"
            )


def check_ignore_index_excluded_from_count(ws: Path, candidate, _support) -> None:
    cfg = _cfg(candidate)
    torch.manual_seed(2)
    model = candidate.TinyGPT(cfg)
    for seq, keep in ((12, 7), (20, 5)):
        ids = torch.randint(0, cfg.vocab_size, (2, seq), generator=_seeded(300 + seq))
        labels = ids.clone()
        labels[:, keep:] = cfg.ignore_index
        _, n_tokens = model.loss_sum(ids, labels)
        expected = 2 * (keep - 1)
        if int(n_tokens) != expected:
            raise CheckFailure(
                f"token count {int(n_tokens)} != {expected}; ignore_index positions "
                "must not be counted"
            )


def check_padding_does_not_change_loss(ws: Path, candidate, _support) -> None:
    cfg = _cfg(candidate)
    torch.manual_seed(2)
    model = candidate.TinyGPT(cfg)
    core_len = 11
    core = torch.randint(0, cfg.vocab_size, (2, core_len), generator=_seeded(41))
    loss_a, n_a = model.loss_sum(core, core.clone())

    for pad_len in (1, 6):
        ids = torch.cat([core, torch.zeros(2, pad_len, dtype=torch.long)], dim=1)
        labels = ids.clone()
        labels[:, core_len:] = cfg.ignore_index
        mask = torch.ones(2, core_len + pad_len, dtype=torch.bool)
        mask[:, core_len:] = False
        loss_b, n_b = model.loss_sum(ids, labels, mask)

        if int(n_a) != int(n_b):
            raise CheckFailure(
                f"padding by {pad_len} changed the supervised token count "
                f"{int(n_a)} -> {int(n_b)}"
            )
        rel = abs(float(loss_a.detach()) - float(loss_b.detach())) / max(1e-6, abs(float(loss_a.detach())))
        if rel > 1e-4:
            raise CheckFailure(
                f"padding by {pad_len} changed the loss by {rel:.2%}; padded positions "
                "must be fully masked out of both attention and the objective"
            )


def check_matches_reference_forward(ws: Path, candidate, _support) -> None:
    """The strongest numerical check: identical weights must give identical logits."""
    cand, ref = _paired_models(candidate)
    cfg = ref.cfg
    for seq in HIDDEN_SEQ_LENS:
        for batch in (1, 3):
            ids = torch.randint(0, cfg.vocab_size, (batch, seq), generator=_seeded(seq * 7 + batch))
            with torch.no_grad():
                got = cand(ids)
                want = ref(ids)
            if not torch.allclose(got, want, atol=ATOL, rtol=RTOL):
                delta = float((got - want).abs().max())
                raise CheckFailure(
                    f"logits differ from reference by {delta:.3e} at batch={batch}, seq={seq} "
                    f"(atol={ATOL}, rtol={RTOL})"
                )


# --------------------------------------------------------------------------- #
# Gradient accumulation (F4) - L2 and L3
# --------------------------------------------------------------------------- #
def _build_support_model(support):
    torch.manual_seed(20260809)
    return support.TinyGPT(support.TinyGPTConfig())


def _micro_batches(support, candidate, n_pads: Iterable[int], seed: int, seq: int = 12, batch: int = 2):
    """Build micro-batches whose supervised-token counts differ across the group.

    The container type comes from the candidate module, since `MicroBatch` is
    part of the public API the candidate is required to preserve.
    """
    cfg = support.TinyGPTConfig()
    micro_batch_cls = candidate.MicroBatch
    g = _seeded(seed)
    out = []
    for n_pad in n_pads:
        ids = torch.randint(0, cfg.vocab_size, (batch, seq), generator=g)
        labels = ids.clone()
        if n_pad:
            labels[:, -n_pad:] = cfg.ignore_index
        out.append(micro_batch_cls(ids, labels))
    return out


def _grad_vector(model) -> torch.Tensor:
    return torch.cat(
        [
            (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
            for p in model.parameters()
        ]
    )


def _compare_accum_vs_full(candidate, support, n_pads, seed) -> None:
    model = _build_support_model(support)
    mbs = _micro_batches(support, candidate, n_pads, seed)

    candidate.accumulate_gradients(model, mbs)
    g_accum = _grad_vector(model)

    # Ground truth from trgym.reference, NOT from the workspace copy.
    ref_mbs = [ref_train.MicroBatch(mb.input_ids, mb.labels, mb.padding_mask) for mb in mbs]
    ref_train.full_batch_gradients(model, ref_mbs)
    g_full = _grad_vector(model)

    denom = max(1e-8, float(g_full.abs().max()))
    rel = float((g_accum - g_full).abs().max()) / denom
    if rel > 1e-4:
        raise CheckFailure(
            f"accumulated gradients differ from the full-batch gradients by "
            f"{rel:.3%} (relative max abs) with padding pattern {tuple(n_pads)}"
        )


def check_grad_accum_matches_full_batch_even(ws: Path, candidate, support) -> None:
    _compare_accum_vs_full(candidate, support, (0, 0, 0), seed=53)


def check_grad_accum_matches_full_batch_uneven(ws: Path, candidate, support) -> None:
    for pads in ((0, 3, 5), (7, 1, 0, 4)):
        _compare_accum_vs_full(candidate, support, pads, seed=59)


def check_grad_accum_invariant_to_split(ws: Path, candidate, support) -> None:
    """L3: the same data must produce the same update regardless of how it is split."""
    all_pads = (0, 3, 5, 2)
    model_a = _build_support_model(support)
    mbs_a = _micro_batches(support, candidate, all_pads, seed=61)
    candidate.accumulate_gradients(model_a, mbs_a)
    g_a = _grad_vector(model_a)

    # Same examples, one micro-batch per row instead of per group of rows.
    model_b = _build_support_model(support)
    mbs_b = _micro_batches(support, candidate, all_pads, seed=61)
    split: list = []
    MB = type(mbs_b[0])
    for mb in mbs_b:
        for row in range(mb.input_ids.shape[0]):
            split.append(
                MB(
                    mb.input_ids[row : row + 1],
                    mb.labels[row : row + 1],
                    None if mb.padding_mask is None else mb.padding_mask[row : row + 1],
                )
            )
    candidate.accumulate_gradients(model_b, split)
    g_b = _grad_vector(model_b)

    denom = max(1e-8, float(g_a.abs().max()))
    rel = float((g_a - g_b).abs().max()) / denom
    if rel > 1e-4:
        raise CheckFailure(
            f"re-splitting the same data into {len(split)} micro-batches changed the "
            f"gradient by {rel:.3%}; accumulation must be split-invariant"
        )


# --------------------------------------------------------------------------- #
# Registry / runner
# --------------------------------------------------------------------------- #
CHECKS: dict[str, Callable] = {
    name[len("check_") :]: fn
    for name, fn in list(globals().items())
    if name.startswith("check_") and callable(fn)
}

LEVELS = {
    "forward_shapes": 1,
    "loss_runs_and_is_finite": 1,
    "grad_accum_runs": 1,
    "strict_causality": 2,
    "single_token_identity": 2,
    "padding_keys_masked": 2,
    "rope_norm_preserved": 2,
    "rope_position_zero_identity": 2,
    "rope_relative_property": 2,
    "rms_norm_matches_float32_math": 2,
    "rms_norm_dtype_preserved": 2,
    "rms_norm_float16_no_collapse": 2,
    "rms_norm_matches_reference_fp16": 2,
    "loss_is_next_token": 2,
    "ignore_index_excluded_from_count": 2,
    "padding_does_not_change_loss": 2,
    "matches_reference_forward": 2,
    "grad_accum_matches_full_batch_even": 2,
    "grad_accum_matches_full_batch_uneven": 2,
    "grad_accum_invariant_to_split": 3,
    # visible suite: same properties, one fixed public configuration each
    "visible_forward_smoke": 1,
    "visible_loss_smoke": 1,
    "visible_causality_len8": 2,
    "visible_rope_relative_len16": 2,
    "visible_rms_norm_fp16_scale300": 2,
    "visible_grad_accum_vs_local_oracle": 2,
    "visible_ignore_index_count_len12": 2,
}


def run_checks(
    workspace: Path, task_id: str, names: Iterable[str]
) -> list[tuple[str, bool, str]]:
    """Run `names` against the candidate in `workspace`.

    Returns [(check_name, passed, detail)]. A load failure fails every check
    rather than raising, so the caller always gets a complete row.
    """
    from trgym.tasks.registry import get_task

    workspace = Path(workspace)
    spec = get_task(task_id)
    names = list(names)

    torch.manual_seed(0)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # noqa: BLE001 - not all builds support every op
        pass

    try:
        if spec.target_file == "train_loop.py":
            support = load_module(workspace / "tiny_gpt.py")
            candidate = load_module(workspace / "train_loop.py", extra_sys_path=workspace)
        else:
            candidate = load_module(workspace / spec.target_file)
            support = candidate
    except CandidateLoadError as exc:
        return [(n, False, f"load error: {exc}") for n in names]
    except Exception as exc:  # noqa: BLE001
        return [(n, False, f"load error: {type(exc).__name__}: {exc}") for n in names]

    # Imported lazily: trgym.verifier.visible imports helpers from this module,
    # so a top-level import here would be circular.
    from trgym.verifier.visible import VISIBLE_CHECKS

    registry = {**CHECKS, **VISIBLE_CHECKS}

    results: list[tuple[str, bool, str]] = []
    for name in names:
        fn = registry.get(name)
        if fn is None:
            results.append((name, False, f"unknown check {name!r}"))
            continue
        try:
            fn(workspace, candidate, support)
            results.append((name, True, ""))
        except CheckFailure as exc:
            results.append((name, False, str(exc)))
        except CandidateLoadError as exc:
            results.append((name, False, f"API error: {exc}"))
        except Exception as exc:  # noqa: BLE001
            results.append((name, False, f"{type(exc).__name__}: {exc}"))
    return results
