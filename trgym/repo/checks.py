"""Graded checks for repo-level tasks.

Three levels, same contract as Tier E:

    L1 contract    imports, public API, shapes/dtypes, patch applies
    L2 semantic    hidden configurations vs the protected gold repo, forward
                   and gradient
    L3 behavioural short training runs: finiteness, convergence, the LR trace,
                   and whether clipping actually clips

The gold repo lives outside the workspace and is built on demand from the
pristine template, so nothing the candidate can reach is used as ground truth.
"""

from __future__ import annotations

import importlib
import itertools
import math
import sys
import tempfile
from pathlib import Path
from typing import Callable, Iterable

import torch

# Visible layer, helpers included, live in `visible_runtime` so they can be planted in
# a candidate workspace without shipping the hidden oracle alongside them. Imported
# here rather than restated so there is exactly one definition of each.
# See PROTOCOL_CHANGELOG R9.
from trgym.repo.visible_runtime import (  # noqa: E402
    VISIBLE_SHAPE,
    CheckFailure,
    RepoModules,
    _seeded,
    check_repo_visible_loss_is_finite,
    check_repo_visible_rope_position_zero,
    check_repo_visible_short_train_runs,
    check_repo_visible_single_token_attention,
    check_repo_visible_smoke,
)

# Sequence lengths and batch shapes the visible suite never uses.
HIDDEN_SHAPES = ((1, 9), (2, 17), (3, 5), (1, 31), (4, 12))

ATOL, RTOL = 1e-5, 1e-4

_GOLD_CACHE: dict[str, Path] = {}


def gold_repo(task_id: str) -> Path:
    """Build (once per process) the protected reference repo, outside any workspace."""
    if task_id not in _GOLD_CACHE:
        from trgym.repo.build import build_gold
        from trgym.tasks.repo_specs import get_repo_task

        tmp = Path(tempfile.mkdtemp(prefix="trgym_gold_"))
        _GOLD_CACHE[task_id] = build_gold(get_repo_task(task_id), tmp / "gold")
    return _GOLD_CACHE[task_id]


def _paired_models(cand: RepoModules, gold: RepoModules, seed: int = 20260810):
    """Candidate and gold model with identical weights."""
    torch.manual_seed(seed)
    gm = gold.model.TinyGPT(gold.config.Config())
    torch.manual_seed(seed)
    cm = cand.model.TinyGPT(cand.config.Config())
    try:
        cm.load_state_dict(gm.state_dict(), strict=True)
    except Exception as exc:  # noqa: BLE001
        raise CheckFailure(f"parameter layout differs from the reference: {exc}")
    cm.eval()
    gm.eval()
    return cm, gm


# --------------------------------------------------------------------------- #
# L2 -- semantic, against the protected gold repo
# --------------------------------------------------------------------------- #
def check_repo_matches_gold_logits(ws: Path, task_id: str) -> None:
    with RepoModules(ws) as cand, RepoModules(gold_repo(task_id)) as gold:
        cm, gm = _paired_models(cand, gold)
        cfg = gm.cfg
        for b, s in HIDDEN_SHAPES:
            ids = torch.randint(1, cfg.vocab_size, (b, s), generator=_seeded(s * 31 + b))
            with torch.no_grad():
                got, want = cm(ids), gm(ids)
            if not torch.allclose(got, want, atol=ATOL, rtol=RTOL):
                delta = float((got - want).abs().max())
                raise CheckFailure(
                    f"logits differ from the reference by {delta:.3e} at batch={b}, seq={s}"
                )


def check_repo_strict_causality(ws: Path, task_id: str) -> None:
    with RepoModules(ws) as cand:
        cfg = cand.config.Config()
        torch.manual_seed(1)
        model = cand.model.TinyGPT(cfg).eval()
        for s in (5, 13, 24):
            ids = torch.randint(1, cfg.vocab_size, (1, s), generator=_seeded(100 + s))
            with torch.no_grad():
                base = model(ids)
            for pos in {1, s // 2, s - 1}:
                bumped = ids.clone()
                bumped[0, pos] = (bumped[0, pos] % (cfg.vocab_size - 1)) + 1
                with torch.no_grad():
                    other = model(bumped)
                if not torch.allclose(base[:, :pos], other[:, :pos], atol=ATOL, rtol=RTOL):
                    delta = float((base[:, :pos] - other[:, :pos]).abs().max())
                    raise CheckFailure(
                        f"changing token {pos} of a length-{s} sequence moved earlier "
                        f"positions by {delta:.3e}: the model can see the future"
                    )


def check_repo_padding_keys_masked(ws: Path, task_id: str) -> None:
    with RepoModules(ws) as cand:
        b, h, s, hd = 2, 2, 9, 8
        g = _seeded(13)
        q, k, v = (torch.randn(b, h, s, hd, generator=g) for _ in range(3))
        pad = torch.ones(b, s, dtype=torch.bool)
        pad[:, 6:] = False
        out1 = cand.attention.causal_attention(q, k, v, pad)
        v2 = v.clone()
        v2[:, :, 6:] = torch.randn(b, h, 3, hd, generator=g) * 50.0
        out2 = cand.attention.causal_attention(q, k, v2, pad)
        if not torch.allclose(out1[:, :, :6], out2[:, :, :6], atol=ATOL, rtol=RTOL):
            raise CheckFailure("padded key positions still influence unpadded outputs")


def check_repo_rope_relative_property(ws: Path, task_id: str) -> None:
    """<RoPE(q,m), RoPE(k,n)> must depend only on m - n."""
    with RepoModules(ws) as cand:
        cos, sin = cand.positional.build_rope_cache(40, 16, 10000.0)
        g = _seeded(19)
        q = torch.randn(1, 1, 1, 16, generator=g)
        k = torch.randn(1, 1, 1, 16, generator=g)

        def dot(m: int, n: int) -> float:
            qm, _ = cand.positional.apply_rope(q, q, cos[m : m + 1], sin[m : m + 1])
            kn, _ = cand.positional.apply_rope(k, k, cos[n : n + 1], sin[n : n + 1])
            return float((qm * kn).sum())

        for (m1, n1), (m2, n2) in (((9, 4), (14, 9)), ((20, 3), (33, 16))):
            a, b = dot(m1, n1), dot(m2, n2)
            if abs(a - b) > 1e-3 * max(1.0, abs(a)):
                raise CheckFailure(
                    f"not translation invariant: <q@{m1},k@{n1}>={a:.5f} vs "
                    f"<q@{m2},k@{n2}>={b:.5f} at the same distance"
                )


def check_repo_rope_norm_preserved(ws: Path, task_id: str) -> None:
    with RepoModules(ws) as cand:
        cos, sin = cand.positional.build_rope_cache(16, 16, 10000.0)
        q = torch.randn(2, 3, 16, 16, generator=_seeded(17))
        q_rot, _ = cand.positional.apply_rope(q, q, cos, sin)
        if not torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5):
            raise CheckFailure("RoPE must be norm-preserving")


def check_repo_supervised_token_count(ws: Path, task_id: str) -> None:
    """Padded positions must not be counted, and must match the reference exactly."""
    with RepoModules(ws) as cand, RepoModules(gold_repo(task_id)) as gold:
        cfg_c, cfg_g = cand.config.Config(), gold.config.Config()
        for seed in (7, 21):
            bc = cand.data.make_batches(cfg_c, 1, seed=seed)[0]
            bg = gold.data.make_batches(cfg_g, 1, seed=seed)[0]
            if int(bc.n_supervised) != int(bg.n_supervised):
                raise CheckFailure(
                    f"batch reports {int(bc.n_supervised)} supervised tokens, "
                    f"reference says {int(bg.n_supervised)}"
                )
            torch.manual_seed(5)
            cm = cand.model.TinyGPT(cfg_c)
            _, n_tokens = cm.loss_sum(bc.input_ids, bc.labels, bc.padding_mask)
            if int(n_tokens) != int(bg.n_supervised):
                raise CheckFailure(
                    f"loss counts {int(n_tokens)} tokens, reference says "
                    f"{int(bg.n_supervised)}: padded positions are being supervised"
                )


def check_repo_padding_does_not_change_loss(ws: Path, task_id: str) -> None:
    """Padding a batch further must not move the loss on the real tokens."""
    with RepoModules(ws) as cand:
        cfg = cand.config.Config()
        torch.manual_seed(5)
        model = cand.model.TinyGPT(cfg)
        g = _seeded(41)
        core = [cand.data.make_sequence(cfg, 11, g) for _ in range(2)]

        batch_a = cand.data.collate(cfg, core)
        loss_a, n_a = model.loss_sum(batch_a.input_ids, batch_a.labels, batch_a.padding_mask)

        # Same two sequences, plus a longer third one that forces extra padding.
        padded = core + [cand.data.make_sequence(cfg, 20, g)]
        batch_b = cand.data.collate(cfg, padded)
        loss_b, _ = model.loss_sum(
            batch_b.input_ids[:2], batch_b.labels[:2], batch_b.padding_mask[:2]
        )
        n_b = int((batch_b.labels[:2, 1:] != cfg.ignore_index).sum())

        if int(n_a) != n_b:
            raise CheckFailure(
                f"extra padding changed the supervised token count {int(n_a)} -> {n_b}"
            )
        rel = abs(float(loss_a.detach()) - float(loss_b.detach())) / max(
            1e-6, abs(float(loss_a.detach()))
        )
        if rel > 1e-4:
            raise CheckFailure(f"extra padding changed the loss by {rel:.2%}")


def check_repo_no_pad_probability_mass(ws: Path, task_id: str) -> None:
    """After a short run the pad token must not dominate the prediction."""
    with RepoModules(ws) as cand:
        cfg = cand.config.Config()
        torch.manual_seed(cfg.seed)
        model = cand.model.TinyGPT(cfg)
        optimizer = cand.optim.make_optimizer(model, cfg)
        batches = cand.data.make_batches(cfg, 25, seed=cfg.seed + 1)
        for batch in batches:
            loss_sum, n_tokens = model.loss_sum(
                batch.input_ids, batch.labels, batch.padding_mask
            )
            if int(n_tokens) == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            (loss_sum / int(n_tokens)).backward()
            optimizer.step()

        probe = cand.data.collate(cfg, [cand.data.make_sequence(cfg, 12, _seeded(3))])
        with torch.no_grad():
            probs = torch.softmax(model(probe.input_ids)[0, -1], dim=-1)
        pad_mass = float(probs[cfg.pad_token])
        if pad_mass > 0.10:
            raise CheckFailure(
                f"pad token holds {pad_mass:.1%} of the probability mass after training; "
                "padded positions are being trained as targets"
            )


# --------------------------------------------------------------------------- #
# L3 -- behavioural
# --------------------------------------------------------------------------- #
def check_repo_training_stays_finite(ws: Path, task_id: str) -> None:
    with RepoModules(ws) as cand:
        cfg = cand.config.Config()
        history = cand.train.train(cfg, steps=cfg.total_steps, verbose=False)
        bad = [i for i, v in enumerate(history["loss"]) if not math.isfinite(v)]
        if bad:
            raise CheckFailure(f"loss became non-finite at step(s) {bad[:5]}")
        bad_g = [i for i, v in enumerate(history["grad_norm"]) if not math.isfinite(v)]
        if bad_g:
            raise CheckFailure(f"grad norm became non-finite at step(s) {bad_g[:5]}")


def check_repo_training_converges(ws: Path, task_id: str) -> None:
    """The reference reaches ~0.1 in 40 CPU steps; require clearly-better-than-chance.

    The bar is a band, not a point: 2.0 is far below ln(vocab)=5.55 and far above
    the reference, so ordinary run-to-run noise cannot flip it.
    """
    with RepoModules(ws) as cand:
        cfg = cand.config.Config()
        history = cand.train.train(cfg, steps=cfg.total_steps, verbose=False)
        final = history["final_loss"]
        if not math.isfinite(final):
            raise CheckFailure("final loss is not finite")
        if final > 2.0:
            raise CheckFailure(
                f"final loss {final:.3f} after {cfg.total_steps} steps; the reference "
                f"reaches ~0.1 and chance is {math.log(cfg.vocab_size):.2f}"
            )


def check_repo_contract_return_types(ws: Path, task_id: str) -> None:
    """L1 contract: the documented return types of the public surface.

    Verifier v2 addition. Closes the confirmed hole (VERIFIER_V2_PROTOCOL.md, H1):
    v1 accepted a submission that computed the gradient lifecycle correctly and
    returned a Tensor where the contract says float. Correctness of the maths and
    honouring the interface are different claims, and v1 could only express one.

    Type contracts only -- never formatting, naming or complexity.
    """
    with RepoModules(ws) as cand:
        cfg = cand.config.Config()
        torch.manual_seed(cfg.seed)
        model = cand.model.TinyGPT(cfg)
        batches = cand.data.make_batches(cfg, 2, seed=cfg.seed + 5)

        b = batches[0]
        for name, value, want in (
            ("Batch.input_ids", b.input_ids, torch.Tensor),
            ("Batch.labels", b.labels, torch.Tensor),
            ("Batch.padding_mask", b.padding_mask, torch.Tensor),
        ):
            if not isinstance(value, want):
                raise CheckFailure(f"{name} is {type(value).__name__}, not {want.__name__}")
        if not isinstance(b.n_supervised, int):
            raise CheckFailure(
                f"Batch.n_supervised is {type(b.n_supervised).__name__}, not int"
            )

        logits = model(b.input_ids, b.padding_mask)
        if not isinstance(logits, torch.Tensor):
            raise CheckFailure(f"forward returns {type(logits).__name__}, not Tensor")
        if logits.dtype != torch.float32:
            raise CheckFailure(f"forward returns dtype {logits.dtype}, not float32")
        if logits.shape != (b.input_ids.shape[0], b.input_ids.shape[1], cfg.vocab_size):
            raise CheckFailure(f"forward returns shape {tuple(logits.shape)}")

        out = model.loss_sum(b.input_ids, b.labels, b.padding_mask)
        if not (isinstance(out, tuple) and len(out) == 2):
            raise CheckFailure("loss_sum must return a (loss_sum, n_tokens) 2-tuple")
        loss_val, n_tok = out
        # The loss MUST be a Tensor -- backward() depends on it. The token count
        # is a scalar every call site immediately coerces, so int and Tensor are
        # both faithful to the interface. Demanding Tensor there would enforce an
        # implementation detail, which VERIFIER_V2_PROTOCOL.md explicitly excludes
        # from contract checking. See PROTOCOL_CHANGELOG R4.
        if not isinstance(loss_val, torch.Tensor):
            raise CheckFailure(
                f"loss_sum[0] is {type(loss_val).__name__}, not Tensor; backward() "
                "requires a Tensor"
            )
        if not isinstance(n_tok, (int, torch.Tensor)):
            raise CheckFailure(
                f"loss_sum[1] is {type(n_tok).__name__}; expected an integer scalar "
                "(int or Tensor)"
            )
        if int(n_tok) < 0:
            raise CheckFailure("loss_sum[1] must be a non-negative token count")

        opt = cand.optim.make_optimizer(model, cfg)
        if not isinstance(opt, torch.optim.Optimizer):
            raise CheckFailure(f"make_optimizer returns {type(opt).__name__}")
        sched = cand.optim.make_scheduler(opt, cfg)
        for attr in ("step", "get_last_lr"):
            if not callable(getattr(sched, attr, None)):
                raise CheckFailure(f"scheduler is missing callable {attr}()")

        loss = cand.train.accumulate_gradients(model, batches)
        # `type() is float`, not isinstance: numpy.float64 subclasses float, so
        # isinstance would silently accept a numpy scalar. That matters because a
        # numpy scalar breaks json.dumps downstream while looking identical here.
        # E2 in VERIFIER_V2_PROTOCOL.md exists to pin this.
        if type(loss) is not float:  # noqa: E721 - exact type is the contract
            raise CheckFailure(
                f"accumulate_gradients returns {type(loss).__name__}, not a Python "
                "float; the documented contract is exactly float"
            )

        history = cand.train.train(cfg, steps=3, verbose=False)
        if not isinstance(history, dict):
            raise CheckFailure(f"train() returns {type(history).__name__}, not dict")
        if type(history.get("final_loss")) is not float:  # noqa: E721
            raise CheckFailure(
                f"train()['final_loss'] is {type(history.get('final_loss')).__name__}, "
                "not a Python float"
            )
        for key in ("loss", "lr", "grad_norm"):
            seq = history.get(key)
            if not isinstance(seq, list) or not all(type(v) is float for v in seq):  # noqa: E721
                raise CheckFailure(f"train()['{key}'] must be a list of Python float")
        # The documented history must survive serialisation; a numpy scalar does not.
        import json as _json

        try:
            _json.dumps({k: v for k, v in history.items() if k != "all_finite"})
        except TypeError as exc:
            raise CheckFailure(f"train() history is not JSON-serialisable: {exc}")


def check_repo_contract_public_api(ws: Path, task_id: str) -> None:
    """L1 contract: module-level public signatures match the reference.

    AST comparison, so it cannot be fooled by a runtime shim, and it ignores
    everything inside function bodies.
    """
    import ast

    gold = gold_repo(task_id)
    problems: list[str] = []

    for module in sorted((Path(ws) / "tinygpt").glob("*.py")):
        rel = f"tinygpt/{module.name}"
        gold_path = gold / rel
        if not gold_path.exists():
            problems.append(f"{rel}: not part of the reference package")
            continue

        def signatures(text: str) -> dict[str, str]:
            # A UTF-8 BOM makes ast.parse raise on an otherwise valid file; an
            # editor artifact is not an API contract violation.
            text = text.lstrip("\ufeff")
            try:
                tree = ast.parse(text)
            except SyntaxError as exc:
                raise CheckFailure(f"{rel} does not parse: {exc}")
            found: dict[str, str] = {}
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not node.name.startswith("_"):
                        found[node.name] = ast.unparse(node.args)
                elif isinstance(node, ast.ClassDef):
                    found[node.name] = "class"
                    for sub in node.body:
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if not sub.name.startswith("_") or sub.name == "__init__":
                                found[f"{node.name}.{sub.name}"] = ast.unparse(sub.args)
            return found

        want = signatures(gold_path.read_text(encoding="utf-8"))
        have = signatures(module.read_text(encoding="utf-8"))
        for name, sig in want.items():
            if name not in have:
                problems.append(f"{rel}: public symbol {name!r} removed")
            elif have[name] != sig:
                problems.append(
                    f"{rel}: {name} signature changed -- reference({sig}) vs ({have[name]})"
                )

    if problems:
        raise CheckFailure("; ".join(problems[:4]))


def check_repo_grad_accum_matches_full_batch(ws: Path, task_id: str) -> None:
    """Accumulating over K micro-batches must equal one backward over the whole batch.

    The invariant only holds if the cross-entropy is normalized by the *global*
    supervised-token count. Re-averaging per micro-batch is the HuggingFace
    October-2024 bug; it is invisible with uniform padding and wrong the moment
    micro-batches carry different token counts, which this data always does.
    """
    # No gold handle: the ground truth here is computed below from the candidate's own
    # model, not from the reference. The unused `RepoModules(gold_repo(task_id)) as gold`
    # that used to sit here built the gold tree on every call for nothing -- flagged by
    # hand while decomposing the checks for R16, and independently by ruff as F841.
    with RepoModules(ws) as cand:
        cfg = cand.config.Config()
        torch.manual_seed(cfg.seed)
        model = cand.model.TinyGPT(cfg)
        batches = cand.data.make_batches(cfg, 3, seed=cfg.seed + 7)

        def flat_grads() -> torch.Tensor:
            return torch.cat(
                [
                    (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                    for p in model.parameters()
                ]
            )

        cand.train.accumulate_gradients(model, batches)
        g_accum = flat_grads()

        # Ground truth, computed here rather than taken from the candidate's
        # module. Batches are padded to their own widths so they cannot be
        # concatenated; summing the per-batch loss sums and dividing once by the
        # global token count is exactly equivalent to a single full-batch
        # backward, because backward is additive over the summed loss.
        model.zero_grad(set_to_none=True)
        total_sum = None
        total_tokens = 0
        for b in batches:
            loss_sum, n_tok = model.loss_sum(b.input_ids, b.labels, b.padding_mask)
            total_sum = loss_sum if total_sum is None else total_sum + loss_sum
            total_tokens += int(n_tok)
        if total_tokens == 0:
            raise CheckFailure("fixture produced no supervised tokens")
        (total_sum / total_tokens).backward()
        g_full = flat_grads()

        denom = max(1e-8, float(g_full.abs().max()))
        rel = float((g_accum - g_full).abs().max()) / denom
        if rel > 1e-4:
            raise CheckFailure(
                f"accumulated gradients differ from the full-batch gradients by "
                f"{rel:.3%}; accumulation must share one global token denominator"
            )


def check_repo_padding_mask_reaches_model(ws: Path, task_id: str) -> None:
    """Training must pass the padding mask through, not silently drop it.

    Detected by making padded key positions carry values that would change the
    output if they were attended to.
    """
    with RepoModules(ws) as cand:
        cfg = cand.config.Config()
        torch.manual_seed(cfg.seed)
        model = cand.model.TinyGPT(cfg)
        batches = cand.data.make_batches(cfg, 1, seed=cfg.seed + 11)
        batch = batches[0]
        if bool(batch.padding_mask.all()):
            raise CheckFailure("fixture has no padding; cannot test mask propagation")

        with torch.no_grad():
            masked = model(batch.input_ids, batch.padding_mask)
            unmasked = model(batch.input_ids, None)
        if torch.allclose(masked, unmasked, atol=1e-6):
            raise CheckFailure(
                "passing the padding mask changes nothing: it is being ignored"
            )

        seen = {}

        def spy(*args, **kwargs):
            seen["padding_mask"] = kwargs.get("padding_mask", args[3] if len(args) > 3 else None)
            return original(*args, **kwargs)

        original = cand.attention.causal_attention
        cand.attention.causal_attention = spy
        try:
            cand.train.train(cfg, steps=1, verbose=False)
        finally:
            cand.attention.causal_attention = original
        if seen.get("padding_mask") is None:
            raise CheckFailure(
                "the training loop calls attention without a padding mask; padded "
                "positions are being attended to during training"
            )


def check_repo_lr_schedule_matches_gold(ws: Path, task_id: str) -> None:
    """The LR trace must match the reference schedule step for step."""
    with RepoModules(ws) as cand, RepoModules(gold_repo(task_id)) as gold:
        steps = 20
        got = cand.train.train(cand.config.Config(), steps=steps, verbose=False)["lr"]
        want = gold.train.train(gold.config.Config(), steps=steps, verbose=False)["lr"]
        if len(got) != len(want):
            raise CheckFailure(f"LR trace has {len(got)} entries, expected {len(want)}")
        for i, (a, b) in enumerate(zip(got, want)):
            if abs(a - b) > 1e-9 + 1e-6 * abs(b):
                raise CheckFailure(
                    f"learning rate diverges from the reference at step {i}: "
                    f"{a:.3e} vs {b:.3e} (the schedule is advancing at the wrong rate)"
                )


def check_repo_weight_decay_excludes_gains(ws: Path, task_id: str) -> None:
    """1-D parameters (norm gains) must sit in a zero-weight-decay group."""
    with RepoModules(ws) as cand:
        cfg = cand.config.Config()
        torch.manual_seed(0)
        model = cand.model.TinyGPT(cfg)
        optimizer = cand.optim.make_optimizer(model, cfg)
        gains = {id(p) for n, p in model.named_parameters() if p.ndim < 2}
        for group in optimizer.param_groups:
            wd = group.get("weight_decay", 0.0)
            if wd == 0.0:
                continue
            leaked = [p for p in group["params"] if id(p) in gains]
            if leaked:
                raise CheckFailure(
                    f"{len(leaked)} one-dimensional parameter(s) are in a group with "
                    f"weight_decay={wd}; normalization gains must not be decayed"
                )


def _grad_norms_at_step_time(ws: Path, steps: int = 8) -> list[float]:
    """Gradient norms observed at the moment `optimizer.step()` is called.

    Instrumenting the optimizer rather than reading the training history is the
    point: the history records whatever the loop chose to report, which is
    exactly what a lifecycle bug gets wrong.
    """
    with RepoModules(ws) as cand:
        cfg = cand.config.Config()
        import tinygpt.train as train_mod  # noqa: PLC0415 - the candidate's module

        observed: list[float] = []
        original_step = torch.optim.AdamW.step

        def spy(self, *args, **kwargs):
            total = 0.0
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        total += float(p.grad.detach().norm() ** 2)
            observed.append(math.sqrt(total))
            return original_step(self, *args, **kwargs)

        torch.optim.AdamW.step = spy
        try:
            train_mod.train(cfg, steps=steps, verbose=False)
        finally:
            torch.optim.AdamW.step = original_step
        return observed


def check_repo_gradients_reach_optimizer(ws: Path, task_id: str) -> None:
    """Every update must see non-zero gradients.

    Zeroing gradients after `backward()` but before `optimizer.step()` leaves the
    loop looking perfectly healthy -- finite loss, sensible LR curve, no warning
    -- while every update is applied to nothing.
    """
    observed = _grad_norms_at_step_time(ws)
    if not observed:
        raise CheckFailure("no optimizer step was taken")
    zero_steps = [i for i, n in enumerate(observed) if n < 1e-12]
    if zero_steps:
        raise CheckFailure(
            f"optimizer.step() ran with all-zero gradients at step(s) "
            f"{zero_steps[:5]} of {len(observed)}: the update has no effect"
        )


def check_repo_clipping_is_effective(ws: Path, task_id: str) -> None:
    """After the loop clips at `grad_clip`, no update may have used a larger norm.

    Catches clipping that runs but too late to matter -- the classic
    clip-after-step ordering bug.
    """
    with RepoModules(ws) as cand:
        clip = cand.config.Config().grad_clip
    observed = _grad_norms_at_step_time(ws)
    if not observed:
        raise CheckFailure("no optimizer step was taken")
    worst = max(observed)
    if worst > clip * 1.05:
        raise CheckFailure(
            f"an update was applied with gradient norm {worst:.3f} > grad_clip="
            f"{clip}; clipping is not taking effect before the step"
        )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
CHECKS: dict[str, Callable[[Path, str], None]] = {
    name[len("check_") :]: fn
    for name, fn in list(globals().items())
    if name.startswith("check_repo_") and callable(fn)
}

LEVELS = {
    "repo_visible_smoke": 1,
    "repo_visible_loss_is_finite": 1,
    "repo_visible_single_token_attention": 1,
    "repo_visible_rope_position_zero": 1,
    "repo_visible_short_train_runs": 3,
    "repo_matches_gold_logits": 2,
    "repo_strict_causality": 2,
    "repo_padding_keys_masked": 2,
    "repo_rope_relative_property": 2,
    "repo_rope_norm_preserved": 2,
    "repo_supervised_token_count": 2,
    "repo_padding_does_not_change_loss": 2,
    "repo_no_pad_probability_mass": 3,
    "repo_training_stays_finite": 3,
    "repo_training_converges": 3,
    "repo_lr_schedule_matches_gold": 3,
    "repo_weight_decay_excludes_gains": 2,
    "repo_clipping_is_effective": 3,
    "repo_gradients_reach_optimizer": 3,
    "repo_grad_accum_matches_full_batch": 2,
    "repo_contract_return_types": 1,
    "repo_contract_public_api": 1,
    "repo_padding_mask_reaches_model": 2,
}


def run_repo_checks(
    workspace: Path, task_id: str, names: Iterable[str]
) -> list[tuple[str, bool, str]]:
    """Run `names` against the repo in `workspace`. Never raises."""
    torch.manual_seed(0)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # noqa: BLE001
        pass

    results: list[tuple[str, bool, str]] = []
    for name in names:
        fn = CHECKS.get(name)
        if fn is None:
            results.append((name, False, f"unknown check {name!r}"))
            continue
        try:
            fn(Path(workspace), task_id)
            results.append((name, True, ""))
        except CheckFailure as exc:
            results.append((name, False, str(exc)))
        except Exception as exc:  # noqa: BLE001
            results.append((name, False, f"{type(exc).__name__}: {exc}"))
    return results
