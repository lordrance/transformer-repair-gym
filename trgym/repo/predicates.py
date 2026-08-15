"""The trusted half of grading. Owns gold; never runs candidate code.

Runs OUTSIDE the candidate container, after it has exited. It receives validated
observations (see `obs_protocol`), builds or consults gold, and decides PASS/FAIL. The
candidate's Python is never imported here -- only its *numbers* cross, and they arrive as
plain buffers with a checked dtype and shape.

Two properties are worth stating precisely, because the difference matters:

  * A gold-comparison predicate is **unforgeable**. The candidate cannot know the value
    it would have to claim, because gold is absent from the container it ran in.
  * A predicate over purely candidate-internal quantities is **forgeable** -- a candidate
    that fabricates its own gradient norms can pass `gradients_reach_optimizer`. Moving
    the comparison out of the candidate process does not change that, and pretending
    otherwise would repeat the mistake R14 and R15 were both about. These are marked
    `forgeable=True` below and listed in SECURITY_MODEL.md.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any, Callable

import torch

from trgym.repo.visible_runtime import CheckFailure

HIDDEN_SHAPES = ((1, 9), (2, 17), (3, 5), (1, 31), (4, 12))
ATOL, RTOL = 1e-5, 1e-4
GOLD_SEED = 20260810


def _seeded(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# --------------------------------------------------------------------------- #
# Which observation group each check consumes
# --------------------------------------------------------------------------- #
CHECK_GROUP: dict[str, str] = {
    "repo_matches_gold_logits": "gold_logits",
    "repo_strict_causality": "strict_causality",
    "repo_padding_keys_masked": "padding_keys_masked",
    "repo_rope_relative_property": "rope_relative",
    "repo_rope_norm_preserved": "rope_norm",
    "repo_supervised_token_count": "supervised_token_count",
    "repo_padding_does_not_change_loss": "padding_does_not_change_loss",
    "repo_no_pad_probability_mass": "no_pad_probability_mass",
    "repo_training_stays_finite": "training_history",
    "repo_training_converges": "training_history",
    "repo_lr_schedule_matches_gold": "lr_schedule",
    "repo_weight_decay_excludes_gains": "weight_decay_groups",
    "repo_clipping_is_effective": "grad_norms_at_step_time",
    "repo_gradients_reach_optimizer": "grad_norms_at_step_time",
    "repo_grad_accum_matches_full_batch": "grad_accum",
    "repo_contract_return_types": "contract_types",
    "repo_padding_mask_reaches_model": "padding_mask_reaches_model",
    "repo_visible_smoke": "visible_smoke",
    "repo_visible_loss_is_finite": "visible_loss_is_finite",
    "repo_visible_single_token_attention": "visible_single_token_attention",
    "repo_visible_rope_position_zero": "visible_rope_position_zero",
    "repo_visible_short_train_runs": "visible_short_train_runs",
    # Pure AST over source the trusted side can read for itself. No candidate execution
    # is involved at all, so this one needs no observation group.
    "repo_contract_public_api": "",
}

FORGEABLE = {
    "repo_strict_causality", "repo_padding_does_not_change_loss",
    "repo_no_pad_probability_mass", "repo_training_stays_finite",
    "repo_training_converges", "repo_weight_decay_excludes_gains",
    "repo_clipping_is_effective", "repo_gradients_reach_optimizer",
    "repo_grad_accum_matches_full_batch", "repo_contract_return_types",
    "repo_padding_mask_reaches_model", "repo_visible_smoke",
    "repo_visible_loss_is_finite", "repo_visible_short_train_runs",
}
"""Checks whose ground truth is entirely internal to the candidate.

v0.2-B moved five out of this set by anchoring them to gold on a trusted-supplied fixture:
`repo_padding_keys_masked`, `repo_rope_relative_property`, `repo_rope_norm_preserved`,
`repo_visible_single_token_attention`, `repo_visible_rope_position_zero`. Each is a pure
function with a unique correct answer, so requiring a match to gold cannot fail a
legitimately-different implementation — and a candidate cannot fabricate the answer,
because it does not have gold.

What remains here is what genuinely cannot be converted. The training-dynamics checks
(convergence, clipping, gradient flow, accumulation) describe the candidate's own
trajectory, which is *supposed* to diverge from gold's when the candidate is buggy;
comparing them to gold would be wrong, not stricter. `repo_contract_return_types` reports
Python types that only running the candidate can observe. See SECURITY_MODEL.md.
"""

GOLD_ANCHORED = {
    "repo_matches_gold_logits", "repo_supervised_token_count",
    "repo_lr_schedule_matches_gold", "repo_contract_public_api",
    "repo_padding_keys_masked", "repo_rope_relative_property",
    "repo_rope_norm_preserved", "repo_visible_single_token_attention",
    "repo_visible_rope_position_zero",
}
"""Checks a candidate cannot pass by fabricating its own numbers."""


def groups_for(names) -> list[str]:
    """The observation groups needed to decide `names`, deduplicated, order preserved."""
    out: list[str] = []
    for name in names:
        grp = CHECK_GROUP.get(name)
        if grp and grp not in out:
            out.append(grp)
    return out


# --------------------------------------------------------------------------- #
# Public task inputs sent INTO the candidate container
# --------------------------------------------------------------------------- #
def build_inputs(task_id: str, groups) -> dict:
    """The minimum public inputs the candidate needs. Nothing gold-derived.

    Everything here is uniform random noise or a token id: it carries no information about
    gold's *implementation*, which is the thing being protected. Drawing the fixtures on
    this side rather than in the container has a second purpose beyond determinism -- it
    stops a candidate choosing inputs that flatter it, and it is what makes the gold
    cross-checks in v0.2-B possible, because the trusted side can only recompute gold's
    answer for inputs it knows.
    """
    from trgym.repo.obs_protocol import encode_tensor

    inputs: dict = {}

    if "gold_logits" in groups:
        cfg = _gold_config(task_id)
        ids = [
            torch.randint(1, cfg.vocab_size, (b, s), generator=_seeded(s * 31 + b))
            for b, s in HIDDEN_SHAPES
        ]
        inputs["gold_logit_ids"] = [encode_tensor(t) for t in ids]

    if "padding_keys_masked" in groups:
        q, k, v, v2, pad = _masking_fixture()
        inputs["mask_q"] = encode_tensor(q)
        inputs["mask_k"] = encode_tensor(k)
        inputs["mask_v"] = encode_tensor(v)
        inputs["mask_v2"] = encode_tensor(v2)
        inputs["mask_pad"] = encode_tensor(pad)

    if "rope_relative" in groups:
        q, k = _rope_fixture()
        inputs["rope_q"] = encode_tensor(q)
        inputs["rope_k"] = encode_tensor(k)

    return inputs


# --------------------------------------------------------------------------- #
# Trusted-side fixtures, shared by `build_inputs` and the gold cross-checks so the
# two cannot drift. Both sides must draw byte-identical tensors or every comparison
# below is meaningless.
# --------------------------------------------------------------------------- #
def _masking_fixture():
    b, h, s, hd = 2, 2, 9, 8
    g = _seeded(13)
    q, k, v = (torch.randn(b, h, s, hd, generator=g) for _ in range(3))
    pad = torch.ones(b, s, dtype=torch.bool)
    pad[:, 6:] = False
    v2 = v.clone()
    v2[:, :, 6:] = torch.randn(b, h, 3, hd, generator=g) * 50.0
    return q, k, v, v2, pad


def _rope_fixture():
    g = _seeded(19)
    q = torch.randn(1, 1, 1, 16, generator=g)
    k = torch.randn(1, 1, 1, 16, generator=g)
    return q, k


# --------------------------------------------------------------------------- #
# Gold, built in the trusted process only
# --------------------------------------------------------------------------- #
def _gold_config(task_id: str):
    from trgym.repo.checks import gold_repo
    from trgym.repo.visible_runtime import RepoModules

    with RepoModules(gold_repo(task_id)) as gold:
        return gold.config.Config()


def _gold_logits(task_id: str, candidate_state: dict[str, torch.Tensor], ids_list):
    """Gold's logits under the CANDIDATE's weights.

    The original check loaded gold's weights into the candidate model. Doing it the other
    way tests the same thing -- two implementations, one set of weights -- while keeping
    every gold-derived tensor on this side of the boundary. `strict=True` still catches a
    parameter layout that differs from the reference, which was the other job it did.
    """
    from trgym.repo.checks import gold_repo
    from trgym.repo.visible_runtime import RepoModules

    with RepoModules(gold_repo(task_id)) as gold:
        torch.manual_seed(GOLD_SEED)
        gm = gold.model.TinyGPT(gold.config.Config())
        try:
            gm.load_state_dict(candidate_state, strict=True)
        except Exception as exc:  # noqa: BLE001
            raise CheckFailure(f"parameter layout differs from the reference: {exc}")
        gm.eval()
        with torch.no_grad():
            return [gm(ids) for ids in ids_list]


def _gold_supervised_counts(task_id: str, seeds) -> list[int]:
    from trgym.repo.checks import gold_repo
    from trgym.repo.visible_runtime import RepoModules

    with RepoModules(gold_repo(task_id)) as gold:
        cfg = gold.config.Config()
        return [int(gold.data.make_batches(cfg, 1, seed=s)[0].n_supervised) for s in seeds]


def _gold_pure_outputs(task_id: str) -> dict:
    """Gold's answers for the pure-function fixtures. v0.2-B.

    These five checks used to be decided from quantities the candidate reported about
    itself, which an untrusted process can simply fabricate. They are now *also* compared
    against gold's output on the identical trusted-supplied input.

    The conversion is only sound because these are pure functions with a unique correct
    answer: `build_rope_cache(4, 16, 10000)` has exactly one right value, so requiring a
    match cannot fail a legitimately-different-but-correct implementation. The same
    argument does NOT hold for the training-dynamics checks, which is why they are left
    forgeable rather than converted.

    Nothing computed here ever enters the candidate container.
    """
    from trgym.repo.checks import gold_repo
    from trgym.repo.visible_runtime import RepoModules

    q, k, v, v2, pad = _masking_fixture()
    rq, rk = _rope_fixture()

    with RepoModules(gold_repo(task_id)) as gold:
        cos40, sin40 = gold.positional.build_rope_cache(40, 16, 10000.0)
        cos16, sin16 = gold.positional.build_rope_cache(16, 16, 10000.0)
        cos4, sin4 = gold.positional.build_rope_cache(4, 16, 10000.0)

        def dot(m: int, n: int) -> float:
            qm, _ = gold.positional.apply_rope(rq, rq, cos40[m : m + 1], sin40[m : m + 1])
            kn, _ = gold.positional.apply_rope(rk, rk, cos40[n : n + 1], sin40[n : n + 1])
            return float((qm * kn).sum())

        qn = torch.randn(2, 3, 16, 16, generator=_seeded(17))
        q_rot, _ = gold.positional.apply_rope(qn, qn, cos16, sin16)

        single_q = torch.randn(1, 1, 1, 16, generator=_seeded(11))

        return {
            "cos0": cos4[0],
            "sin0": sin4[0],
            "rope_dots": [[dot(9, 4), dot(14, 9)], [dot(20, 3), dot(33, 16)]],
            "q_rot": q_rot,
            "mask_out1": gold.attention.causal_attention(q, k, v, pad),
            "mask_out2": gold.attention.causal_attention(q, k, v2, pad),
            "single_out": gold.attention.causal_attention(single_q, single_q, single_q),
        }


def _agrees(got, want, *, atol: float = 1e-5, rtol: float = 1e-4) -> bool:
    return bool(torch.allclose(got, want, atol=atol, rtol=rtol))


def _gold_lr_trace(task_id: str, steps: int) -> list[float]:
    from trgym.repo.checks import gold_repo
    from trgym.repo.visible_runtime import RepoModules

    with RepoModules(gold_repo(task_id)) as gold:
        return [
            float(v)
            for v in gold.train.train(gold.config.Config(), steps=steps, verbose=False)["lr"]
        ]


# --------------------------------------------------------------------------- #
# Predicates. Each raises CheckFailure, or returns None for PASS.
# --------------------------------------------------------------------------- #
def _get(obs: dict, group: str, key: str) -> Any:
    full = f"{group}.{key}"
    if full not in obs:
        raise CheckFailure(f"the candidate returned no observation {full!r}")
    return obs[full]


def _pred_matches_gold_logits(obs, task_id, ws):
    names = _get(obs, "gold_logits", "sd_names")
    values = _get(obs, "gold_logits", "sd_values")
    if len(names) != len(values):
        raise CheckFailure("state_dict names and values disagree in length")
    state = {str(n): v for n, v in zip(names, values)}
    cand_logits = _get(obs, "gold_logits", "logits")

    cfg = _gold_config(task_id)
    ids_list = [
        torch.randint(1, cfg.vocab_size, (b, s), generator=_seeded(s * 31 + b))
        for b, s in HIDDEN_SHAPES
    ]
    if len(cand_logits) != len(ids_list):
        raise CheckFailure(
            f"expected {len(ids_list)} logit tensors, got {len(cand_logits)}"
        )
    want_all = _gold_logits(task_id, state, ids_list)
    for (b, s), got, want in zip(HIDDEN_SHAPES, cand_logits, want_all):
        if got.shape != want.shape:
            raise CheckFailure(
                f"logits shape {tuple(got.shape)} != reference {tuple(want.shape)} "
                f"at batch={b}, seq={s}"
            )
        if not torch.allclose(got, want, atol=ATOL, rtol=RTOL):
            delta = float((got - want).abs().max())
            raise CheckFailure(
                f"logits differ from the reference by {delta:.3e} at batch={b}, seq={s}"
            )


def _pred_strict_causality(obs, task_id, ws):
    lengths = _get(obs, "strict_causality", "lengths")
    bases = _get(obs, "strict_causality", "base")
    positions = _get(obs, "strict_causality", "positions")
    bumped = _get(obs, "strict_causality", "bumped")
    for s, base, pos_list, others in zip(lengths, bases, positions, bumped):
        for pos, other in zip(pos_list, others):
            if not torch.allclose(base[:, :pos], other[:, :pos], atol=ATOL, rtol=RTOL):
                delta = float((base[:, :pos] - other[:, :pos]).abs().max())
                raise CheckFailure(
                    f"changing token {pos} of a length-{s} sequence moved earlier "
                    f"positions by {delta:.3e}: the model can see the future"
                )


def _pred_padding_keys_masked(obs, task_id, ws):
    out1 = _get(obs, "padding_keys_masked", "out1")
    out2 = _get(obs, "padding_keys_masked", "out2")
    if not torch.allclose(out1[:, :, :6], out2[:, :, :6], atol=ATOL, rtol=RTOL):
        raise CheckFailure("padded key positions still influence unpadded outputs")
    # v0.2-B: the property above is a self-report until it is anchored to gold. The
    # fixture was supplied by this side, so gold's answer is computable here.
    want = _gold_pure_outputs(task_id)
    if not (_agrees(out1, want["mask_out1"]) and _agrees(out2, want["mask_out2"])):
        raise CheckFailure(
            "attention output does not match the reference on the supplied fixture"
        )


def _pred_rope_relative(obs, task_id, ws):
    pairs = _get(obs, "rope_relative", "pairs")
    dots = _get(obs, "rope_relative", "dots")
    for (first, second), (a, b) in zip(pairs, dots):
        if abs(a - b) > 1e-3 * max(1.0, abs(a)):
            raise CheckFailure(
                f"not translation invariant: <q@{first[0]},k@{first[1]}>={a:.5f} vs "
                f"<q@{second[0]},k@{second[1]}>={b:.5f} at the same distance"
            )
    want = _gold_pure_outputs(task_id)["rope_dots"]
    for (ga, gb), (a, b) in zip(want, dots):
        if abs(a - ga) > 1e-4 * max(1.0, abs(ga)) or abs(b - gb) > 1e-4 * max(1.0, abs(gb)):
            raise CheckFailure(
                f"RoPE inner products do not match the reference: got ({a:.5f}, {b:.5f}), "
                f"reference ({ga:.5f}, {gb:.5f})"
            )


def _pred_rope_norm(obs, task_id, ws):
    before = _get(obs, "rope_norm", "q_norm")
    after = _get(obs, "rope_norm", "q_rot_norm")
    if not torch.allclose(before, after, atol=1e-5):
        raise CheckFailure("RoPE must be norm-preserving")
    q_rot = _get(obs, "rope_norm", "q_rot")
    if not _agrees(q_rot, _gold_pure_outputs(task_id)["q_rot"]):
        raise CheckFailure("rotated tensor does not match the reference")


def _pred_supervised_token_count(obs, task_id, ws):
    seeds = _get(obs, "supervised_token_count", "seeds")
    n_sup = _get(obs, "supervised_token_count", "n_supervised")
    n_tok = _get(obs, "supervised_token_count", "loss_n_tokens")
    want = _gold_supervised_counts(task_id, seeds)
    for got_sup, got_tok, expected in zip(n_sup, n_tok, want):
        if int(got_sup) != expected:
            raise CheckFailure(
                f"batch reports {int(got_sup)} supervised tokens, "
                f"reference says {expected}"
            )
        if int(got_tok) != expected:
            raise CheckFailure(
                f"loss counts {int(got_tok)} tokens, reference says {expected}: "
                "padded positions are being supervised"
            )


def _pred_padding_does_not_change_loss(obs, task_id, ws):
    n_a = int(_get(obs, "padding_does_not_change_loss", "n_a"))
    n_b = int(_get(obs, "padding_does_not_change_loss", "n_b"))
    loss_a = float(_get(obs, "padding_does_not_change_loss", "loss_a"))
    loss_b = float(_get(obs, "padding_does_not_change_loss", "loss_b"))
    if n_a != n_b:
        raise CheckFailure(f"extra padding changed the supervised token count {n_a} -> {n_b}")
    rel = abs(loss_a - loss_b) / max(1e-6, abs(loss_a))
    if rel > 1e-4:
        raise CheckFailure(f"extra padding changed the loss by {rel:.2%}")


def _pred_no_pad_probability_mass(obs, task_id, ws):
    pad_mass = float(_get(obs, "no_pad_probability_mass", "pad_mass"))
    if pad_mass > 0.10:
        raise CheckFailure(
            f"pad token holds {pad_mass:.1%} of the probability mass after training; "
            "padded positions are being trained as targets"
        )


def _pred_training_stays_finite(obs, task_id, ws):
    loss = _get(obs, "training_history", "loss")
    grad = _get(obs, "training_history", "grad_norm")
    bad = [i for i, v in enumerate(loss) if not math.isfinite(v)]
    if bad:
        raise CheckFailure(f"loss became non-finite at step(s) {bad[:5]}")
    bad_g = [i for i, v in enumerate(grad) if not math.isfinite(v)]
    if bad_g:
        raise CheckFailure(f"grad norm became non-finite at step(s) {bad_g[:5]}")


def _pred_training_converges(obs, task_id, ws):
    final = float(_get(obs, "training_history", "final_loss"))
    steps = int(_get(obs, "training_history", "total_steps"))
    vocab = int(_get(obs, "training_history", "vocab_size"))
    if not math.isfinite(final):
        raise CheckFailure("final loss is not finite")
    if final > 2.0:
        raise CheckFailure(
            f"final loss {final:.3f} after {steps} steps; the reference reaches ~0.1 "
            f"and chance is {math.log(vocab):.2f}"
        )


def _pred_lr_schedule(obs, task_id, ws):
    got = _get(obs, "lr_schedule", "lr")
    steps = int(_get(obs, "lr_schedule", "steps"))
    want = _gold_lr_trace(task_id, steps)
    if len(got) != len(want):
        raise CheckFailure(f"LR trace has {len(got)} entries, expected {len(want)}")
    for i, (a, b) in enumerate(zip(got, want)):
        if abs(a - b) > 1e-9 + 1e-6 * abs(b):
            raise CheckFailure(
                f"learning rate diverges from the reference at step {i}: "
                f"{a:.3e} vs {b:.3e} (the schedule is advancing at the wrong rate)"
            )


def _pred_weight_decay_excludes_gains(obs, task_id, ws):
    decays = _get(obs, "weight_decay_groups", "weight_decay")
    leaked = _get(obs, "weight_decay_groups", "n_gains_in_group")
    for wd, n in zip(decays, leaked):
        if float(wd) != 0.0 and int(n) > 0:
            raise CheckFailure(
                f"{int(n)} one-dimensional parameter(s) are in a group with "
                f"weight_decay={wd}; normalization gains must not be decayed"
            )


def _pred_clipping_is_effective(obs, task_id, ws):
    observed = _get(obs, "grad_norms_at_step_time", "observed")
    clip = float(_get(obs, "grad_norms_at_step_time", "grad_clip"))
    if not observed:
        raise CheckFailure("no optimizer step was taken")
    worst = max(float(v) for v in observed)
    if worst > clip * 1.05:
        raise CheckFailure(
            f"an update was applied with gradient norm {worst:.3f} > grad_clip="
            f"{clip}; clipping is not taking effect before the step"
        )


def _pred_gradients_reach_optimizer(obs, task_id, ws):
    observed = _get(obs, "grad_norms_at_step_time", "observed")
    if not observed:
        raise CheckFailure("no optimizer step was taken")
    zero_steps = [i for i, n in enumerate(observed) if float(n) < 1e-12]
    if zero_steps:
        raise CheckFailure(
            f"optimizer.step() ran with all-zero gradients at step(s) "
            f"{zero_steps[:5]} of {len(observed)}: the update has no effect"
        )


def _pred_grad_accum(obs, task_id, ws):
    if int(_get(obs, "grad_accum", "total_tokens")) == 0:
        raise CheckFailure("fixture produced no supervised tokens")
    g_accum = _get(obs, "grad_accum", "g_accum")
    g_full = _get(obs, "grad_accum", "g_full")
    if g_accum.shape != g_full.shape:
        raise CheckFailure("gradient vectors have mismatched shapes")
    denom = max(1e-8, float(g_full.abs().max()))
    rel = float((g_accum - g_full).abs().max()) / denom
    if rel > 1e-4:
        raise CheckFailure(
            f"accumulated gradients differ from the full-batch gradients by "
            f"{rel:.3%}; accumulation must share one global token denominator"
        )


def _pred_contract_return_types(obs, task_id, ws):
    g = "contract_types"
    for label, key in (
        ("Batch.input_ids", "input_ids_type"),
        ("Batch.labels", "labels_type"),
        ("Batch.padding_mask", "padding_mask_type"),
    ):
        got = _get(obs, g, key)
        if got != "Tensor":
            raise CheckFailure(f"{label} is {got}, not Tensor")
    if _get(obs, g, "n_supervised_type") != "int":
        raise CheckFailure(
            f"Batch.n_supervised is {_get(obs, g, 'n_supervised_type')}, not int"
        )

    if _get(obs, g, "logits_type") != "Tensor":
        raise CheckFailure(f"forward returns {_get(obs, g, 'logits_type')}, not Tensor")
    if _get(obs, g, "logits_dtype") != "torch.float32":
        raise CheckFailure(
            f"forward returns dtype {_get(obs, g, 'logits_dtype')}, not float32"
        )
    if list(_get(obs, g, "logits_shape")) != list(_get(obs, g, "want_logits_shape")):
        raise CheckFailure(
            f"forward returns shape {tuple(_get(obs, g, 'logits_shape'))}"
        )

    if not _get(obs, g, "loss_sum_is_tuple") or int(_get(obs, g, "loss_sum_len")) != 2:
        raise CheckFailure("loss_sum must return a (loss_sum, n_tokens) 2-tuple")
    if _get(obs, g, "loss_value_type") != "Tensor":
        raise CheckFailure(
            f"loss_sum[0] is {_get(obs, g, 'loss_value_type')}, not Tensor; "
            "backward() requires a Tensor"
        )
    if _get(obs, g, "n_tokens_type") not in ("int", "Tensor"):
        raise CheckFailure(
            f"loss_sum[1] is {_get(obs, g, 'n_tokens_type')}; expected an integer "
            "scalar (int or Tensor)"
        )
    if int(_get(obs, g, "n_tokens_value")) < 0:
        raise CheckFailure("loss_sum[1] must be a non-negative token count")

    if not _get(obs, g, "optimizer_is_torch_optimizer"):
        raise CheckFailure(f"make_optimizer returns {_get(obs, g, 'optimizer_type')}")
    missing = _get(obs, g, "scheduler_missing")
    if missing:
        raise CheckFailure(f"scheduler is missing callable {missing[0]}()")

    if _get(obs, g, "accum_exact_type") != "float":
        raise CheckFailure(
            f"accumulate_gradients returns {_get(obs, g, 'accum_exact_type')}, not a "
            "Python float; the documented contract is exactly float"
        )

    if _get(obs, g, "history_type") != "dict":
        raise CheckFailure(f"train() returns {_get(obs, g, 'history_type')}, not dict")
    if _get(obs, g, "final_loss_exact_type") != "float":
        raise CheckFailure(
            f"train()['final_loss'] is {_get(obs, g, 'final_loss_exact_type')}, "
            "not a Python float"
        )
    for key in ("loss", "lr", "grad_norm"):
        if not _get(obs, g, f"{key}_is_list") or not _get(obs, g, f"{key}_all_float"):
            raise CheckFailure(f"train()['{key}'] must be a list of Python float")
    if not _get(obs, g, "history_json_serialisable"):
        raise CheckFailure(
            f"train() history is not JSON-serialisable: {_get(obs, g, 'history_json_error')}"
        )


def _pred_padding_mask_reaches_model(obs, task_id, ws):
    g = "padding_mask_reaches_model"
    if not _get(obs, g, "has_padding"):
        raise CheckFailure("fixture has no padding; cannot test mask propagation")
    masked = _get(obs, g, "masked")
    unmasked = _get(obs, g, "unmasked")
    if torch.allclose(masked, unmasked, atol=1e-6):
        raise CheckFailure("passing the padding mask changes nothing: it is being ignored")
    if not _get(obs, g, "mask_reached_attention"):
        raise CheckFailure(
            "the training loop calls attention without a padding mask; padded "
            "positions are being attended to during training"
        )


def _pred_contract_public_api(obs, task_id, ws):
    """AST only, over sources the trusted side reads itself. No candidate execution."""
    from trgym.repo.checks import gold_repo

    gold = gold_repo(task_id)
    problems: list[str] = []

    def signatures(text: str, rel: str) -> dict[str, str]:
        text = text.lstrip("﻿")
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

    for module in sorted((Path(ws) / "tinygpt").glob("*.py")):
        rel = f"tinygpt/{module.name}"
        gold_path = gold / rel
        if not gold_path.exists():
            problems.append(f"{rel}: not part of the reference package")
            continue
        want = signatures(gold_path.read_text(encoding="utf-8"), rel)
        have = signatures(module.read_text(encoding="utf-8"), rel)
        for name, sig in want.items():
            if name not in have:
                problems.append(f"{rel}: public symbol {name!r} removed")
            elif have[name] != sig:
                problems.append(
                    f"{rel}: {name} signature changed -- reference({sig}) vs ({have[name]})"
                )

    if problems:
        raise CheckFailure("; ".join(problems[:4]))


PREDICATES: dict[str, Callable] = {
    "repo_matches_gold_logits": _pred_matches_gold_logits,
    "repo_strict_causality": _pred_strict_causality,
    "repo_padding_keys_masked": _pred_padding_keys_masked,
    "repo_rope_relative_property": _pred_rope_relative,
    "repo_rope_norm_preserved": _pred_rope_norm,
    "repo_supervised_token_count": _pred_supervised_token_count,
    "repo_padding_does_not_change_loss": _pred_padding_does_not_change_loss,
    "repo_no_pad_probability_mass": _pred_no_pad_probability_mass,
    "repo_training_stays_finite": _pred_training_stays_finite,
    "repo_training_converges": _pred_training_converges,
    "repo_lr_schedule_matches_gold": _pred_lr_schedule,
    "repo_weight_decay_excludes_gains": _pred_weight_decay_excludes_gains,
    "repo_clipping_is_effective": _pred_clipping_is_effective,
    "repo_gradients_reach_optimizer": _pred_gradients_reach_optimizer,
    "repo_grad_accum_matches_full_batch": _pred_grad_accum,
    "repo_contract_return_types": _pred_contract_return_types,
    "repo_padding_mask_reaches_model": _pred_padding_mask_reaches_model,
    "repo_contract_public_api": _pred_contract_public_api,
    "repo_visible_smoke": lambda obs, t, ws: _visible_smoke(obs),
    "repo_visible_loss_is_finite": lambda obs, t, ws: _visible_loss(obs),
    "repo_visible_single_token_attention": lambda obs, t, ws: _visible_single(obs, t),
    "repo_visible_rope_position_zero": lambda obs, t, ws: _visible_rope0(obs, t),
    "repo_visible_short_train_runs": lambda obs, t, ws: _visible_short(obs),
}


def _visible_smoke(obs):
    shape = list(_get(obs, "visible_smoke", "shape"))
    want = list(_get(obs, "visible_smoke", "want_shape"))
    if shape != want:
        raise CheckFailure(f"logits shape {tuple(shape)} is wrong")
    if not _get(obs, "visible_smoke", "all_finite"):
        raise CheckFailure("logits contain NaN or inf")


def _visible_loss(obs):
    if not _get(obs, "visible_loss_is_finite", "loss_finite") or int(
        _get(obs, "visible_loss_is_finite", "n_tokens")
    ) <= 0:
        raise CheckFailure("loss is not a finite number over a positive token count")


def _visible_single(obs, task_id):
    q = _get(obs, "visible_single_token_attention", "q")
    out = _get(obs, "visible_single_token_attention", "out")
    if not torch.allclose(out, q, atol=1e-6):
        raise CheckFailure("attention over a single token must return that token's value")
    if not _agrees(out, _gold_pure_outputs(task_id)["single_out"], atol=1e-6):
        raise CheckFailure("single-token attention does not match the reference")


def _visible_rope0(obs, task_id):
    cos0 = _get(obs, "visible_rope_position_zero", "cos0")
    sin0 = _get(obs, "visible_rope_position_zero", "sin0")
    if not torch.allclose(cos0, torch.ones(16), atol=1e-6):
        raise CheckFailure("cos at position 0 must be all ones")
    if not torch.allclose(sin0, torch.zeros(16), atol=1e-6):
        raise CheckFailure("sin at position 0 must be all zeros")
    want = _gold_pure_outputs(task_id)
    if not (_agrees(cos0, want["cos0"], atol=1e-6)
            and _agrees(sin0, want["sin0"], atol=1e-6)):
        raise CheckFailure("RoPE cache at position 0 does not match the reference")


def _visible_short(obs):
    loss = _get(obs, "visible_short_train_runs", "loss")
    if not all(math.isfinite(float(v)) for v in loss):
        raise CheckFailure(f"loss became non-finite within 5 steps: {loss}")


# --------------------------------------------------------------------------- #
def evaluate(
    task_id: str,
    names,
    observations: dict,
    errors: dict,
    workspace: Path,
) -> list[tuple[str, bool, str]]:
    """Decide every check in `names`. Never raises."""
    out: list[tuple[str, bool, str]] = []
    for name in names:
        fn = PREDICATES.get(name)
        if fn is None:
            out.append((name, False, f"unknown check {name!r}"))
            continue
        group = CHECK_GROUP.get(name, "")
        if group and group in errors:
            # The observation could not be produced at all -- usually the candidate's
            # package does not import. That is a failure of the check, reported with the
            # candidate's own traceback rather than a generic message.
            out.append((name, False, errors[group]))
            continue
        try:
            fn(observations, task_id, workspace)
            out.append((name, True, ""))
        except CheckFailure as exc:
            out.append((name, False, str(exc)))
        except Exception as exc:  # noqa: BLE001
            out.append((name, False, f"{type(exc).__name__}: {exc}"))
    return out
