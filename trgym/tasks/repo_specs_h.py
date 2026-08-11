"""Tier H -- interacting-bug stepping stones above Tier M.

Built after the Tier M measurement showed 3 of 5 tasks at 4/4 FULL_FIX for
`deepseek-v4-flash` (see `DIFFICULTY_DISTRIBUTION.md`). This is redesign round 1
of the **two** permitted for the whole session; the Tier M results are frozen and
retained, not replaced.

Difficulty is added along the dimensions guardrail G6
([arXiv:2603.24202](https://arxiv.org/abs/2603.24202)) endorses -- stepping stones
inside the same family, longer investigation horizon -- and explicitly **not** by
finding more obscure mathematics. Every H task is the same core capability as its
M parent:

    E (single file, location given)
      -> M (8-module repo, location hidden, symptom only)
        -> H (M, plus a SECOND interacting defect in a different module)

The mechanism that makes them harder: **fixing one defect is not enough.** A model
that localizes one bug, patches it, sees the visible tests still pass, and stops
now scores PARTIAL_FIX rather than FULL_FIX. That converts "did you find the bug"
into "did you keep going after the first find", which is the behaviour a longer
horizon is supposed to test.

Each H task also removes its parent's strongest tell, recorded per task below.
"""

from __future__ import annotations

from trgym.tasks.repo_specs import RepoTaskSpec
from trgym.tasks.spec import Mutation

# --------------------------------------------------------------------------- #
# H1 <- M1 : causal mask off-by-one AND the padding mask dropped
# --------------------------------------------------------------------------- #
H1 = RepoTaskSpec(
    task_id="h1_attention_double_defect",
    tier="M",  # tier field is E|M; H is recorded via task_id and TASK_CHAINS.md
    family="attention masking / causality",
    family_id="F1",
    provenance="REAL-DERIVED",
    symptom=(
        "Two things we cannot reconcile. Our training loss is lower than it has "
        "any right to be, yet held-out quality is worse than the older checkpoint. "
        "And the loss for a fixed set of examples shifts depending on which other "
        "examples share the batch, which makes runs incomparable. Both started "
        "after the same refactor. No warnings, finite gradients, sensible "
        "learning-rate curve."
    ),
    evidence=(
        "https://github.com/huggingface/transformers/issues/30095",
        "https://github.com/huggingface/transformers/issues/36150",
        "https://github.com/huggingface/transformers/issues/40214",
    ),
    mutations={
        "tinygpt/attention.py": (
            Mutation(
                find="causal = torch.ones(seq_len, seq_len, dtype=torch.bool, "
                "device=scores.device).tril(diagonal=0)",
                replace="causal = torch.ones(seq_len, seq_len, dtype=torch.bool, "
                "device=scores.device).tril(diagonal=1)",
            ),
            Mutation(
                find="    if padding_mask is not None:\n"
                "        keep = keep & padding_mask[:, None, None, :]",
                replace="    # padding handled downstream\n"
                "    if False:\n"
                "        keep = keep & padding_mask[:, None, None, :]",
            ),
        )
    },
    hidden_checks=(
        "repo_strict_causality",
        "repo_padding_keys_masked",
        "repo_matches_gold_logits",
        "repo_training_converges",
    ),
    visible_checks=("repo_visible_smoke", "repo_visible_single_token_attention"),
    # Tell removed: M1's clean "val loss too good" story is now confounded by a
    # second symptom, so the model cannot stop at the first plausible cause.
)

# --------------------------------------------------------------------------- #
# H2 <- M2 : RoPE convention AND a frequency-base change in config
# --------------------------------------------------------------------------- #
H2 = RepoTaskSpec(
    task_id="h2_position_double_defect",
    tier="M",
    family="position encoding (RoPE)",
    family_id="F2",
    provenance="REAL-DERIVED",
    symptom=(
        "Convergence plateaus higher than our previous checkout and the gap widens "
        "with sequence length. We tried reverting one file at a time and no single "
        "revert restored the old curve, which is what is confusing us. Shapes all "
        "match, no test fails, and short sequences look nearly fine."
    ),
    evidence=(
        "https://github.com/huggingface/transformers/issues/25199",
        "https://github.com/huggingface/transformers/issues/31859",
        "https://github.com/huggingface/transformers/issues/33826",
    ),
    mutations={
        "tinygpt/positional.py": (
            Mutation(
                find="""    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)""",
                replace="""    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)""",
            ),
        ),
        "tinygpt/config.py": (
            Mutation(find="    rope_theta: float = 10000.0",
                     replace="    rope_theta: float = 500.0"),
        ),
    },
    hidden_checks=(
        "repo_rope_relative_property",
        "repo_rope_norm_preserved",
        "repo_matches_gold_logits",
        "repo_training_converges",
    ),
    visible_checks=("repo_visible_smoke", "repo_visible_rope_position_zero"),
    # Tell removed: the symptom now states that no single-file revert helps,
    # which is true and forces a two-file hypothesis.
)

# --------------------------------------------------------------------------- #
# H3 <- M3 : wrong accumulation denominator AND clipping after the step
# --------------------------------------------------------------------------- #
H3 = RepoTaskSpec(
    task_id="h3_accumulation_and_clipping",
    tier="M",
    family="training loop / normalization and clipping",
    family_id="F4",
    provenance="REAL",
    symptom=(
        "Training works, just worse than it should, and the numbers are not "
        "reproducible across configurations: changing grad_accum_steps changes the "
        "loss curve even at a fixed effective batch size, and batches with heavy "
        "padding are the worst affected. We also see occasional very large updates "
        "despite having gradient clipping switched on."
    ),
    evidence=(
        "https://huggingface.co/blog/gradient_accumulation",
        "https://unsloth.ai/blog/gradient",
        "https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html",
    ),
    mutations={
        "tinygpt/train.py": (
            Mutation(
                find="""    total_loss = 0.0
    for mb in micro_batches:
        loss_sum, _ = model.loss_sum(mb.input_ids, mb.labels, mb.padding_mask)
        (loss_sum / total_tokens).backward()
        total_loss += float(loss_sum.detach())
    return total_loss / total_tokens""",
                replace="""    n_micro = len(micro_batches)
    total_loss = 0.0
    for mb in micro_batches:
        loss_sum, n_tok = model.loss_sum(mb.input_ids, mb.labels, mb.padding_mask)
        if int(n_tok) == 0:
            continue
        loss = loss_sum / n_tok
        (loss / n_micro).backward()
        total_loss += float(loss.detach())
    return total_loss / n_micro""",
            ),
            Mutation(
                find="""        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        # optimizer.step() must come before scheduler.step(); the reverse order
        # silently skips the first entry of the schedule.
        optimizer.step()
        scheduler.step()""",
                replace="""        optimizer.step()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scheduler.step()""",
            ),
        )
    },
    hidden_checks=(
        "repo_grad_accum_matches_full_batch",
        "repo_clipping_is_effective",
        "repo_gradients_reach_optimizer",
        "repo_training_converges",
    ),
    visible_checks=("repo_visible_smoke", "repo_visible_short_train_runs"),
    requires_training_run=True,
    # Tell removed: M3's signature was gradient norm exactly 0.0, which is
    # unmissable once you print it. Here training genuinely learns, just wrongly,
    # so the model has to reason about an invariant rather than spot a zero.
)

# --------------------------------------------------------------------------- #
# H4 <- M4 : schedule over-stepping, decay on gains, AND a dropped batch window
# --------------------------------------------------------------------------- #
H4 = RepoTaskSpec(
    task_id="h4_schedule_triple_defect",
    tier="M",
    family="optimizer / scheduler / data window",
    family_id="F4",
    provenance="REAL",
    symptom=(
        "The learning rate is at zero by roughly the halfway point, so the second "
        "half of every run is wasted. Our normalization gains also drift towards "
        "zero over a long run. And the number of examples we appear to consume per "
        "epoch does not match what the data loader was configured to produce, "
        "which we noticed while trying to reconcile the other two."
    ),
    evidence=(
        "https://github.com/pytorch/pytorch/issues/44511",
        "https://github.com/Lightning-AI/pytorch-lightning/issues/21339",
        "https://discuss.pytorch.org/t/userwarning-detected-call-of-lr-scheduler-step-before-optimizer-step/142833",
    ),
    mutations={
        "tinygpt/train.py": (
            Mutation(
                find="""        window = batches[step * cfg.grad_accum_steps : (step + 1) * cfg.grad_accum_steps]
        loss = accumulate_gradients(model, window)""",
                replace="""        window = batches[step * cfg.grad_accum_steps : (step + 1) * cfg.grad_accum_steps - 1]
        loss = accumulate_gradients(model, window)
        for _ in range(cfg.grad_accum_steps - 1):
            scheduler.step()""",
            ),
        ),
        "tinygpt/optim.py": (
            Mutation(
                find="""        if param.ndim >= 2:
            decay.append(param)
        else:
            no_decay.append(param)""",
                replace="""        decay.append(param)""",
            ),
        ),
    },
    hidden_checks=(
        "repo_lr_schedule_matches_gold",
        "repo_weight_decay_excludes_gains",
        "repo_grad_accum_matches_full_batch",
        "repo_training_converges",
    ),
    visible_checks=("repo_visible_smoke", "repo_visible_short_train_runs"),
    requires_training_run=True,
    # Tell removed: three defects means the LR trace alone no longer explains
    # everything the symptom describes.
)

# --------------------------------------------------------------------------- #
# H5 <- M5 : labels, loss denominator, AND the mask never reaching the model
#            *** HELD-OUT: not used for any calibration ***
# --------------------------------------------------------------------------- #
H5 = RepoTaskSpec(
    task_id="h5_masking_triple_defect",
    tier="M",
    family="data / label construction / loss / mask propagation",
    family_id="F5",
    provenance="REAL-DERIVED",
    symptom=(
        "Reported loss for the same examples depends on what else is in the batch, "
        "so we cannot compare runs. At inference the model puts a lot of mass on "
        "token 0. Sorting the dataset by length made the curve jump visibly. We "
        "have checked the loss function twice and it looks right to us."
    ),
    evidence=(
        "https://github.com/huggingface/transformers/issues/40214",
        "https://huggingface.co/blog/gradient_accumulation",
        "https://github.com/huggingface/transformers/issues/30095",
    ),
    mutations={
        "tinygpt/data.py": (
            Mutation(
                find="    labels = torch.full((len(sequences), width), cfg.ignore_index, dtype=torch.long)",
                replace="    labels = torch.full((len(sequences), width), cfg.pad_token, dtype=torch.long)",
            ),
        ),
        "tinygpt/model.py": (
            Mutation(
                find="    n_tokens = (shift_labels != ignore_index).sum()",
                replace="    n_tokens = torch.tensor(shift_labels.numel())",
            ),
        ),
        "tinygpt/train.py": (
            Mutation(
                find="        loss_sum, _ = model.loss_sum(mb.input_ids, mb.labels, mb.padding_mask)",
                replace="        loss_sum, _ = model.loss_sum(mb.input_ids, mb.labels)",
            ),
        ),
    },
    hidden_checks=(
        "repo_supervised_token_count",
        "repo_padding_does_not_change_loss",
        "repo_padding_mask_reaches_model",
        "repo_no_pad_probability_mass",
    ),
    visible_checks=("repo_visible_smoke", "repo_visible_loss_is_finite"),
    # Tell removed: the symptom now says the loss function "looks right", which is
    # true of the label construction and the mask plumbing being wrong instead.
)


REPO_TASKS_H: tuple[RepoTaskSpec, ...] = (H1, H2, H3, H4, H5)

# H5 is reserved: it is graded exactly once, at the end, and no task or verifier
# is adjusted after seeing its result. See §16 of the phase brief.
HELD_OUT_TASK_IDS = ("h5_masking_triple_defect",)
