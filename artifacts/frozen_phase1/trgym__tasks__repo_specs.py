"""Tier M -- repo-level, symptom-only, multi-file debugging tasks.

The difference from Tier E is not that the mathematics is more obscure. It is
that the **debugging horizon** is longer:

  * the candidate is given a small package (8 modules + tests), not one file;
  * nothing says which file is wrong;
  * the symptom is what a user would report -- a loss curve, an LR trace, a
    plateau -- not a named property;
  * three of the five can only be observed by *running training*, not by
    reading a forward pass;
  * two of the five span more than one file, so a single-hop edit cannot fix them.

Every mutation is anchored in REAL_BUG_EVIDENCE.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from trgym.tasks.spec import Mutation, Provenance

Tier = Literal["E", "M"]


@dataclass(frozen=True)
class RepoTaskSpec:
    task_id: str
    tier: Tier
    family: str
    family_id: str
    provenance: Provenance
    symptom: str
    """What a user reports. Must not name the root cause or the file."""
    evidence: tuple[str, ...]
    mutations: dict[str, tuple[Mutation, ...]]
    """filename (relative to the repo root) -> edits to apply."""
    hidden_checks: tuple[str, ...]
    visible_checks: tuple[str, ...]
    requires_training_run: bool = False
    """True when the symptom is invisible without executing a training loop."""
    editable: tuple[str, ...] = field(
        default_factory=lambda: (
            "tinygpt/config.py",
            "tinygpt/norm.py",
            "tinygpt/positional.py",
            "tinygpt/attention.py",
            "tinygpt/model.py",
            "tinygpt/data.py",
            "tinygpt/optim.py",
            "tinygpt/train.py",
        )
    )

    def __post_init__(self) -> None:
        if not self.mutations:
            raise ValueError(f"{self.task_id}: no mutations")
        for path in self.mutations:
            if path not in self.editable:
                raise ValueError(f"{self.task_id}: mutates non-editable {path}")
        if set(self.visible_checks) & set(self.hidden_checks):
            raise ValueError(f"{self.task_id}: visible and hidden suites must be distinct")

    @property
    def n_files_touched(self) -> int:
        return len(self.mutations)


# --------------------------------------------------------------------------- #
# M1 -- causal attention regression, located by the candidate
# --------------------------------------------------------------------------- #
M1 = RepoTaskSpec(
    task_id="m1_attention_regression",
    tier="M",
    family="attention masking / causality",
    family_id="F1",
    provenance="REAL-DERIVED",
    symptom=(
        "Since last week's refactor our training loss falls much faster than it "
        "used to and settles far below anything we saw before, but the model is "
        "useless: sampled continuations are incoherent and a held-out next-token "
        "probe scores at chance. Nothing in the run looks abnormal otherwise -- "
        "no warnings, finite gradients, sensible learning-rate curve. We have not "
        "been able to work out which part of the model changed behaviour."
    ),
    evidence=(
        "https://github.com/huggingface/transformers/issues/30095",
        "https://github.com/huggingface/transformers/issues/36150",
    ),
    mutations={
        "tinygpt/attention.py": (
            Mutation(
                find="causal = torch.ones(seq_len, seq_len, dtype=torch.bool, "
                "device=scores.device).tril(diagonal=0)",
                replace="causal = torch.ones(seq_len, seq_len, dtype=torch.bool, "
                "device=scores.device).tril(diagonal=1)",
            ),
        )
    },
    hidden_checks=(
        "repo_strict_causality",
        "repo_matches_gold_logits",
        "repo_padding_keys_masked",
        "repo_training_converges",
    ),
    visible_checks=("repo_visible_smoke", "repo_visible_single_token_attention"),
)

# --------------------------------------------------------------------------- #
# M2 -- position encoding, downstream symptom only
# --------------------------------------------------------------------------- #
M2 = RepoTaskSpec(
    task_id="m2_position_encoding",
    tier="M",
    family="position encoding (RoPE)",
    family_id="F2",
    provenance="REAL-DERIVED",
    symptom=(
        "Training converges, but it plateaus noticeably higher than an equivalent "
        "run from our previous checkout, and the gap gets worse the longer the "
        "sequences are. Short sequences look almost fine. We ported some code "
        "between two repositories recently and suspect something about how the "
        "model handles token order, but the shapes all match and no test fails."
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
        )
    },
    hidden_checks=(
        "repo_rope_relative_property",
        "repo_rope_norm_preserved",
        "repo_matches_gold_logits",
        "repo_training_converges",
    ),
    visible_checks=("repo_visible_smoke", "repo_visible_rope_position_zero"),
)

# --------------------------------------------------------------------------- #
# M3 -- training stability, only visible by running training
# --------------------------------------------------------------------------- #
M3 = RepoTaskSpec(
    task_id="m3_gradient_lifecycle",
    tier="M",
    family="training loop / gradient lifecycle",
    family_id="F6",
    provenance="REAL-DERIVED",
    symptom=(
        "The loss does not move. Forty steps, four hundred steps, same number, "
        "sitting right around where an untrained model starts. Everything else "
        "looks healthy: finite gradients, no warnings, a sensible learning-rate "
        "curve, and the loss for a freshly initialised model is exactly what you "
        "would predict. A single forward and backward pass in isolation looks "
        "correct too. It simply never learns anything."
    ),
    evidence=(
        "https://docs.pytorch.org/docs/stable/optim.html",
        "https://discuss.pytorch.org/t/zero-grad-optimizer-or-net/1887",
    ),
    mutations={
        "tinygpt/train.py": (
            Mutation(
                find="""    model.zero_grad(set_to_none=True)

    total_tokens = sum(mb.n_supervised for mb in micro_batches)
    if total_tokens == 0:
        return 0.0

    total_loss = 0.0
    for mb in micro_batches:
        loss_sum, _ = model.loss_sum(mb.input_ids, mb.labels, mb.padding_mask)
        (loss_sum / total_tokens).backward()
        total_loss += float(loss_sum.detach())
    return total_loss / total_tokens""",
                replace="""    total_tokens = sum(mb.n_supervised for mb in micro_batches)
    if total_tokens == 0:
        return 0.0

    total_loss = 0.0
    for mb in micro_batches:
        loss_sum, _ = model.loss_sum(mb.input_ids, mb.labels, mb.padding_mask)
        (loss_sum / total_tokens).backward()
        total_loss += float(loss_sum.detach())

    model.zero_grad(set_to_none=True)
    return total_loss / total_tokens""",
            ),
        ),
    },
    hidden_checks=(
        "repo_gradients_reach_optimizer",
        "repo_training_converges",
        "repo_clipping_is_effective",
        "repo_matches_gold_logits",
    ),
    visible_checks=("repo_visible_smoke", "repo_visible_short_train_runs"),
    requires_training_run=True,
)

# --------------------------------------------------------------------------- #
# M4 -- optimizer / scheduler / accumulation interaction, two files
# --------------------------------------------------------------------------- #
M4 = RepoTaskSpec(
    task_id="m4_schedule_accumulation",
    tier="M",
    family="optimizer / scheduler / accumulation",
    family_id="F4",
    provenance="REAL",
    symptom=(
        "Our learning rate reaches zero around halfway through every run, so the "
        "back half of the schedule does nothing and we finish several times worse "
        "than we used to. Where it happens moves when we change grad_accum_steps. "
        "Separately, the normalization gains keep drifting towards zero over a long "
        "run, which we do not think should be happening to them."
    ),
    evidence=(
        "https://github.com/pytorch/pytorch/issues/44511",
        "https://github.com/Lightning-AI/pytorch-lightning/issues/21339",
        "https://discuss.pytorch.org/t/userwarning-detected-call-of-lr-scheduler-step-before-optimizer-step/142833",
    ),
    mutations={
        "tinygpt/train.py": (
            Mutation(
                find="""    for step in range(steps):
        window = batches[step * cfg.grad_accum_steps : (step + 1) * cfg.grad_accum_steps]
        loss = accumulate_gradients(model, window)""",
                replace="""    for step in range(steps):
        window = batches[step * cfg.grad_accum_steps : (step + 1) * cfg.grad_accum_steps]
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
        "repo_training_converges",
    ),
    visible_checks=("repo_visible_smoke", "repo_visible_short_train_runs"),
    requires_training_run=True,
)

# --------------------------------------------------------------------------- #
# M5 -- data / label / loss masking interaction, two files
# --------------------------------------------------------------------------- #
M5 = RepoTaskSpec(
    task_id="m5_masking_interaction",
    tier="M",
    family="data / label construction / loss masking",
    family_id="F5",
    provenance="REAL-DERIVED",
    symptom=(
        "The reported loss for the same examples changes depending on what else "
        "happens to be in the batch, so our numbers are not comparable between "
        "runs. Sorting the dataset by length made the curve jump visibly. At "
        "inference the model puts a lot of probability on token 0, which is not a "
        "token we ever want it to emit."
    ),
    evidence=(
        "https://github.com/huggingface/transformers/issues/40214",
        "https://huggingface.co/blog/gradient_accumulation",
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
    },
    hidden_checks=(
        "repo_padding_does_not_change_loss",
        "repo_supervised_token_count",
        "repo_no_pad_probability_mass",
        "repo_training_converges",
    ),
    visible_checks=("repo_visible_smoke", "repo_visible_loss_is_finite"),
)


REPO_TASKS: tuple[RepoTaskSpec, ...] = (M1, M2, M3, M4, M5)
REPO_TASKS_BY_ID = {t.task_id: t for t in REPO_TASKS}


def _all_tasks() -> dict[str, RepoTaskSpec]:
    """Tier M plus Tier H, if the H module is present."""
    out = dict(REPO_TASKS_BY_ID)
    try:
        from trgym.tasks.repo_specs_h import REPO_TASKS_H

        out.update({t.task_id: t for t in REPO_TASKS_H})
    except ImportError:
        pass
    return out


def get_repo_task(task_id: str) -> RepoTaskSpec:
    REPO_TASKS_BY_ID.update(_all_tasks())
    if task_id not in REPO_TASKS_BY_ID:
        raise KeyError(f"unknown repo task {task_id!r}; known: {sorted(REPO_TASKS_BY_ID)}")
    return REPO_TASKS_BY_ID[task_id]
