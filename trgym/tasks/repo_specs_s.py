"""Tier S -- repo-scale localization.

Same public API as Tier M, ~48 files instead of 8. The difference is not the difficulty
of the mathematics; it is the size of the search space. `tinygpt/attention.py` is a thin
facade, and the causal mask it eventually calls lives four levels down in
`tinygpt/_ops/masking.py`. An agent that greps for the symptom's vocabulary lands on the
facade and has to keep going.

The editable set is computed from the template rather than hand-listed. A hand-listed set
silently stops matching the moment the template gains a module, and the failure mode is
the worst kind: the file exists, the agent can read it, but an edit to it is discarded.

These are ordinary `RepoTaskSpec`s. The only tier-specific thing is `template_dir`, which
is why `gold_repo()` and the whole R16 trusted-comparator path work here unchanged --
the grading side never learns that tiers exist.
"""

from __future__ import annotations

from pathlib import Path

from trgym.tasks.repo_specs import RepoTaskSpec
from trgym.tasks.spec import Mutation

TEMPLATE_S = Path(__file__).resolve().parents[1] / "repo_template_s"


def _editable() -> tuple[str, ...]:
    """Every Python file in the Tier S package, workspace-relative and sorted."""
    root = TEMPLATE_S / "tinygpt"
    return tuple(
        sorted(
            f"tinygpt/{p.relative_to(root).as_posix()}" for p in root.rglob("*.py")
        )
    )


EDITABLE_S = _editable()

# Every Tier S task carries the full hidden suite. The point of the tier is localization,
# not a narrower oracle: a defect anywhere in 48 files must still be caught by the same
# checks that grade Tier M, or "the agent did not find it" and "the grader did not look"
# become indistinguishable.
HIDDEN_S = (
    "repo_strict_causality",
    "repo_matches_gold_logits",
    "repo_padding_keys_masked",
    "repo_supervised_token_count",
    "repo_padding_does_not_change_loss",
    "repo_no_pad_probability_mass",
    "repo_lr_schedule_matches_gold",
    "repo_training_converges",
    "repo_grad_accum_matches_full_batch",
)
VISIBLE_S = ("repo_visible_smoke", "repo_visible_loss_is_finite")


S1 = RepoTaskSpec(
    task_id="s1_causal_mask_offbyone",
    tier="S",
    family="attention masking / causality",
    family_id="F1",
    provenance="REAL-DERIVED",
    symptom=(
        "Validation loss is suspiciously low -- well below what this model size should "
        "reach on this data -- but generated continuations turn to gibberish as soon as "
        "we sample past the prompt. The training curve looks better than ever. We "
        "refactored the attention internals last sprint and split them across modules."
    ),
    evidence=(
        "https://github.com/huggingface/transformers/issues/30095",
        "Teacher-forcing leak: a mask admitting one future position makes next-token "
        "prediction near-trivial in training and useless at inference.",
    ),
    mutations={
        "tinygpt/_ops/masking.py": (
            Mutation(
                find="    return ones.tril(diagonal=0)",
                replace="    return ones.tril(diagonal=1)",
            ),
        )
    },
    hidden_checks=HIDDEN_S,
    visible_checks=VISIBLE_S,
    editable=EDITABLE_S,
    template_dir=TEMPLATE_S,
)

S2 = RepoTaskSpec(
    task_id="s2_padding_supervised_as_labels",
    tier="S",
    family="loss normalization / padding",
    family_id="F3",
    provenance="REAL-DERIVED",
    symptom=(
        "After a few dozen steps the model overwhelmingly predicts token 0, yet the "
        "loss still goes down. Batches containing short sequences seem to be hurt the "
        "most. Nothing in the training loop looks wrong and every shape checks out."
    ),
    evidence=(
        "https://github.com/huggingface/transformers/issues/28056",
        "Padded positions written as a real label rather than ignore_index are counted "
        "by the cross-entropy, so the model is explicitly trained to emit padding.",
    ),
    mutations={
        "tinygpt/_data/collate.py": (
            Mutation(
                find=(
                    "    labels = torch.full((len(sequences), width), cfg.ignore_index, "
                    "dtype=torch.long)"
                ),
                replace=(
                    "    labels = torch.full((len(sequences), width), cfg.pad_token, "
                    "dtype=torch.long)"
                ),
            ),
        )
    },
    hidden_checks=HIDDEN_S,
    visible_checks=VISIBLE_S,
    editable=EDITABLE_S,
    template_dir=TEMPLATE_S,
    requires_training_run=True,
)

S3 = RepoTaskSpec(
    task_id="s3_warmup_offbyone",
    tier="S",
    family="optimizer / schedule lifecycle",
    family_id="F4",
    provenance="REAL-DERIVED",
    symptom=(
        "The very first optimizer step appears to accomplish nothing, and our whole "
        "learning-rate curve looks shifted by one step against the reference run we "
        "keep for regression. Final loss lands close but never quite matches."
    ),
    evidence=(
        "https://github.com/huggingface/transformers/issues/21299",
        "A warmup multiplier of step/warmup rather than (step+1)/warmup makes the "
        "first step's learning rate exactly zero.",
    ),
    mutations={
        "tinygpt/_optim/schedule.py": (
            Mutation(
                find="        return (step + 1) / max(1, warmup_steps)",
                replace="        return step / max(1, warmup_steps)",
            ),
        )
    },
    hidden_checks=HIDDEN_S,
    visible_checks=VISIBLE_S,
    editable=EDITABLE_S,
    template_dir=TEMPLATE_S,
    requires_training_run=True,
)

REPO_TASKS_S: tuple[RepoTaskSpec, ...] = (S1, S2, S3)

# The files a correct repair must touch, per task. Kept next to the specs so the freeze
# script and the localization metrics read one definition rather than two.
RELEVANT_FILES: dict[str, tuple[str, ...]] = {
    "s1_causal_mask_offbyone": ("tinygpt/_ops/masking.py",),
    "s2_padding_supervised_as_labels": ("tinygpt/_data/collate.py",),
    "s3_warmup_offbyone": ("tinygpt/_optim/schedule.py",),
}
