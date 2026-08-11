"""The five Phase-0 tasks, one per bug family.

Every mutation below is anchored to REAL_BUG_EVIDENCE.md. None of them is a
generic Python defect: fixing any of them requires knowing what the tensor
operation is *supposed* to compute.
"""

from __future__ import annotations

from trgym.tasks.spec import Mutation, TaskSpec

# --------------------------------------------------------------------------- #
# F1 - attention masking: off-by-one in the causal mask
# --------------------------------------------------------------------------- #
T1 = TaskSpec(
    task_id="t1_causal_mask_off_by_one",
    family="attention masking / causality",
    family_id="F1",
    provenance="REAL-DERIVED",
    target_file="tiny_gpt.py",
    symptom=(
        "Validation loss looks great -- it drops well below what we get from any "
        "comparable model of this size -- but sampled text is incoherent and the "
        "model scores at chance on a held-out next-token probe. The gap appeared "
        "after a refactor of the attention mask."
    ),
    evidence=(
        "https://github.com/huggingface/transformers/issues/30095",
        "https://github.com/huggingface/transformers/issues/36150",
    ),
    mutations=(
        Mutation(
            find="causal = torch.ones(seq_len, seq_len, dtype=torch.bool, "
            "device=scores.device).tril(diagonal=0)",
            replace="causal = torch.ones(seq_len, seq_len, dtype=torch.bool, "
            "device=scores.device).tril(diagonal=1)",
        ),
    ),
    hidden_checks=(
        "forward_shapes",
        "strict_causality",
        "single_token_identity",
        "padding_keys_masked",
        "matches_reference_forward",
    ),
    visible_checks=("visible_forward_smoke", "visible_causality_len8"),
)

# --------------------------------------------------------------------------- #
# F2 - position encoding: interleaved vs halves RoPE convention
# --------------------------------------------------------------------------- #
T2 = TaskSpec(
    task_id="t2_rope_pairing_convention",
    family="position encoding (RoPE)",
    family_id="F2",
    provenance="REAL-DERIVED",
    target_file="tiny_gpt.py",
    symptom=(
        "Training converges but plateaus noticeably higher than expected, and the "
        "gap widens with sequence length. Attention scores between two tokens no "
        "longer depend only on their relative distance."
    ),
    evidence=(
        "https://github.com/huggingface/transformers/issues/25199",
        "https://github.com/huggingface/transformers/issues/31859",
        "https://github.com/huggingface/transformers/issues/33826",
    ),
    mutations=(
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
    hidden_checks=(
        "forward_shapes",
        "rope_norm_preserved",
        "rope_position_zero_identity",
        "rope_relative_property",
        "matches_reference_forward",
    ),
    visible_checks=("visible_forward_smoke", "visible_rope_relative_len16"),
)

# --------------------------------------------------------------------------- #
# F3 - normalization numerics: missing fp32 upcast in RMSNorm
# --------------------------------------------------------------------------- #
T3 = TaskSpec(
    task_id="t3_rmsnorm_missing_upcast",
    family="normalization numerics / training stability",
    family_id="F3",
    provenance="REAL-DERIVED",
    target_file="tiny_gpt.py",
    symptom=(
        "Runs in float32 are fine. The moment we switch the activations to "
        "float16 the hidden states go to exactly zero part-way through the "
        "network and the loss freezes at ln(vocab_size). Nothing raises."
    ),
    evidence=(
        "https://github.com/huggingface/transformers/issues/35945",
        "https://github.com/huggingface/transformers/blob/main/src/transformers/"
        "models/llama/modeling_llama.py",
    ),
    mutations=(
        Mutation(
            find="""    input_dtype = x.dtype
    x32 = x.to(torch.float32)
    variance = x32.pow(2).mean(dim=-1, keepdim=True)
    x32 = x32 * torch.rsqrt(variance + eps)
    # Cast BOTH operands back, otherwise fp32 weight * bf16 activations silently
    # promotes the whole residual stream back to fp32 (HF issue #35945).
    return weight.to(input_dtype) * x32.to(input_dtype)""",
            replace="""    input_dtype = x.dtype
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight.to(input_dtype) * x""",
        ),
    ),
    hidden_checks=(
        "forward_shapes",
        "rms_norm_matches_float32_math",
        "rms_norm_dtype_preserved",
        "rms_norm_float16_no_collapse",
        "rms_norm_matches_reference_fp16",
        "matches_reference_forward",
    ),
    visible_checks=("visible_forward_smoke", "visible_rms_norm_fp16_scale300"),
)

# --------------------------------------------------------------------------- #
# F4 - training loop: gradient accumulation loss normalization
# --------------------------------------------------------------------------- #
T4 = TaskSpec(
    task_id="t4_grad_accum_normalization",
    family="optimizer / training loop",
    family_id="F4",
    provenance="REAL",
    target_file="train_loop.py",
    symptom=(
        "Changing grad_accum_steps changes the loss curve, even though the "
        "effective batch size is held constant. Larger accumulation makes the "
        "reported loss smaller and the model trains measurably worse. Batches "
        "with heavy padding are the worst affected."
    ),
    evidence=(
        "https://huggingface.co/blog/gradient_accumulation",
        "https://unsloth.ai/blog/gradient",
    ),
    mutations=(
        Mutation(
            find="""    total_tokens = 0
    with torch.no_grad():
        for mb in micro_batches:
            shift_labels = mb.labels[:, 1:]
            total_tokens += int((shift_labels != model.cfg.ignore_index).sum())

    if total_tokens == 0:
        return 0.0

    total_loss = 0.0
    for mb in micro_batches:
        loss_sum, _ = model.loss_sum(mb.input_ids, mb.labels, mb.padding_mask)
        # Normalize by the GLOBAL token count, not this micro-batch's count.
        (loss_sum / total_tokens).backward()
        total_loss += float(loss_sum.detach())

    return total_loss / total_tokens""",
            replace="""    n_micro = len(micro_batches)
    if n_micro == 0:
        return 0.0

    total_loss = 0.0
    for mb in micro_batches:
        loss_sum, n_tokens = model.loss_sum(mb.input_ids, mb.labels, mb.padding_mask)
        if int(n_tokens) == 0:
            continue
        loss = loss_sum / n_tokens
        (loss / n_micro).backward()
        total_loss += float(loss.detach())

    return total_loss / n_micro""",
        ),
    ),
    hidden_checks=(
        "grad_accum_runs",
        "grad_accum_matches_full_batch_even",
        "grad_accum_matches_full_batch_uneven",
        "grad_accum_invariant_to_split",
    ),
    visible_checks=("visible_grad_accum_vs_local_oracle",),
    support_files=("tiny_gpt.py",),
)

# --------------------------------------------------------------------------- #
# F5 - loss construction: ignore_index dropped from the objective
# --------------------------------------------------------------------------- #
T5 = TaskSpec(
    task_id="t5_loss_ignore_index_dropped",
    family="loss construction / label masking",
    family_id="F5",
    provenance="REAL-DERIVED",
    target_file="tiny_gpt.py",
    symptom=(
        "The reported loss for the same examples changes when we change how much "
        "the batch is padded, and the model puts a lot of probability mass on the "
        "pad token at inference. Sorting the dataset by length made the loss curve "
        "visibly jump."
    ),
    evidence=(
        "https://github.com/huggingface/transformers/issues/40214",
        "https://huggingface.co/blog/gradient_accumulation",
    ),
    mutations=(
        Mutation(
            find="""    loss_sum = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="sum",
    )
    n_tokens = (shift_labels != ignore_index).sum()""",
            replace="""    safe_labels = shift_labels.clamp(min=0)
    loss_sum = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        safe_labels.view(-1),
        reduction="sum",
    )
    n_tokens = torch.tensor(safe_labels.numel())""",
        ),
    ),
    hidden_checks=(
        "forward_shapes",
        "loss_runs_and_is_finite",
        "loss_is_next_token",
        "ignore_index_excluded_from_count",
        "padding_does_not_change_loss",
    ),
    visible_checks=("visible_forward_smoke", "visible_loss_smoke", "visible_ignore_index_count_len12"),
)


TASKS: tuple[TaskSpec, ...] = (T1, T2, T3, T4, T5)
TASKS_BY_ID = {t.task_id: t for t in TASKS}


def get_task(task_id: str) -> TaskSpec:
    if task_id not in TASKS_BY_ID:
        raise KeyError(f"unknown task {task_id!r}; known: {sorted(TASKS_BY_ID)}")
    return TASKS_BY_ID[task_id]
