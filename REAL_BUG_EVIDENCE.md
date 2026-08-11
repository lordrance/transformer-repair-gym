# REAL_BUG_EVIDENCE

Provenance for the five Phase-0 bug families. The rule for this file: no task
exists unless there is a public issue, PR, blog post or paper describing the
same failure happening to real people on real models.

## Provenance labels

| Label | Meaning | Tier E | Tier M |
|---|---|---|---|
| **REAL** | Reproduces a specific documented bug essentially verbatim, including its mathematical form | 1 | 1 |
| **REAL-DERIVED** | Re-instantiates a bug pattern documented in real issues/PRs inside our own model | 4 | 4 |
| **SYNTHETIC** | Invented mutation with no specific real-world referent | 0 | 0 |

Phase 0 required ≥3 REAL/REAL-DERIVED out of 5; Phase 1 required the same of the
new Tier M set. Delivered: **10/10, zero synthetic.**

Tier E (§F1–F5) is documented first; Tier M (§M1–M5) follows.

---

## F1 — Attention masking / causality

**Task:** `t1_causal_mask_off_by_one` · **Provenance:** REAL-DERIVED

**Sources**
- [huggingface/transformers#30095](https://github.com/huggingface/transformers/issues/30095) — `_prepare_4d_attention_mask_for_sdpa` is not causal but is used where causal masking is expected
- [huggingface/transformers#36150](https://github.com/huggingface/transformers/issues/36150) — `is_causal=False` silently has no effect; a causal mask is applied anyway
- [huggingface/transformers#40214](https://github.com/huggingface/transformers/issues/40214) — chunked attention with left padding truncates the first chunk, causing a train/inference mismatch

**Why it really happens.** Causal masking is expressed three different ways
across a modern codebase — an additive float mask, a boolean `tril`, and the
backend's own `is_causal` flag — and they disagree about whether the diagonal is
included and about how padding composes with causality. Every refactor that
moves between representations is an opportunity for an off-by-one.

**Symptom.** Validation loss drops *below* what the architecture should support,
while generation is incoherent. Loss looking too good is the tell: the model is
reading the token it is being asked to predict.

**Root cause (injected).** `tril(diagonal=0)` → `tril(diagonal=1)`. Position *i*
may attend to *i+1*: exactly one token of lookahead.

**How the fix is verified.** `strict_causality` perturbs a token at position *p*
and asserts positions `0..p-1` are bit-identical, across sequence lengths
3/5/16/31. `matches_reference_forward` compares logits against the pristine model
with shared weights.

**Reduction to the tiny model.** One line in `causal_attention`. No dependence on
any specific model family.

---

## F2 — Position encoding (RoPE)

**Task:** `t2_rope_pairing_convention` · **Provenance:** REAL-DERIVED

**Sources**
- [huggingface/transformers#25199](https://github.com/huggingface/transformers/issues/25199) — HF uses GPT-NeoX style `rotate_half`; Meta's LLaMA uses GPT-J style interleaving
- [huggingface/transformers#31859](https://github.com/huggingface/transformers/issues/31859) — RoPE implementation differs from the official Meta implementation
- [huggingface/transformers#33826](https://github.com/huggingface/transformers/issues/33826) — inconsistency in the LLaMA RoPE implementation
- `convert_llama_weights_to_hf.py` carries a `permute()` that exists *solely* to reconcile the two conventions

**Why it really happens.** Both conventions are correct in isolation. They are
incompatible with each other, and which one is right depends on how the QK
weights were laid out. Port a checkpoint, reimplement a kernel, or copy a
snippet from the wrong repo, and the conventions silently cross. Because the
result is still a well-formed rotation-like op, nothing raises.

**Symptom.** Training converges but plateaus higher; the gap widens with
sequence length; attention between two tokens stops depending only on their
relative distance.

**Root cause (injected).** `rotate_half` switched from halves pairing
(`i` ↔ `i + d/2`) to interleaved pairing (`2i` ↔ `2i+1`), while
`build_rope_cache` still emits `cat([freqs, freqs])` for the halves convention.

**How the fix is verified.** `rope_relative_property` asserts
⟨RoPE(q,m), RoPE(k,n)⟩ depends only on `m − n` — the defining property of RoPE,
and the thing the mismatched convention destroys. `rope_norm_preserved` checks
it is still a rotation.

**Note on the visible/hidden split.** `rope_position_zero_identity` (cos=1,
sin=0 at position 0) holds under *both* conventions, so it is a genuinely lazy
smoke test rather than one chosen to hide the bug.

---

## F3 — Normalization numerics / training stability

**Task:** `t3_rmsnorm_missing_upcast` · **Provenance:** REAL-DERIVED

**Sources**
- [huggingface/transformers#35945](https://github.com/huggingface/transformers/issues/35945) — mixed precision with `torch.autocast` is broken for many models; RMSNorm's silent upcast propagates through q_norm/v_norm
- [huggingface/transformers#24519](https://github.com/huggingface/transformers/issues/24519) — questions about the dtype handling in `modeling_llama.py`
- `LlamaRMSNorm` upstream explicitly does `hidden_states.to(torch.float32)` before computing the variance, then casts back — the upcast is load-bearing, not incidental
- PyTorch's own `LayerNorm`/`BatchNorm` upcast internal accumulations to fp32 regardless of input dtype

**Why it really happens.** The upcast is one easily-deleted line that looks
redundant in an fp32 test suite. It only matters under reduced precision, which
is exactly the configuration that is not covered by the fast CI job.

**Symptom.** fp32 runs are fine. Under fp16 the hidden states go to exactly zero
part-way through the network and the loss freezes at `ln(vocab_size)`. Nothing
raises.

**Root cause (injected).** The `.to(torch.float32)` before `x.pow(2).mean(-1)` is
removed.

**Measured, not assumed.** With fp16 activations at scale ~300, `x²` exceeds the
fp16 max of 65504 → `variance = inf` → `rsqrt(inf) = 0` → output RMS is exactly
`0.0`. With the upcast, output RMS is `1.0000`. Reproduced on CPU
(`torch 2.9.1+cpu`) before the task was written.

> bf16 is the *wrong* dtype for this task — it shares fp32's exponent range and
> does not overflow. The task uses fp16 deliberately.

**How the fix is verified.** `rms_norm_float16_no_collapse` at input scales
100/300/800; `rms_norm_dtype_preserved` across fp32/fp16/bf16.

---

## F4 — Optimizer / training loop

**Task:** `t4_grad_accum_normalization` · **Provenance:** **REAL**

**Sources**
- [huggingface.co/blog/gradient_accumulation](https://huggingface.co/blog/gradient_accumulation) — HF's own writeup of the bug and the fix
- [unsloth.ai/blog/gradient](https://unsloth.ai/blog/gradient) — the report that surfaced it, 15 Oct 2024
- Reported by Benjamin Marie; patched in `transformers` the following day; affected multi-GPU training too

**Why it really happens.** Cross-entropy for causal LM is normalized by the
number of non-ignored tokens. Gradient accumulation computes that mean
*independently per micro-batch* and then sums. When micro-batches contain
different numbers of supervised tokens — i.e. whenever padding is uneven — the
denominators do not combine, and the result is not the full-batch gradient. This
sat in essentially every popular trainer, unnoticed, because with uniform
padding it is exactly right.

**Symptom.** Changing `grad_accum_steps` changes the loss curve at constant
effective batch size. Heavier padding makes it worse.

**Root cause (injected).** Two-pass global token count replaced by per-micro-batch
`loss_sum / n_tokens` then `/ n_micro`.

**How the fix is verified.** The invariant is exact and checkable on CPU:
accumulating over K micro-batches must equal one backward over the concatenation.
`grad_accum_matches_full_batch_uneven` compares full gradient vectors under
padding patterns `(0,3,5)` and `(7,1,0,4)`; `grad_accum_invariant_to_split` (L3)
re-splits the same data one row per micro-batch and requires the same update.

**Trap.** The workspace contains a copy of `full_batch_gradients`. Grading
deliberately uses the copy in `trgym.reference` instead — see cheat `c5`.

---

## F5 — Loss construction / label masking

**Task:** `t5_loss_ignore_index_dropped` · **Provenance:** REAL-DERIVED

**Sources**
- [huggingface/transformers#40214](https://github.com/huggingface/transformers/issues/40214) — padding interacting incorrectly with the objective
- [huggingface.co/blog/gradient_accumulation](https://huggingface.co/blog/gradient_accumulation) — same writeup; the denominator is "number of non-padded, non-ignored tokens", which is precisely what this task breaks

**Why it really happens.** `ignore_index` is a keyword argument. Drop it, or
`clamp` labels to make an indexing error go away, and padding positions become
supervised targets. The loss stays finite and decreasing, so nothing looks wrong
— but its value now depends on batch composition.

**Symptom.** The reported loss for the same examples changes with padding; the
model puts probability mass on the pad token; sorting the dataset by length
makes the loss curve jump.

**Root cause (injected).** `ignore_index=ignore_index` removed from
`F.cross_entropy`, labels `clamp(min=0)`'d, and the token count changed to
`numel()`.

**How the fix is verified.** `ignore_index_excluded_from_count` (exact token
counts under two masking patterns) and `padding_does_not_change_loss` (the same
core sequence padded by 1 and by 6 must give the same loss to within 1e-4
relative).

---

---

# Tier M — repo-level, symptom-only

Five tasks over an 8-module package. Same evidence discipline; what changes is
that the candidate is not told which file is wrong, the symptom is what a user
would report, and three of the five cannot be diagnosed without running training.

## M1 — Attention regression, located by the candidate

`m1_attention_regression` · REAL-DERIVED · 1 file (`attention.py`)

Same root cause as F1 (`tril(diagonal=0)` → `tril(diagonal=1)`), same evidence
([#30095](https://github.com/huggingface/transformers/issues/30095),
[#36150](https://github.com/huggingface/transformers/issues/36150)). What is new
is the presentation: the symptom describes a loss curve that looks *too good*
plus a held-out probe at chance, and says nothing about masks.

The "too good" framing is the realistic one. Lookahead leakage does not make
training look broken; it makes it look excellent, which is why it survives code
review.

## M2 — Position encoding, downstream symptom only

`m2_position_encoding` · REAL-DERIVED · 1 file (`positional.py`)

Halves vs interleaved RoPE convention, evidence as F2
([#25199](https://github.com/huggingface/transformers/issues/25199),
[#31859](https://github.com/huggingface/transformers/issues/31859),
[#33826](https://github.com/huggingface/transformers/issues/33826)). The symptom
mentions only a higher plateau that worsens with sequence length and a recent
port between repositories — the actual circumstance under which the two
conventions get crossed. The word RoPE does not appear.

## M3 — Gradient lifecycle

`m3_gradient_lifecycle` · REAL-DERIVED · 1 file (`train.py`) · **training-run only**

`model.zero_grad(set_to_none=True)` moves from the top of `accumulate_gradients`
to the bottom. It then runs after `backward()` but before `optimizer.step()`, so
every update is applied to zero gradients.

- **Sources:** [PyTorch optim docs](https://docs.pytorch.org/docs/stable/optim.html) on the
  required `zero_grad` → `backward` → `step` ordering;
  [discuss.pytorch.org on zero_grad placement](https://discuss.pytorch.org/t/zero-grad-optimizer-or-net/1887).
- **Why it really happens:** the three calls live in different functions once a
  loop is refactored for gradient accumulation, and nothing enforces their order.
- **Measured signature:** gold 5.26 → 0.11 over 40 steps; buggy 5.26 → 5.33, with
  gradient norm **exactly 0.0** at every step. Finite throughout, no warning, LR
  curve correct.

**A negative result worth recording.** Three other candidates for this slot were
tested and rejected because their real behaviour did not match the symptom they
would have advertised:

| candidate | measured outcome |
|---|---|
| clip-after-step, `lr=3e-1` | loss spikes to 11.79 then recovers to 0.80. No NaN. |
| omit `zero_grad` entirely | final loss **0.086 — better than gold's 0.111.** Accumulated gradients act as momentum at this scale. |
| both, plus `lr=3e-2` | final 0.021. Also better. |

The general lesson: **with AdamW, gradient-magnitude bugs do not produce NaN**,
because the update is normalized per parameter and bounded by the learning rate.
"NaN after a few steps" is the wrong symptom for an Adam training loop; NaN comes
from *forward* numerics (F3's fp16 overflow) or from the loss, not from large
gradients. A task advertising a symptom its mutation does not produce would be
worse than no task.

## M4 — Scheduler / accumulation / weight-decay interaction

`m4_schedule_accumulation` · **REAL** · 2 files (`train.py`, `optim.py`) · **training-run only**

Two independent real bugs whose symptoms compound:

1. `scheduler.step()` called once per micro-batch instead of once per optimizer
   step, so the schedule advances `grad_accum_steps`× too fast.
2. Weight decay applied to every parameter including 1-D normalization gains.

- **Sources:** [pytorch#44511](https://github.com/pytorch/pytorch/issues/44511),
  [Lightning#21339](https://github.com/Lightning-AI/pytorch-lightning/issues/21339),
  and PyTorch's own `UserWarning: Detected call of lr_scheduler.step() before
  optimizer.step()` — which fires during this task's buggy run. Failure to keep
  the order "will result in PyTorch skipping the first value of the learning rate
  schedule".
- **Why it really happens:** scheduler stepping is one line, and which loop it
  belongs in is exactly the thing that gets confused when accumulation is added.
  Excluding gains from decay is a convention, not something the API enforces.
- **Measured signature:** LR reaches 0 at step 20 of 40 (gold: step 39);
  final loss 0.644 vs gold 0.111.

## M5 — Data / label / loss masking interaction

`m5_masking_interaction` · REAL-DERIVED · 2 files (`data.py`, `model.py`)

`collate` fills padded label positions with `pad_token` instead of
`ignore_index`, **and** the objective counts `numel()` rather than the unmasked
positions. Either alone is partially observable; together the loss becomes a
function of batch composition.

- **Sources:** [#40214](https://github.com/huggingface/transformers/issues/40214),
  [HF gradient-accumulation writeup](https://huggingface.co/blog/gradient_accumulation)
  (the denominator is the count of non-padded, non-ignored tokens — precisely what
  this breaks).
- **Why it really happens:** label construction and loss reduction are written by
  different people at different times, and the padding contract between them is
  implicit.
- **Measured signature:** padding the batch differently changes the reported loss;
  after a short run the pad token holds a large share of the probability mass.

## Symptom discipline

Every Tier M symptom is asserted, in
`tests/test_harness.py::test_every_task_is_reachable_through_the_harness`, not to
contain any of: `attention.py`, `positional.py`, `optim.py`, `data.py`,
`train.py`, `tril`, `rotate_half`, `zero_grad`, `ignore_index`, `weight_decay`.

A symptom that names the file or the root cause turns a repo-level task back into
a single-hop one, which is the entire difference between the two tiers.

---

## What the Tier E five have in common

None of them is a generic Python defect. There is no `NameError`, no `None`
dereference, no off-by-one in a list index. Fixing any of them requires knowing
what the tensor operation is supposed to compute:

| Task | You must understand |
|---|---|
| F1 | what causality means for an attention mask, and that the diagonal is included |
| F2 | that RoPE's defining property is translation invariance of the QK inner product |
| F3 | that variance accumulation needs more range than the activation dtype provides |
| F4 | that CE is a per-token mean, so accumulation must share one denominator |
| F5 | that padded positions are not training targets and must not enter the denominator |

That is the evidence for Q1 in the feasibility report.
