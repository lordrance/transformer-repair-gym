> **⚠️ SUPERSEDED 2026-08-10.** This is the Phase 0.5 difficulty analysis, written
> when only Tier E had been measured. Current difficulty data — Tier M, Tier H, a
> cross-model check, and a measured gradient — is in
> [`DIFFICULTY_DISTRIBUTION.md`](DIFFICULTY_DISTRIBUTION.md). Retained for the
> record of the AdamW/NaN negative result, which still stands.
# DIFFICULTY_CALIBRATION

Measured 2026-08-10 on 40 trajectories (5 tasks × 4 rollouts × 2 models).
Difficulty is measured by `human_label == TRUE_FIX` from the independent
equivalence probe, never by the reward.

## Why this document decides the project

GRPO computes advantages **within a group** of rollouts on the same prompt:

```
A_i = (r_i - mean(group)) / (std(group) + eps)
```

If every rollout in a group earns the same reward, the advantages are all zero
and the group contributes **exactly zero gradient**. This is asserted as a unit
test, not an opinion: `tests/test_minimal_grpo.py::test_constant_reward_group_produces_zero_advantage`.

A task at 100 % pass rate is therefore worth precisely as much to training as a
task at 0 %: nothing. Difficulty calibration is not polish; it is the difference
between a training run that learns and one that burns money.

## Results

| task | family | pro TRUE_FIX | flash TRUE_FIX | class |
|---|---|---|---|---|
| `t1_causal_mask_off_by_one` | F1 attention masking | 4/4 | 4/4 | **TOO_EASY** |
| `t2_rope_pairing_convention` | F2 position encoding | 4/4 | 4/4 | **TOO_EASY** |
| `t3_rmsnorm_missing_upcast` | F3 normalization numerics | 4/4 | 4/4 | **TOO_EASY** |
| `t4_grad_accum_normalization` | F4 training loop | 3/4 † | 4/4 | **TOO_EASY** |
| `t5_loss_ignore_index_dropped` | F5 loss construction | 4/4 | 4/4 | **TOO_EASY** |

† The missing rollout is a protocol slip (a dropped blank line inside the block
the model declared it was deleting), not a failure to understand the bug. The
proposed replacement code was correct. Counting it as a difficulty signal would
be wrong.

**9/10 cells TOO_EASY, 0 TOO_HARD, 0 broken.** The cheaper model scored a clean
20/20.

## Reward variance per group of 4 (hardened)

```
pro    T1 [1,1,1,1]   T2 [1,1,1,1]   T3 [1,1,1,1]   T4 [1,0,0,1]   T5 [1,1,1,1]
flash  T1 [1,1,1,1]   T2 [1,1,1,1]   T3 [1,1,1,1]   T4 [1,1,1,1]   T5 [1,1,1,1]
```

Nine of ten groups have zero variance. The one that does not owes its variance to
a patch-format error and an API-contract rejection — neither of which is a signal
about the bug. **Estimated usable gradient from this task set: zero.**

## Diagnosis: the tasks are single-hop

Each task shows the model the entire file containing the bug, describes the
symptom in terms that name the failing property, and asks for a fix. A capable
model reads the file, spots the anomaly, and writes the correction — one
inference step, no search.

The bugs are *real*; the evidence in `REAL_BUG_EVIDENCE.md` stands. They took
teams months to notice in production because those teams were not staring at a
250-line file with a note saying "the loss depends on padding". Presented this
way, the difficulty of noticing has been removed and only the difficulty of
knowing the mathematics remains — and a frontier model has that.

**The corollary matters more than the finding.** Difficulty here does not come
from bug obscurity. Making the mathematics more esoteric would produce tasks that
are unfair rather than hard. Difficulty has to come from *search*: the model must
not be able to see the answer in one look.

## Redesign options, ordered by expected value

### 1. Hide the location (highest value, cheapest)
Ship the repository, not the file. Give the failing symptom and 6–10 plausible
files; the bug is in one of them. Forces localization before repair.
*Effort:* small — the workspace builder already materializes directories.
*Risk:* prompt length grows; use retrieval or a file listing rather than pasting
everything.

### 2. Multi-turn with a real failing test
The model gets a shell: run the test, read the traceback, inspect tensors, edit,
re-run. Turns a lookup into an investigation, and produces the trajectory
structure the flagship wants anyway.
*Effort:* medium — needs the harness layer.
*Blocker:* `verifiers.v1` (harness/runtimes) **does not import on Windows**
(`fcntl`). Budget for WSL2 or a Linux host. See `LICENSE_AUDIT.md`.

### 3. Symptom-only, no root-cause hint
Current symptoms name the property that fails ("attention scores no longer depend
only on relative distance"). Replace with what a user reports: a loss curve, a
throughput regression, a checkpoint that behaves differently after a restart.
*Effort:* small. *Caveat:* raises difficulty and ambiguity together — pair with
option 2 so the model can investigate rather than guess.

### 4. Interacting bugs
Two mutations whose symptoms mask each other. Fixing one alone does not restore
correctness, so partial credit becomes meaningful and reward shaping has
something to shape.
*Effort:* medium. *Caveat:* verify solvability before shipping.

### 5. Bugs only observable through training dynamics
Correct forward and backward on a single step, wrong only over many steps —
optimizer state handling, LR-schedule ordering, EMA/checkpoint-resume bugs.
Verification needs a short controlled training run, which the L3 contract already
supports.
*Effort:* medium-high; each rollout costs seconds of CPU instead of milliseconds.

### What not to do

- **Do not delete these five.** They belong in the final mix as curriculum-easy
  anchors and as regression tests proving the verifier still discriminates.
- **Do not calibrate against a frontier model.** Q10 of the main report: the
  policy that gets trained is the one whose pass rate matters.
- **Do not add difficulty by weakening the hints without adding investigation
  tools.** That trades a solvable task for a guessing game.

## Required next measurement

Before redesigning anything, sample the **actual trainable policy** (e.g.
`Qwen3.5-4B` via Tinker) on these same 5 tasks × 4 rollouts. Cost: cents.

It is entirely possible that a task set which is trivial for `deepseek-v4-flash`
is at 0–1/4 for a 4B model — in which case the redesign target is a *curriculum*,
not harder tasks. Redesigning before that measurement would be guessing at which
direction to move.
