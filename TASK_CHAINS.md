# TASK_CHAINS

Stepping stones, per guardrail G6
([arXiv:2603.24202](https://arxiv.org/abs/2603.24202)): *"stepping stones, i.e.
easier and harder variants of the same core task, support curriculum-based
training"*, and *"data diversity and structure, rather than volume alone, become
the limiting factor"*.

So difficulty climbs **inside a family**, never by reaching for more obscure
mathematics. Each chain is one capability at three depths. 13 tasks total
(5 E + 5 M + 3 H sharing families with M, plus 2 H in F4/F5), under the 15 cap.

## What each tier changes

| dimension | E | M | H |
|---|---|---|---|
| files shown | 1 | 8 modules + tests | 8 modules + tests |
| bug location disclosed | **yes, in the prompt** | no | no |
| symptom names the failing property | **yes** | no — a loss curve / LR trace | no |
| number of defects | 1 | 1 | **2–3, interacting** |
| fixing one defect suffices | n/a | **yes** | **no** |
| modules that must change | 1 | 1–2 | 1–3 |
| diagnosable without running training | yes | 2 of 5 | 1 of 5 |
| turns | 1 | 14, with tools | 14, with tools |

The mechanism that makes H harder is deliberately **not** subtlety. It is that a
model which localizes one defect, patches it, sees the visible tests still pass,
and stops now earns PARTIAL_FIX. H converts *"can you find the bug"* into *"do you
keep going after the first find"*, which is what a longer investigation horizon is
supposed to test.

---

## Chain F1 — attention masking / causality

| tier | task | defect(s) | what is added |
|---|---|---|---|
| E | `t1_causal_mask_off_by_one` | `tril(diagonal=1)` | — (baseline: file given, property named) |
| M | `m1_attention_regression` | same | location hidden; symptom is "loss too good, held-out at chance" |
| **H** | `h1_attention_double_defect` | same **+ padding mask disabled** | two symptoms from one refactor; the clean "loss too good" story is confounded, so the first plausible cause is not the whole cause |

Evidence: HF [#30095](https://github.com/huggingface/transformers/issues/30095),
[#36150](https://github.com/huggingface/transformers/issues/36150),
[#40214](https://github.com/huggingface/transformers/issues/40214).

**Tell removed at H:** M1's symptom points at one thing. H1 reports two
irreconcilable observations, and neither alone explains both.

## Chain F2 — position encoding (RoPE)

| tier | task | defect(s) | what is added |
|---|---|---|---|
| E | `t2_rope_pairing_convention` | halves → interleaved | — |
| M | `m2_position_encoding` | same | location hidden; symptom is a plateau that worsens with length |
| **H** | `h2_position_double_defect` | same **+ `rope_theta` 10000 → 500** | second file; the symptom states truthfully that *no single-file revert restores the old curve*, forcing a two-file hypothesis |

Evidence: HF [#25199](https://github.com/huggingface/transformers/issues/25199)
(verbatim confirmation of the two conventions),
[#31859](https://github.com/huggingface/transformers/issues/31859),
[#33826](https://github.com/huggingface/transformers/issues/33826).

## Chain F4 — training loop / optimizer / scheduler

This chain is wider because the family is: four distinct real defects share it.

| tier | task | defect(s) | what is added |
|---|---|---|---|
| E | `t4_grad_accum_normalization` | per-micro-batch loss normalization | — (the one verbatim REAL incident) |
| M | `m4_schedule_accumulation` | scheduler over-steps **+** weight decay on norm gains | 2 files; LR trace is the observable |
| **H** | `h3_accumulation_and_clipping` | wrong accumulation denominator **+** clipping applied after `optimizer.step()` | training *learns, just wrongly*; no single number gives it away |
| **H** | `h4_schedule_triple_defect` | scheduler over-steps **+** decay on gains **+** one micro-batch dropped per window | 3 defects; the LR trace no longer explains everything the symptom describes |

Also in this chain: `m3_gradient_lifecycle` (`zero_grad` between `backward` and
`step`), evidence [PyTorch optim docs](https://docs.pytorch.org/docs/stable/optim.html).

**Tell removed at H3:** M3's signature is gradient norm **exactly 0.0** — obvious
the moment anyone prints it. H3 keeps training functional, so the model has to
reason about the accumulation *invariant* rather than spot a zero.

## Chain F5 — data / labels / loss / mask propagation

| tier | task | defect(s) | what is added |
|---|---|---|---|
| E | `t5_loss_ignore_index_dropped` | `ignore_index` removed from the objective | — |
| M | `m5_masking_interaction` | label padding written as `pad_token` **+** `numel()` denominator | 2 files, data → model |
| **H** | `h5_masking_triple_defect` | both **+ padding mask never passed to the model in `train.py`** | 3 files spanning data → model → train; the symptom says the loss function "looks right", which is true — the label construction and the mask plumbing are what is wrong |

Evidence: HF [#40214](https://github.com/huggingface/transformers/issues/40214),
[HF gradient-accumulation writeup](https://huggingface.co/blog/gradient_accumulation).

## Chain F3 — normalization numerics (E only)

`t3_rmsnorm_missing_upcast` has no M or H variant. The fp16 overflow it depends on
is a property of the *dtype*, not of repository structure, so a repo-level version
would add search without adding anything new about the bug. Left as an E-tier
anchor rather than padded out for symmetry.

---

## Held-out reservation

`h5_masking_triple_defect` is **reserved**. It is graded exactly once, at the end,
and no task or verifier is adjusted after seeing its result (phase brief §16).
Recorded in `trgym/tasks/repo_specs_h.HELD_OUT_TASK_IDS`.

The other four H tasks were built and their gold/buggy discrimination verified
before any model saw them, but they are part of redesign round 1 and so are
formally *calibration* tasks, not held-out.

## Redesign budget

| round | trigger | scope | status |
|---|---|---|---|
| 1 | Tier M measured at 3/5 tasks = 4/4 FULL_FIX for `deepseek-v4-flash` | build Tier H (5 tasks), evaluate once | **used** |
| 2 | — | — | **unused, and will stay unused if round 1 produces a spread** |

Tier M results are frozen and retained in full
(`artifacts/tier_m_primary.jsonl`, `TIER_M_REAL_MODEL_AUDIT.csv`). Tier H does not
replace them; both are reported. If after round 1 the set is still uniformly easy
or uniformly hard, the outcome is `DIFFICULTY_CALIBRATION_FAILED` and no third
round is permitted.
