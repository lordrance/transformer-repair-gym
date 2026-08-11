# VERIFIER_QUALITY_MATRIX

Scoring every verifier on the three axes from *The Verification Horizon*
([arXiv:2606.26300](https://arxiv.org/abs/2606.26300)): **faithfulness**,
**robustness**, **scalability**. The paper's central claim is adopted rather than
argued with:

> *"every verifier we can build is only a proxy for human intent, never the intent
> itself"* … *"no fixed reward function can remain effective as policy capability
> continues to grow; and verification must co-evolve with the generator."*

So this document does not claim a correct verifier. It states what each one
measures, where it is known to be wrong, and what it costs.

Evidence: `VERIFIER_FUZZ_AUDIT.md`, `artifacts/verifier_cost.json`,
`artifacts/verifier_fuzz_audit.json`.

---

## The three verifiers

| | naive (v2) | hardened (v1) | independent probe |
|---|---|---|---|
| role | the realistic weak grader we are studying | the reward actually proposed | audit oracle; never a reward |
| oracle location | `tests/test_visible.py` **inside the workspace** | gold repo built fresh, **outside every mount** | gold repo, separate seeds |
| what it runs | `pytest tests/test_visible.py` in the candidate's cwd | 3–4 hidden checks across L1/L2/L3 | logits + counts + training + LR trace on unseen configs |
| configurations | one fixed public fixture per property | 5 hidden shapes, unseen step counts | 8 further shapes, `SEED=424242424` |
| anti-tamper | none, deliberately | grader files regenerated; gold unreachable | n/a |

---

## Faithfulness — does reward mean "the bug is fixed"?

| verifier | score | basis |
|---|---|---|
| naive | **1 / 5** | Tests the right *properties* but at one fixed fixture, so "passes" means "is correct on this one input". Measured: **9/9** non-genuine submissions accepted. |
| hardened | **4 / 5** | 0/9 non-genuine accepted, 0/3 genuine full fixes rejected. Fails on one axis: it cannot distinguish SEMANTIC_FIX from FULL_FIX (Gap 1 below), so its 1.0 conflates "fixed and contract-clean" with "fixed". |
| independent probe | **4 / 5** | Compares against the gold repo on disjoint configurations. Blind to optimizer group structure (Gap 2), where it is *weaker than the verifier it audits*. |

Neither hardened nor the probe is scored 5. There is no evidence that would
justify it, and G3 says such evidence cannot exist for a fixed verifier.

## Robustness — does it resist shortcuts?

| probe class | naive | hardened |
|---|---|---|
| A fixture conditioning (3) | 0/3 caught | **3/3 caught** |
| B oracle / test tampering (3) | 0/3 caught | **3/3 caught** |
| C partial semantic repair (3) | 0/3 caught | **3/3 caught** |
| D contract edge (1 non-full) | 0/1 caught | **0/1 caught** ← Gap 1 |

Naive FPR **1.00**, hardened FPR **0.00**, disagreement **0.69**.

The load-bearing design decision is architectural, not heuristic: class B fails
against hardened because the gold repo is **not mounted into the container**, so
there is nothing to tamper with. Phase 0.5's `e4_t4_tamper_local_oracle`
demonstrated that no static gate catches oracle rewriting — detection was the
fallback that failed, and unreachability is what works.

## Scalability — measured, not asserted

From `scripts/measure_verifier_cost.py` (mean of 2 runs, CPU-only host):

| check level | checks | wall | per check |
|---|---|---|---|
| L1 contract | 4 | 0.04 s | 0.01 s |
| L2 semantic | 8 | 0.14 s | 0.02 s |
| L3 behavioural | 7 | 1.95 s | 0.28 s |

| task | hidden checks | wall | trains? |
|---|---|---|---|
| `m1_attention_regression` | 4 | 1.21 s | no |
| `m2_position_encoding` | 4 | 0.61 s | no |
| `m3_gradient_lifecycle` | 4 | 0.74 s | yes |
| `m4_schedule_accumulation` | 3 | 1.00 s | yes |
| `m5_masking_interaction` | 4 | 0.73 s | no |

Mean hidden suite: **0.86 s**. Tasks that run training cost **1.03×** the static
ones — the short-training checks are ~20× the per-check cost of L2, but there are
few of them, so the suite-level effect is negligible. That was worth measuring
rather than assuming, since "L3 will be too slow" was the intuition.

**The dominant cost is not verification.** Same task, same checks:

| backend | wall |
|---|---|
| in-process | **0.54 s** |
| Docker container | **8.78 s** |
| container overhead | **+8.24 s (≈16× the work)** |

Consequence for any Phase 2 that trains: at 10k rollouts, checks are ~2.4 CPU-hours
while container startup is ~23 hours. Serial per-rollout containers would be the
bottleneck. The fix is a persistent worker container or a batched grader, not
faster checks. Recorded now so it is not discovered later.

| verifier | scalability score | basis |
|---|---|---|
| naive | 4 / 5 | one `pytest` subprocess, sub-second |
| hardened | 4 / 5 | 0.86 s of real work; **2 / 5** end-to-end once container startup is included |
| independent probe | 3 / 5 | builds gold, imports two packages, runs two 24-step trainings per probe |

---

## Known verification horizon

The classes of wrong-but-passing submission we know the hardened verifier would
miss. Stating these is the point of the exercise; a verifier without a documented
horizon has simply not been audited.

| # | class | status | evidence |
|---|---|---|---|
| **H1** | Semantically correct, public API contract violated (wrong return type) | **confirmed miss** | `D1_m3_return_tensor` passes hardened. Tier E catches this via `grad_accum_runs`; Tier M has no L1 equivalent. |
| **H2** | Correct on all 5 hidden shapes, wrong on a 6th | **plausible** | Hidden shapes are fixed, not sampled per grading. A submission conditioned on the *hidden* set would pass. Untested — we have no probe for it because it requires knowing our fixtures. |
| **H3** | Optimizer/param-group semantics | **oracle weaker than verifier** | `C2` — the independent probe cannot see param groups, so FP/FN rates for this family rest on the verifier alone. |
| **H4** | Convergence within the 2.0-loss band but materially worse than gold | **by construction** | `repo_training_converges` is a band, not a comparison to gold. A fix reaching 1.9 where gold reaches 0.11 passes. Deliberate: a point threshold would be a flake. |
| **H5** | Anything requiring more than 40 training steps to manifest | **out of scope** | All L3 checks run ≤40 steps for cost reasons. Slow-onset instabilities are invisible. |
| **H6** | Behaviour under optimization pressure | **entirely untested** | No policy has been trained. G3 predicts a fixed verifier degrades as the policy strengthens; this audit cannot see that. |

## Two heuristic gates carry unmeasured false-positive risk

Tier E's `gate_no_shape_special_casing` fires on any comparison of a tensor
dimension against an integer > 2; `gate_no_silent_exception_swallowing` fires on
any broad `except`. Both patterns are legitimate in real code. Their FP rate on
human-written fixes is **unmeasured**. They did not fire on any of the four
class-D probes, which is weak positive evidence and not a measurement.

## What would move these scores

1. Add an L1 contract check to `repo/checks.py` (closes H1). Deferred to v2:
   changing the verifier mid-measurement is what `PROTOCOL_CHANGELOG.md` forbids.
2. Sample hidden configurations per grading instead of fixing them (weakens H2).
3. Probe `optimizer.param_groups` in the independent oracle (closes H3).
4. Compare final loss to gold with a ratio band (narrows H4).
5. Persistent grading container (moves hardened scalability from 2/5 to 4/5
   end-to-end).
6. Re-run this entire audit against a trained policy (the only thing that touches
   H6, and it belongs to Optional Phase 2).
