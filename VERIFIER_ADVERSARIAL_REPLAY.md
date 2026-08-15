# VERIFIER_ADVERSARIAL_REPLAY

v0.2-A. What does verifier v2 actually reject that v1 accepts?

Artifact: `artifacts/verifier_adversarial_replay.json`.
Harness: `scripts/verifier_adversarial_replay.py`.

---

## The problem

G2's ordinary replay reports `v1_v2_disagreements == 0` across 89 real trajectories. v2 never
once decided differently from v1, so the gate passes its frozen criterion
(`v2_FPR ≤ v1_FPR`) while demonstrating nothing about the hardening.

G9 stage D found the same hole from the other direction: **nothing exercised the two contract
checks at all.** `repo_contract_return_types` and `repo_contract_public_api` could have been
deleted outright and the entire suite would still have been green.

Both observations have one explanation. v2 = v1 + those two L1 checks, and a real agent
almost never produces a tree that is *semantically correct but violates the documented
interface*. It either fixes the bug or it does not. The distinguishing population exists;
ordinary rollouts just do not sample from it.

So this builds it.

## Method

Each adversarial case starts from **gold** and applies exactly one edit that is semantically
harmless but contract-breaking. If v2 does anything, v1 accepts these and v2 rejects them.

Cases were written from the documented types *before* the suite was first run, each carries a
pre-registered `expect_v1_accept` / `expect_v2_accept`, and none was tuned afterwards. Cases
where v1 also rejects are reported as **not-distinguishing** rather than deleted — "the edit
was less harmless than intended" is a result.

Two controls, both load-bearing:

| control | requirement | observed |
|---|---|---|
| `control_gold` | unmodified gold — **both must ACCEPT**, or the harness is broken | v1 ACCEPT, v2 ACCEPT ✅ |
| `control_semantic_defect` | a real causal-mask bug — **both must REJECT**, showing v1 is narrow, not blind | v1 REJECT, v2 REJECT ✅ |

## Result

```
ordinary replay, v1/v2 disagreements     0 / 89
adversarial replay, v1 accepted         10 / 11
adversarial replay, v2 rejected         11 / 11
v1-accept AND v2-reject (distinguishing) 10 / 11
all pre-registered expectations met      yes
```

| case | category | v1 | v2 | rejected by |
|---|---|---|---|---|
| `accum_returns_tensor` | return type | accept | **reject** | `repo_contract_return_types` |
| `accum_returns_numpy_scalar` | return type | accept | **reject** | `repo_contract_return_types` |
| `history_loss_numpy` | return type | accept | **reject** | `repo_contract_return_types` |
| `final_loss_numpy` | return type | accept | **reject** | `repo_contract_return_types` |
| `loss_sum_returns_list` | return type | accept | **reject** | `repo_contract_return_types` |
| `n_supervised_returns_tensor` | return type | accept | **reject** | `repo_contract_return_types` |
| `scheduler_missing_get_last_lr` | return type | accept | **reject** | `repo_contract_return_types` |
| `public_api_extra_param` | public API | accept | **reject** | `repo_contract_public_api` |
| `public_api_objective_signature` | public API | accept | **reject** | `repo_contract_public_api` |
| `public_api_norm_default` | public API | accept | **reject** | `repo_contract_public_api` |
| `public_api_method_signature` | public API | reject | reject | both — *not distinguishing* |

The one non-distinguishing case was **predicted to behave that way**: making a documented
positional parameter keyword-only breaks every existing call site, so v1 catches it on
semantics before the contract layer is reached. It is kept because a suite where every case
succeeds by construction is a suite that cannot surprise anyone.

## What may and may not be concluded

**Supported.** Normal agent trajectories essentially never exercise the hardened contract
layer, while targeted adversarial replay shows precisely which violations v2 rejects: seven
return-type violations and three public-signature changes that v1 accepts without complaint.
The most instructive is `accum_returns_numpy_scalar` — `numpy.float64` subclasses `float`, so
an `isinstance` check waves it through and `json.dumps` refuses it three steps later. Only an
exact-type check catches that class of defect, which is what H1 in
`VERIFIER_V2_PROTOCOL.md` was written for.

**Not supported.** This says nothing about how often such violations occur in the wild. All
11 cases are constructed. The population was built specifically to separate the verifiers, so
the 10/11 rate is a property of the construction, not a base rate — quoting it as "v2 catches
91 % of contract violations" would be meaningless.

**Unchanged.** G2's ordinary-replay figures stand as they were: coverage 0.9889, v1 FPR
0.2778, v2 FPR 0.2778, `v2_FN_count` 0. Nothing here revises them, and v2 still may not be
described as reducing false positives on real trajectories.

## Reproduce

```bash
uv run python scripts/verifier_adversarial_replay.py
```

Deterministic, in-process, no Docker, no model calls, a few seconds. Accept only if both
controls behave as specified: a suite in which gold is rejected, or in which the real
semantic defect slips through, is measuring nothing.
