# VERIFIER_V2_PROTOCOL

**Frozen 2026-08-10 before any verifier code was changed.** Hash recorded in
`FINAL_WORKLOG.md`. The point of freezing is that the definition of "closed" for
Gap 1 cannot be adjusted after seeing replay results.

## The confirmed hole being closed

Phase 1, `VERIFIER_FUZZ_AUDIT.md` Gap 1 (verification horizon item **H1**):

> `D1_m3_return_tensor` computes the gradient lifecycle **correctly** but returns a
> `Tensor` where the contract says `float`. The hardened verifier **passes it**.
> Measured: `hardened_FP_rate_vs_full_success = 0.10 (1/10)`.

Tier E catches this via `grad_accum_runs` (`isinstance(loss, float)`). The Tier M/H
repo suite has no L1 equivalent. The hole is structural, not a tuning artifact.

## What v2 adds — and nothing else

Two new **L1 contract** checks. No changes to L2 semantics or L3 behaviour.

### `repo_contract_return_types`

Asserts the documented return types of the public surface:

| function | required |
|---|---|
| `train.accumulate_gradients(model, batches)` | `float` |
| `train.train(cfg, steps=...)` | `dict` whose `final_loss` is `float` and `loss`/`lr`/`grad_norm` are lists of `float` |
| `TinyGPT.forward(ids)` | `torch.Tensor`, dtype `float32`, shape `(B, S, vocab)` |
| `TinyGPT.loss_sum(...)` | 2-tuple of `torch.Tensor` |
| `data.collate(cfg, seqs)` | object exposing `input_ids`, `labels`, `padding_mask` as `torch.Tensor`, and `n_supervised` as `int` |
| `optim.make_optimizer(...)` | `torch.optim.Optimizer` |
| `optim.make_scheduler(...)` | object exposing `.step()` and `.get_last_lr()` |

### `repo_contract_public_api`

Compares module-level function and class signatures against the gold repo via AST
for every editable module. Fails on a removed public symbol or a changed signature.

## Explicitly NOT correctness

The following are **not** added and must never gate reward: formatting, line
length, naming style, LOC, comment presence, cyclomatic complexity, import order.
A contract violation is a violation of a *documented interface*, not of taste.

## Reward semantics after v2

Unchanged in structure; the two labels stay separate.

| label | condition |
|---|---|
| `full_success` | L1 contract **and** L2 semantic **and** L3 behavioural all pass |
| `semantic_success` | L2 + L3 pass; L1 contract may fail |

`hardened_v2 = 1.0` iff `full_success`. So a correct computation with a broken
return type now scores `SEMANTIC_FIX`, which is exactly the distinction Phase 0.6
introduced the split for and which v1 could not express at repo level.

## FP / FN definitions for the replay (pre-declared)

Ground truth is the **independent oracle**, never the other verifier.

- **FP** — verifier passes a submission the oracle does not judge a genuine fix
  (`label ∉ {FULL_FIX, SEMANTIC_FIX}`).
- **FP vs full_success** — verifier passes a submission whose label is not
  `FULL_FIX`. This is the metric Gap 1 lives in.
- **FN** — verifier rejects a submission the oracle labels `FULL_FIX`.
- **Disagreement** — v1 and v2 verdicts differ, regardless of which is right.
- **Contract-only rejection** — v2 rejects, v1 accepts, and the oracle says
  `SEMANTIC_FIX`. This is the *intended* new behaviour and is counted separately
  from FN so it cannot be mistaken for a regression.

## Replay population

All historical real-model trajectories whose final candidate source can be
recovered without re-invoking a model:

| source | n | recovery |
|---|---|---|
| Phase 0.5 Tier E, `v4-pro` | 20 | `patched_source` field in the JSONL |
| Phase 0.5 Tier E, `v4-flash` | 20 | `patched_source` field in the JSONL |
| Tier M primary, `v4-flash` | 20 | workspace under `.sandbox_work/` |
| Tier H primary, `v4-flash` | 20 | workspace under `.sandbox_work_h/` |
| Tier M confirmatory, `v4-pro` | 10 | workspace under `.sandbox_work_pro/` |

Target: **90 trajectories**, of which the 50 multi-turn ones are the contractual
minimum. Anything unrecoverable is labelled `UNREPLAYABLE` and stays in the
denominator. Regenerating a trajectory by calling a model is **forbidden**; a
replacement, if ever needed, must be labelled `REPLACEMENT_NOT_HISTORICAL`.

## Fuzz replay

Re-run the 13 existing probes and the 7 constructed exploits under v2, plus **≥3
new contract-edge probes**:

- `E1` correct semantics, `accumulate_gradients` returns a 0-dim `Tensor`
- `E2` correct semantics, `train()` returns `final_loss` as a `numpy` scalar
- `E3` correct semantics, a public function gains an extra required parameter

Expected: v1 accepts all three, v2 rejects all three, the oracle labels all three
`SEMANTIC_FIX`. A probe that fails for a different reason is a broken probe and
must be fixed, never deleted.

## PASS criteria for G2 (all required)

1. Gap 1 closed: `D1_m3_return_tensor` and E1–E3 rejected by v2.
2. Gold passes 100 % of v2 checks on all 15 tasks.
3. No-op / buggy fails 100 % on all 15 tasks.
4. **No unexplained new rejection of a historical `FULL_FIX`** — i.e. v2 FNR on
   `FULL_FIX` is 0, and every contract-only rejection is a genuine
   `SEMANTIC_FIX`.
5. Every v2 ↔ oracle disagreement explained individually in the report.
6. Replay coverage reported; ≥90 % of the population replayed.
7. **The hole is not closed by rejecting more legitimate fixes.** Operationally:
   `v2_FP_rate ≤ v1_FP_rate` **and** the count of v2 rejections of oracle-labelled
   `FULL_FIX` is exactly 0.

Criterion 7 is the anti-cheat on this gate. Making a verifier stricter always
lowers FPR; only lowering it *without* creating false negatives is progress.
