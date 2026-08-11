# TURN_BUDGET_ABLATION_PROTOCOL

**Pre-registered 2026-08-10, before any 24-turn API call.** Frozen so the subset
and the outcome definition cannot be chosen after seeing results.

## The confound being measured

Phase 1 recorded **39 of 40** `deepseek-v4-flash` episodes ending because the
14-turn budget ran out, not because the agent judged itself finished. Tier H was
20/20 budget-exhausted. Two Tier H tasks scored 0/4 and were labelled `TOO_HARD`.

That label is not currently defensible. A task where the agent never voluntarily
stops may be **horizon-limited**, not capability-limited, and the Phase 1 design
cannot distinguish those.

## Hypothesis

H₀: doubling the turn budget from 14 to 24 does not change Tier H outcomes.
H₁: it does, in which case Phase 1's `TOO_HARD` labels are partly an artifact of
the horizon.

**No result direction is required to pass G3.** The gate is about quantifying the
confound, not about 24 turns helping.

## Frozen configuration

| | value |
|---|---|
| task set | **all 5 Tier H tasks**, unmodified, from manifest `tier_h_v2` |
| model | `deepseek-v4-flash` — identical to the 14-turn control |
| episodes per task | 2 |
| **new trajectories** | **10** (within the ≤10 allowance for G3) |
| treatment budget | **24 turns**, commands 24→40, wall 900→1500 s |
| control | **frozen historical 14-turn results**, `artifacts/tier_h_primary.jsonl`. Not re-run. |
| temperature | 1.0 |
| max_tokens | 16,000 per call |
| system prompt | byte-identical (SHA recorded in the run manifest) |
| harness | unchanged; only `Budget` differs |
| verifier | hardened v1 **as frozen for Tier H**, plus the independent oracle |
| grading | Docker sandbox, `--network=none` |
| work dir | `.sandbox_work_h24` (Tier H evidence preserved) |
| output | `artifacts/tier_h_24turn.jsonl` |

The **only** intended difference from the control is the turn budget. Command and
wall budgets are scaled with it because holding them fixed would make turns
unusable and confound the manipulation.

## Subset coverage requirement (satisfied)

All five H tasks are included, which necessarily covers the three required cells:

| cell | task | 14-turn FULL_FIX |
|---|---|---|
| easy / ceiling H task | `h4_schedule_triple_defect` | 4/4 |
| in-band H task | `h1_attention_double_defect`, `h3_accumulation_and_clipping` | 2/4 each |
| zero-scoring H task | `h2_position_double_defect`, `h5_masking_triple_defect` | 0/4 each |

## Metrics (all pre-declared)

Per trajectory: FULL_FIX / SEMANTIC_FIX / PARTIAL_FIX / WRONG / INVALID label
from the independent oracle; turns consumed; whether it submitted or exhausted
budget; first turn a relevant file was read; first patch turn; files read; unique
files; tool actions; prompt/completion tokens; cost.

Per task: FULL_FIX at 14 vs 24; change in submission rate; change in first-patch
turn.

## Analysis rules

- **No p-values, no significance claims.** n = 2 per task at 24 turns. Counts and
  raw trajectories only.
- Verdict token, chosen by a pre-declared rule:
  - **`BUDGET-SENSITIVE`** if any task that scored 0/4 at 14 turns scores ≥1/2 at
    24 turns, **or** the overall FULL_FIX rate rises by ≥2 trajectories out of 10.
  - **`BUDGET-ROBUST WITHIN TESTED RANGE`** otherwise.
- If `BUDGET-SENSITIVE`, every affected task's Phase 1 `TOO_HARD` label is
  rewritten as budget-sensitive in `DIFFICULTY_DISTRIBUTION.md` and in the final
  report.
- The 14-turn control results are **not** re-run and **not** modified.

## What would invalidate this experiment

- Any change to a task, prompt, harness or verifier between control and treatment
  (the run manifest is hashed to detect this).
- Mixing 24-turn trajectories into the Tier H headline numbers. They are reported
  as a separate arm.
- Selecting a subset after seeing partial results.

## Pre-registration hash

Recorded in `FINAL_WORKLOG.md` at the moment of freezing, before the first API
call of this arm.
