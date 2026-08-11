# TURN_BUDGET_ABLATION_REPORT

# Verdict: `BUDGET-SENSITIVE`

Pre-registered in `TURN_BUDGET_ABLATION_PROTOCOL.md`, SHA256
`620FDAAE5D2B8CB527E2BFD008A014BDF76E2B6A194C30C3A8184220DFA01973`, hashed **before
the first API call of this arm**. The verdict rule was fixed in advance and is
applied mechanically below.

## The confound

Phase 1 recorded **39 of 40** `deepseek-v4-flash` episodes ending on budget
exhaustion rather than by the agent's own judgement; Tier H was 20/20. Two Tier H
tasks scored 0/4 and were labelled `TOO_HARD`. That label was not defensible,
because an agent that never voluntarily stops may be horizon-limited rather than
capability-limited.

## Design

Treatment: all 5 Tier H tasks × 2 episodes × **24 turns** (commands 24→40, wall
900→1500 s, scaled with turns and recorded as such). Control: the **frozen**
14-turn results, not re-run. Model, prompt, harness, verifier and temperature
identical. 10 new trajectories, within the ≤10 allowance.

## Results

Labels from the independent oracle (`TIER_H_24TURN_REAL_MODEL_AUDIT.csv`), **zero
oracle/verifier disagreements**.

| task | 14-turn FULL_FIX | 24-turn FULL_FIX | change |
|---|---|---|---|
| `h1_attention_double_defect` | 2/4 (50 %) | **2/2 (100 %)** | ↑ |
| `h2_position_double_defect` | 0/4 (0 %) | **0/2 (0 %)** | — |
| `h3_accumulation_and_clipping` | 2/4 (50 %) | **2/2 (100 %)** | ↑ |
| `h4_schedule_triple_defect` | 4/4 (100 %) | **2/2 (100 %)** | — |
| **`h5_masking_triple_defect`** | **0/4 (0 %)** | **2/2 (100 %)** | **↑↑** |
| **overall** | **8/20 = 40 %** | **8/10 = 80 %** | **+40 pp** |

### Verdict rule, applied

> `BUDGET-SENSITIVE` if any task that scored 0/4 at 14 turns scores ≥1/2 at 24
> turns, **or** the overall FULL_FIX rate rises by ≥2 trajectories out of 10.

**Both conditions fire.** `h5_masking_triple_defect` went **0/4 → 2/2**, and the
overall rate rose from 40 % to 80 %. Verdict: **`BUDGET-SENSITIVE`**.

## The finding that matters most

**The agent stops on its own once it has room to.**

| | 14 turns | 24 turns |
|---|---|---|
| submitted voluntarily | **0/20 (0 %)** | **8/10 (80 %)** |
| ended on budget exhaustion | **20/20 (100 %)** | **0/10 (0 %)** |
| mean turns used | 14.0 (capped) | 19–24, mean ≈ 21 |

At 14 turns the budget was the binding constraint on every single Tier H episode.
At 24 turns it binds on none. The episodes that succeeded submitted at turns
19–24 — i.e. they needed **more than 14 turns and fewer than 25**, a window Phase 1
could not see into.

## Phase 1 conclusions that are now wrong

**`h5_masking_triple_defect` was not too hard.** Phase 1 labelled it `TOO_HARD`
from 0/4. At 24 turns it is 2/2, and the successful episodes edited **all three**
required files — `data.py`, `model.py` *and* `train.py`. The 14-turn episodes had
edited only `data.py`. The task needs three coordinated edits and the agent ran
out of turns after the first, which reads as failure and was not.

This is a direct correction to a published difficulty label, produced by the
experiment designed to look for exactly this.

## `h2` remains 0/2 — but not for a capability reason

Both 24-turn `h2` episodes ended with `too many unusable actions` (three
consecutive unparseable actions) after 17 and 19 turns, having edited **nothing**.
That is a **PROTOCOL_FAILURE**, not a capability failure, and it is a different
failure mode from the 14-turn run.

So `h2`'s 0/4 is *also* not a clean capability signal, for a second and independent
reason. Its true difficulty is still unmeasured. Root-causing the unparseable
actions is required before `h2` can carry any difficulty claim.

## Corrected difficulty interpretation

| task | Phase 1 label | corrected label |
|---|---|---|
| `h1` | PROMISING | PROMISING (2/4 @14, 2/2 @24) |
| `h2` | TOO_HARD | **UNMEASURED — protocol failure at both budgets** |
| `h3` | PROMISING | PROMISING (2/4 @14, 2/2 @24) |
| `h4` | TOO_EASY | TOO_EASY |
| `h5` | TOO_HARD | **BUDGET-SENSITIVE — solvable at 24 turns** |

Every difficulty number in this project carries the qualifier **"at a 14-turn
budget"**, and for Tier H that qualifier is now known to be load-bearing rather
than pedantic.

## Cost

10 trajectories. Prompt cost grows with the square of the horizon, so 24-turn
episodes are materially more expensive per trajectory than 14-turn ones; exact
tokens are in `artifacts/tier_h_24turn.jsonl`. Total new spend for this arm is well
inside the $5 stop.

## Limitations

- **n = 2 per task at 24 turns.** No p-values, no significance claims. `h5`'s
  0/4 → 2/2 is 2 successes out of 2, not a rate.
- One model only (`deepseek-v4-flash`).
- Only two budgets tested. Whether 18 or 32 turns would move things further is
  unknown; the finding is bracketed as *within the tested range*.
- The 14-turn control was collected earlier under the same frozen manifest but not
  re-run, so any drift in the provider's model between runs is unmeasured.
- Command and wall budgets scaled with turns. That is required for the extra turns
  to be usable, but it means the manipulation is "horizon", not "turn count in
  isolation".
