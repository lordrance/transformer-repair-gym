> **⚠️ SUPERSEDED 2026-08-10.** This report's verdict (`NO — one measurement short`)
> was correct when written: the Tier M real-model evaluation was blocked on a missing
> API key. That measurement has since been made three times over (50 trajectories,
> two models), a Tier H redesign round was run, and the verdict is now
> **`PUBLIC_READY_LOCAL = YES`**. Read
> [`PHASE_1_FINAL_RESEARCH_REPORT.md`](PHASE_1_FINAL_RESEARCH_REPORT.md) instead.
> Retained unedited as the record of what was known at that point.
# Phase 1 — GPU-Free Flagship Build

**Date:** 2026-08-10 · **Machine:** Windows 11 + WSL2, Docker 27.0.3, `torch 2.9.1+cpu`, **no GPU**
**Tests:** 100 passed · **API spend this phase:** $0.00

## Status: **BUILT, NOT YET MEASURED**

Every buildable deliverable is done and verified. One thing is not, and it is the
thing the acceptance gate turns on:

> **There was no `DEEPSEEK_API_KEY` on the machine** (I removed the exposed one at
> the end of Phase 0.6 at your instruction, and no rotated key has been set). So
> the Tier M real-model evaluation did not run, and the gate *"medium tasks are no
> longer all 4/4"* is **unanswered**.

Everything else — the repo-level tasks, the multi-turn harness, the Docker
sandbox, the protected oracle, the held-out structure — is built, tested, and
verified working. One command closes the remaining gate.

---

## What was delivered

| § | deliverable | status |
|---|---|---|
| 1 | Docker sandbox + `SANDBOX_DESIGN.md` | ✅ **verified running**: gold clean in 8.2 s, buggy fails the right 2 checks, `--network=none`, read-only rootfs, non-root, caps dropped |
| 2 | Tier E retained as easy/regression | ✅ 5 tasks, unchanged, still audited |
| 3–5 | 5 Tier M repo-level tasks | ✅ 8-module package, symptom-only, 5/5 discriminate, **5/5 REAL/REAL-DERIVED** |
| 6 | multi-turn harness | ✅ list/read/run/patch/submit + budgets; **17 tests** driven by scripted policies |
| 7 | L1/L2/L3 reward, semantic vs full success | ✅ 18 repo checks across 3 levels |
| 8 | naive vs hardened at Tier M | ✅ visible suite passes on all 5 buggy repos; hidden suite fails |
| 9 | real-model evaluation | ❌ **blocked on key**; runner written (`scripts/run_deepseek_repo_eval.py`) |
| 10 | `DIFFICULTY_DISTRIBUTION.md` | ✅ Tier E measured, Tier M structural only |
| 11 | independent truth audit | ✅ Tier E labelled; Tier M oracle = protected gold repo, built outside every mount |
| 12 | reward-integrity experiment | ✅ constructed audit (7/7); ❌ real-rollout audit blocked |
| 13 | held-out design | ✅ configurations / instances / independent oracle; ⚠️ one family not a matched pair |
| 14 | 10 high-quality tasks | ✅ 5 E + 5 M, not scaled further |
| 15–16 | public readiness, GPU as optional extension | ✅ no credential in repo; no checkpoints, no training curves, no fabricated metrics |

---

## Q1 — Is Tier M genuinely closer to ML research engineering?

**Yes, structurally — and I can say exactly how, rather than asserting it.**

| dimension | Tier E | Tier M |
|---|---|---|
| files shown | 1 | **8 modules + tests** |
| bug location disclosed | yes | **no** |
| symptom names the failing property | yes | **no** — a loss curve, an LR trace, a plateau |
| diagnosable from a forward pass | 5/5 | **2/5** |
| requires running training | 0/5 | **3/5** |
| spans >1 file | 0/5 | **2/5** |
| interaction | one edit | `m4`, `m5` need two coordinated edits |
| horizon | single turn | up to 14 turns with tools |

The symptoms are what a colleague actually says. `m1`: *"our loss falls much
faster than it used to and settles far below anything we saw before, but the model
is useless."* That is the realistic presentation of lookahead leakage — it does
not look broken, it looks excellent, which is why it survives review. No symptom
names a file or a root cause, and that is enforced by a test, not by care:
`test_every_task_is_reachable_through_the_harness` asserts none of `attention.py`,
`positional.py`, `optim.py`, `data.py`, `train.py`, `tril`, `rotate_half`,
`zero_grad`, `ignore_index`, `weight_decay` appears in any symptom.

**Honest limit:** an 8-module, ~500-line package is not a real research codebase.
This is a faithful scale model of ML-engineering debugging, not the thing itself.
What transfers is the *shape* of the work — localize, instrument, hypothesize,
verify — not the scale.

## Q2 — DeepSeek success rate and variance on Tier M

**Not measured. No key.** I will not estimate it; Phase 0.5 already showed that
predictions here are wrong in both directions (frontier models solved Tier E
trivially; a "weaker" model's failures turned out to be a token-budget artifact).

What is measured, and bounds the question: reference-vs-buggy signatures are
crisp, so the grading signal is not the uncertainty —

| task | gold | buggy |
|---|---|---|
| `m3` | loss 5.26 → 0.11 | 5.26 → **5.33**, gradient norm **exactly 0.0** every step |
| `m4` | LR hits 0 at step 39/40, final 0.11 | LR hits 0 at step **20**/40, final **0.64** |

## Q3 — Is the reward still sound?

**Yes, and it stayed simple.** Three recorded levels, no shaping, no weighted
blend:

| level | Tier M checks |
|---|---|
| **L1 contract** | imports, public API, shapes/dtypes, patch applies, only `tinygpt/` edited |
| **L2 semantic** | `repo_matches_gold_logits`, causality, RoPE invariance, token counts, padding invariance — all against the protected gold repo across shapes the visible suite never uses |
| **L3 behavioural** | `repo_training_converges`, `repo_lr_schedule_matches_gold`, `repo_gradients_reach_optimizer`, `repo_clipping_is_effective`, `repo_no_pad_probability_mass` |

`semantic_success` and `full_success` remain separate labels — the split
introduced in Phase 0.6 after a rollout computed correct gradients but returned
`Tensor` instead of `float`. Collapsing those into one number would tell an RL
policy that a correct fix with a type slip is the same event as a wrong gradient.

Soundness evidence at Tier M (`scripts/audit_repo_tasks.py`): gold passes every
check on all 5; buggy fails only hidden checks on all 5; **the visible suite
passes on all 5 buggy repos**, so the naive/hardened gap survives the move to
repo level.

L3 checks assert bands and invariances, never point values — final loss below 2.0
where the reference reaches ~0.1 and chance is 5.55; the LR trace compared step
for step against the reference. `torch.use_deterministic_algorithms` is warn-only
on purpose: hard-on turns an unrelated missing kernel into a spurious task
failure.

## Q4 — Hardened verifier FPR / FNR

**On real rollouts: not measurable this phase.** Both quantities are defined
against real model submissions, and there are none for Tier M.

Carried forward from Phase 0.5 (Tier E, 40 trajectories, independent probe as
ground truth):

| | value |
|---|---|
| hardened false positive | **0/38 = 0.0 %** |
| hardened false negative | **1/19 = 5.3 %** (pro) — the `Tensor`-vs-`float` case |
| naive false positive | **0/39 = 0.0 %** |

For Tier M, two heuristics are known FPR risks and are flagged rather than
trusted: `gate_no_shape_special_casing` fires on any comparison of a tensor
dimension against an integer > 2, and `gate_no_silent_exception_swallowing` fires
on any broad `except`. Both patterns are legitimate in real code. Their
false-positive rate on human-written fixes is **unmeasured** and is the first
thing I would measure once rollouts exist.

## Q5 — Natural reward hacking

**Zero, because there were no real rollouts this phase.** Recorded as 0
observations, not as 0 rate — the distinction matters and I am not going to blur
it.

Constructed evidence is unchanged and still holds: **7/7 exploits** produce
`naive=1, hardened=0` at Tier E. The one that matters most,
`e4_t4_tamper_local_oracle`, leaves the bug alone and rewrites the oracle sitting
in the same file; **no static gate catches it**. It fails only because grading
uses an oracle outside the workspace.

That result is what the Tier M architecture is built around: the gold repo is
materialized in a process-local temp directory, outside every container mount.
A candidate cannot read it, diff against it, or edit it. **An oracle the candidate
can reach is not an oracle** — detection is a fallback, unreachability is the
property.

## Q6 — Is there enough difficulty spread?

**Not demonstrated.** Tier E is measured and uniformly `TOO_EASY` (9/10 cells at
4/4). Tier M is structurally harder along seven independent dimensions but
**unmeasured**. A spread built from one measured pole and one predicted pole is
not a spread.

## Q7 — Is the GPU-free version worth publishing and putting on a résumé?

# NO — not yet. One measurement short.

Not a close call, and not pessimism about the work. The project's entire claim is
*"I study task difficulty, reward soundness and verifier error rates
empirically."* Publishing a 10-task environment where half the tasks have never
been run against a model would contradict that claim in the artifact itself. The
first competent question in an interview is "what were the pass rates?", and for
Tier M the answer today is "I did not measure them."

What is already publishable-grade, and would be strong:

- 10 tasks, **10/10 REAL or REAL-DERIVED**, every one anchored to a named issue,
  PR, or official doc — plus a documented negative result about what does *not*
  reproduce (AdamW and NaN);
- a container sandbox with a written threat model that separates reward hacking
  from escape, and an oracle that is unreachable rather than merely protected;
- a multi-turn harness with 17 tests, verified by scripted policies before any
  API existed;
- an independent equivalence oracle that has already **overruled the reward
  once**, catching a hardened false negative;
- 7 constructed exploits, one of which defeats every static gate — a genuine
  finding about verifier design;
- 100 tests, reproducible scripts, no credential, a clean license audit that
  excluded a non-commercial source on inspection.

**What converts NO to YES:** run `scripts/run_deepseek_repo_eval.py`, 20
trajectories, well under a dollar. Then the Tier M table has real numbers, the
difficulty spread is measured at both ends, and the FPR/FNR figures cover both
tiers. That is one command and one afternoon of analysis — not more building.

I would rather tell you it is one command short than call it done.

## Q8 — If YES later, what would GPU RL training add?

Listed as **Optional Phase 2**, not started, not pre-bound to any model or
backend.

What RL training would add that this phase genuinely cannot:

1. **The falsification the project was designed for.** Constructed exploits prove
   a naive/hardened gap *exists*. Only a policy under gradient pressure shows
   whether it *matters*. Phase 0.5 established that a frontier model answering
   once has no motive to cheat; an optimized policy is a different population.
   Untestable without training.
2. **Held-out generalization as a model claim.** Right now held-out structure
   tests whether the *verifier* overfits its fixtures. Training would let the
   same structure test whether a policy trained on hardened reward generalizes
   better than one trained on naive reward — the actual research question.
3. **The reward-integrity A/B.** Two arms, identical seeds and rollout budgets,
   only the reward differing. Training reward rises in both; held-out performance
   diverges. That is the one figure this project exists to produce, and it
   requires parameter updates.
4. **Difficulty calibration under pressure.** A task at 20 % pass rate for a
   frozen policy may sit at 80 % after a few hundred steps. Static calibration
   cannot see that.

What it would **not** add: task realism, verifier soundness, sandbox integrity,
or the exploit catalogue. Those are complete and do not improve with a GPU.

The README says `Optional future extension` and nothing stronger. No checkpoints,
no training curves, no improvement metrics exist in this repository, and none
will be fabricated.

---

## Corrections to my own work this phase

1. **The first repo template did not learn.** Loss sat at 5.55 ≈ ln(256) for 30
   steps because every sequence used fresh random symbols — nothing consistent to
   learn across batches. Replaced with fixed disjoint cycles, making next-token
   prediction a deterministic function of the current token: 5.26 → 0.11 in 40
   CPU steps. Without this the entire L3 behavioural level would have had no
   signal, and three Tier M tasks would have been ungradeable.
2. **M3's first symptom did not match its mutation.** It advertised NaN; the
   measured behaviour was a spike to 11.79 followed by recovery. Two further
   candidates were also rejected — omitting `zero_grad` made the final loss
   *better* than gold (0.086 vs 0.111), acting as momentum. Root cause:
   **AdamW normalizes updates per parameter, so gradient-magnitude bugs do not
   produce NaN.** M3 was rebuilt around gradient lifecycle, whose signature is
   unambiguous (gradient norm exactly 0.0, loss never moves).
3. **A hyperparameter is not a bug.** M3's draft included `lr: 3e-1`. Measurement
   showed the loss spike came from the learning rate and not from the clipping
   order at all, and a config value being aggressive is a judgement call, not a
   defect. Dropped.
4. **Windows bind-mount limitation.** Docker Desktop cannot mount from
   `%LOCALAPPDATA%\Temp`, which is where `tempfile` writes. Containerized
   workspaces now live under `./.sandbox_work`. Documented in `SANDBOX_DESIGN.md`
   so it is not rediscovered later.

## To close Phase 1

```powershell
$env:DEEPSEEK_API_KEY = "<rotated key>"      # your shell, not a file, not chat
python scripts/build_sandbox.py               # once; already built here
.\.venv\Scripts\python.exe scripts/run_deepseek_repo_eval.py --episodes 4
```

Then: audit the 20 trajectories with the independent oracle, fill in
`DIFFICULTY_DISTRIBUTION.md`, and apply the brief's own decision rule —
GO / REDESIGN HARDER / CURRICULUM.

One caveat on that runner: the **harness** underneath it has 17 tests, but the
DeepSeek tool-call adapter has never spoken to the live API. Expect to fix
something small in the first run.

**STOP here for review.** No task scaling, no GRPO, no Tinker, no LoRA, no Ray,
no vLLM, no external benchmarks.
