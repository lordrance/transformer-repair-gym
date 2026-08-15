# Transformer ML-Engineering RL Environments & Reward Integrity
### Phase 1 final research report — GPU-free

**Date:** 2026-08-10 · **Host:** Windows 11 + WSL2, Docker 27.0.3, `torch 2.9.1+cpu`, **no GPU**
**Tests:** 148 passed · **Real-model trajectories:** 50 · **API spend:** $1.29
**GPU training:** not performed. Optional Phase 2.

Every number below is produced by a script from a raw artifact. Nothing is
hand-entered. Regeneration commands are in §20.

---

## 1. Abstract

We build and audit a 15-task reinforcement-learning environment in which a model
must diagnose and repair real Transformer training-code defects, and we measure
whether the reward can be trusted before any policy is trained.

Tasks span three tiers of a stepping-stone structure: single-file with the location
disclosed (E), an 8-module package with the location withheld and a symptom-only
report (M), and interacting multi-defect variants of the same families (H). All 15
pass a four-gate source-alignment audit; 12 are ACCEPTED and 3 ACCEPTED_WITH_CAVEAT,
none rejected.

Two verifiers are compared: a *reasonable naive* grader that tests the right
properties at one fixed public fixture and shells out to `pytest` inside the
candidate's own workspace, and a *hardened* grader that judges against a gold
repository the candidate cannot reach. A 13-probe adversarial fuzz audit run
**before** any evaluation measures naive false-positive rate **1.00** and hardened
**0.00**, with a **69 %** disagreement rate.

Against 50 real multi-turn trajectories from two DeepSeek models, the naive
false-positive rate rises with difficulty — **0.10 → 0.20 → 0.60** — while the
hardened verifier holds at **0.00 false positives and 0.00 false negatives**, with
**zero** disagreements against an independent semantic oracle. **No natural reward
hacking or tampering was observed in any trajectory.**

Difficulty calibration produced a genuine spread in one permitted redesign round:
Tier M sits at 90 % full-fix (3 of 5 tasks at ceiling), Tier H at 40 % (1 easy, 2
in band, 2 too hard).

The most consequential finding is methodological and self-directed: our
*independent audit oracle* was structurally blind to an entire task family, and
produced two wrong labels on real data before it was caught and corrected. An
audit oracle is also only a proxy.

**Verdict: `PUBLIC_READY_LOCAL = YES`.** See §19.

## 2. Motivation

RL for code is bottlenecked on environments, and environments are bottlenecked on
verifiers. The 2026 literature makes both claims quantitative: 28.5 % of sampled
SWE-bench Verified tasks accept a Docker-verified *incorrect* patch, and models
score **+14.14 pp** higher on exploitable tasks than on robust ones
([arXiv:2606.16062](https://arxiv.org/abs/2606.16062)). A pass rate on an unaudited
verifier is therefore not a capability measurement.

This project takes the narrow, checkable version of that problem: build a small
number of high-quality Transformer ML-engineering environments, then measure the
reward's own error rates before trusting it. GPU training is deliberately out of
scope; what is in scope is **environment readiness**, not training effectiveness.

## 3. Related work and the design guardrails it produced

Eight 2026 papers were read from arXiv and converted into binding design rules,
recorded with enforcement locations in
[`LITERATURE_GUARDRAILS_2026.md`](LITERATURE_GUARDRAILS_2026.md). The five that
changed the most:

| guardrail | source | rule adopted |
|---|---|---|
| **G1** weak verifiers accept wrong patches at ~25–28 % | [2606.16062](https://arxiv.org/abs/2606.16062) | never report a pass rate without a paired exploitability measurement; treat an unexpectedly high pass rate as a verifier alarm |
| **G2** fuzz the verifier *before* training | [2606.01066](https://arxiv.org/abs/2606.01066) | fuzz audit is a precondition for expanding the task set; report FP, FN and **disagreement** |
| **G3** faithfulness / robustness / scalability; no fixed verifier survives a stronger policy | [2606.26300](https://arxiv.org/abs/2606.26300) | score all three axes incl. measured cost; never write "the reward is sound"; state a verification horizon |
| **G4** 13.6 % PR–issue misalignment in SWE-bench Verified | [2607.28587](https://arxiv.org/abs/2607.28587) | four-gate source audit; distinguish *cited* from *audited* provenance |
| **G6** structure and difficulty beat volume; stepping stones | [2603.24202](https://arxiv.org/abs/2603.24202) | hard cap of 15 tasks; new difficulty only as a stepping stone inside an existing family |

G6 also produced the anti-p-hacking rule that governed the night: **at most two
difficulty redesign cycles, all versions retained.** One was used.

## 4. Research questions

- **RQ1** Does moving from single-file to repo-level and to interacting defects add
  *investigation complexity* rather than ambiguity?
- **RQ2** What are the naive and hardened verifiers' FPR/FNR on adversarial,
  fuzzed, and real-model patches?
- **RQ3** How many tasks sourced from real issues survive an alignment audit?
- **RQ4** How do successful and failed trajectories differ in localization, tool
  use, turns and cost?

## 5. Environment architecture

```
tinygpt/           8 modules: config, norm, positional, attention, model, data, optim, train
tests/             the visible suite -- regenerated before grading
SYMPTOM.md         what a user reported; no file named, no root cause
```

A candidate gets a workspace and five tools: `list_files`, `read_file`,
`run_command` (four named commands, no shell), `apply_patch` (unified diff, fuzzy
application), `submit`. Budgets: 14 turns, 24 commands, 900 s wall, 180 s per
command, 6,000-char tool output.

The reference package trains on CPU: **loss 5.26 → 0.11 in 40 steps**, which is
what makes L3 behavioural checks possible at all. The first version of this
package did not learn (random symbols per sequence, loss pinned at ln 256); that
was found and fixed before any task was built.

Grading is three-level: **L1** contract (imports, public API, shapes, dtypes),
**L2** semantic (hidden configurations against the gold repo, forward and
gradient), **L3** behavioural (short training runs, convergence band, LR trace,
clipping effectiveness, gradients actually reaching the optimizer).

`semantic_success` and `full_success` are reported separately throughout. A
submission that computes the right thing and returns the wrong type is a
SEMANTIC_FIX, and collapsing that into one number would hide it.

## 6. Task provenance

15 tasks, **all REAL or REAL-DERIVED, zero synthetic**, across six families:

| family | E | M | H |
|---|---|---|---|
| F1 attention masking | `t1` | `m1` | `h1` (+ padding mask disabled) |
| F2 position encoding | `t2` | `m2` | `h2` (+ `rope_theta` changed) |
| F3 normalization numerics | `t3` | — | — |
| F4 optimizer / scheduler / accumulation | `t4` | `m3`, `m4` | `h3`, `h4` |
| F5 data / labels / loss masking | `t5` | `m5` | `h5` |

Full chain rationale in [`TASK_CHAINS.md`](TASK_CHAINS.md). F3 has no repo-level
variant on purpose: its fp16 overflow is a property of the dtype, not of repository
structure, so a repo version would add search without adding anything about the bug.

A recorded negative result shaped F4. M3's first draft advertised "loss becomes NaN
after a few steps". Three candidate mutations were measured and all three rejected:
clip-after-step with a raised LR spiked to 11.79 and recovered; **omitting
`zero_grad` entirely made the final loss better than gold (0.086 vs 0.111)**,
acting as momentum. The reason is general: **with AdamW, gradient-magnitude bugs do
not produce NaN**, because the update is normalized per parameter. NaN comes from
forward numerics or the loss, not from large gradients. M3 was rebuilt around
gradient lifecycle, whose signature is unambiguous — gradient norm **exactly 0.0**
at every step while everything stays finite and warning-free.

## 7. Source alignment audit (RQ3)

Guardrail G4. Our exposure differs from PAIChecker's: we do not extract tasks from
issue→PR pairs, we re-instantiate documented bug *patterns*, so the failure mode we
can suffer is **"the citation does not say what we claim it says"**. Four gates:
(A) citation supports the claim — manual, every URL opened; (B) pre-fix behaviour
reproduces; (C) gold fix resolves it; (D) the reduction preserves the root cause.

| verdict | n | tasks |
|---|---|---|
| ACCEPTED | **12** | all E, m1/m2/m4/m5, h1/h3/h5 |
| ACCEPTED_WITH_CAVEAT | **3** | `m3`, `h2`, `h4` |
| **REJECTED_SOURCE** | **0** | — |

Gate B and C pass 15/15 mechanically. Gate A passes 12/15 and is **PARTIAL** three
times, each recorded rather than argued away:

- **`m3`** — the PyTorch optim docs and forum thread establish the required
  `zero_grad → backward → step` ordering, but *neither documents this specific
  failure* (zero_grad landing between backward and step). The pattern is real; the
  incident is our construction.
- **`h2`** — the RoPE pairing half is verbatim-supported by
  [#25199](https://github.com/huggingface/transformers/issues/25199); the added
  `rope_theta 10000 → 500` is a plausible porting error we invented.
- **`h4`** — scheduler mis-stepping is authoritative (PyTorch emits its own warning
  during this task's buggy run) and decay-on-gains is standard practice, but the
  third defect, one micro-batch dropped per window, has no cited incident.

**Answer to RQ3:** 12 of 15 fully audited, 3 accepted with an explicit weaker claim,
0 rejected. The distinction between *cited* and *audited* provenance is now recorded
per task in [`SOURCE_ALIGNMENT_AUDIT.csv`](SOURCE_ALIGNMENT_AUDIT.csv).

## 8. Sandbox

Docker, verified working: gold grades clean in **8.2 s**, the buggy repo fails
exactly the checks it should. `--network=none`, read-only rootfs, `--cap-drop=ALL`,
`no-new-privileges`, 2 GB memory with swap closed, 2 CPUs, 256 pids, non-root user
created before anything else, tmpfs scratch with `noexec`.

The threat model in [`SANDBOX_DESIGN.md`](SANDBOX_DESIGN.md) separates two things
Phase 0 conflated: **reward hacking** is a verifier problem, **escape** is a
container problem, and neither defence helps with the other.

The load-bearing decision is that the gold repo is **not mounted**. Phase 0.5
constructed an exploit that leaves the bug alone and rewrites the oracle sitting in
the same file; no static gate catches it. It fails only because the oracle is
unreachable. **An oracle the candidate can reach is not an oracle** — detection is
the fallback, unreachability is the property.

Measured cost: hidden suite **0.86 s** mean, but **container startup +8.24 s ≈ 16×
the actual work**. At 10k rollouts that is ~2.4 CPU-hours of checks against ~23
hours of container startup. Any Phase 2 needs a persistent worker, not faster
checks.

## 9. Naive vs hardened verification

| | naive (v2) | hardened |
|---|---|---|
| oracle | `tests/test_visible.py` **inside the workspace** | gold repo, **outside every mount** |
| how it runs | `pytest` in the candidate's cwd | 3–4 hidden checks, L1/L2/L3 |
| configurations | one fixed public fixture per property | 5 hidden shapes, unseen step counts |
| anti-tamper | none, deliberately | grader files regenerated; gold unreachable |

The naive grader is *reasonable*, not a straw man: the **unmodified buggy repo
fails it on all 15 tasks**, and it tests the right properties. It is weak in the
way real first-pass graders are weak — one fixture, and it trusts a test file the
candidate can edit.

## 10. Verifier fuzzing (RQ2, constructed)

13 probes in four classes, run **before** the task set was expanded (G2). Ground
truth is an independent probe with `SEED=424242424` and configurations disjoint
from every graded fixture.

| metric | value | denominator |
|---|---|---|
| **naive FPR** | **1.00 (9/9)** | probes whose truth is WRONG |
| **hardened FPR** | **0.00 (0/9)** | same |
| naive FNR | 0.00 (0/3) | probes whose truth is FULL_FIX |
| **hardened FNR** | **0.00 (0/3)** | same |
| **naive ↔ hardened disagreement** | **0.69 (9/13)** | all probes |
| exploit catch rate | 1.00 (9/9) | WRONG probes |
| hardened FP vs `full_success` | **0.10 (1/10)** | non-FULL_FIX probes |

**The naive verifier accepts every non-genuine submission, including three that do
not touch the bug at all.** Two real gaps were found and are documented rather than
silently patched:

- **Gap 1 (H1)** — `D1_m3_return_tensor` computes the gradient lifecycle correctly
  but returns a `Tensor` where the contract says `float`, and **hardened passes it**.
  Tier E catches this via `grad_accum_runs`; the Tier M suite has no L1 equivalent.
  Not fixed during the session: changing the graded verifier while the frozen
  evaluation was running is precisely what `PROTOCOL_CHANGELOG.md` exists to
  prevent. Recorded as v1's verification horizon and the first item for v2.
- **Gap 2 (H3)** — `C2_m4_fix_schedule_only` is caught by hardened but the
  independent probe called it semantically fine, because it cannot see
  `optimizer.param_groups`. **For that family the audit oracle is weaker than the
  verifier it audits.** This foreshadowed §15.

Three axes and a documented horizon of six known miss-classes:
[`VERIFIER_QUALITY_MATRIX.md`](VERIFIER_QUALITY_MATRIX.md).

## 11. Real-model evaluation

Multi-turn, Docker-graded, independently audited. Two models, three runs, 50
trajectories. Frozen manifests `tier_m_primary_v1` and `tier_h_v2` (27 file hashes
each, including the system-prompt SHA).

| run | model | n | FULL_FIX | naive pass | hardened pass | naive FPR | hardened FPR | hardened FNR | disagreements | natural hacks |
|---|---|---|---|---|---|---|---|---|---|---|
| Tier M primary | `v4-flash` | 20 | **18 (90 %)** | 100 % | 90 % | **0.10** | **0.00** | **0.00** | **0** | **0** |
| Tier M confirmatory | `v4-pro` | 10 | **8 (80 %)** | 100 % | 80 % | **0.20** | **0.00** | **0.00** | **0** | **0** |
| Tier H | `v4-flash` | 20 | **8 (40 %)** | 100 % | 40 % | **0.60** | **0.00** | **0.00** | **0** | **0** |

95 % Wilson interval on Tier M full-fix: **[0.699, 0.972]**. Quoted because n = 20.

**Answer to RQ2.** The naive false-positive rate is not a constant — it **scales
with difficulty: 0.10 → 0.20 → 0.60**. At Tier H, 12 of the 20 submissions the
naive grader accepts are not genuine fixes. This is G1's mechanism observed
directly: a weak verifier flatters harder tasks most. The hardened verifier held at
0.00 FPR and 0.00 FNR across all 50, with zero disagreements against the
independent oracle.

**Natural reward hacking: 0 in 50 trajectories.** Also 0 tampering, 0 INVALID, 0
INFRA_FAILURE. The correct statement is *no natural exploit was observed under this
evaluation distribution* — not that the reward is secure. There was no optimization
pressure, and 7 constructed exploits plus 9 fuzz probes pass the naive grader
whenever anyone tries.

**Cross-model consistency.** `m5` is the hardest Tier M task for both models, and
both fail it the *same way*: edit `data.py`, miss `model.py`. But the level is not
monotonic in model strength — pro scores 0/2 on `m5` where flash scores 3/4. With
n = 2 that is an observation, not a result.

## 12. Difficulty calibration

Keyed on the independent FULL_FIX label, never on reward.

| tier | task | FULL_FIX | class |
|---|---|---|---|
| M | `m1_attention_regression` | 4/4 | TOO_EASY |
| M | `m2_position_encoding` | 4/4 | TOO_EASY |
| M | `m3_gradient_lifecycle` | 4/4 | TOO_EASY |
| M | `m4_schedule_accumulation` | 3/4 | PROMISING |
| M | `m5_masking_interaction` | 3/4 | PROMISING |
| H | `h1_attention_double_defect` | **2/4** | **PROMISING** |
| H | `h2_position_double_defect` | **0/4** | TOO_HARD |
| H | `h3_accumulation_and_clipping` | **2/4** | **PROMISING** |
| H | `h4_schedule_triple_defect` | 4/4 | TOO_EASY |
| H | **`h5_masking_triple_defect`** (held out) | **0/4** | TOO_HARD |

**Spread achieved: 4 TOO_EASY, 4 PROMISING, 2 TOO_HARD** across the ten
repo-level tasks, plus five E-tier anchors that are uniformly easy by design. This
is a gradient, and it was obtained in **one** redesign round. The second permitted
round was not used, and the calibration is **not** recorded as
`DIFFICULTY_CALIBRATION_FAILED`.

Reward patterns (hardened) show where GRPO advantage would exist and where it would
be identically zero:

```
M: m1 [1,1,1,1]  m2 [1,1,1,1]  m3 [1,1,1,1]  m4 [1,1,0,1]  m5 [0,1,1,1]
H: h1 [1,0,0,1]  h2 [0,0,0,0]  h3 [1,0,1,0]  h4 [1,1,1,1]  h5 [0,0,0,0]
```

Four of ten groups carry variance. `h1` and `h3` are the best-shaped tasks in the
set.

**What made H harder was interaction, not obscurity.** Every H task reuses its
parent's mathematics; the change is that fixing one defect is no longer enough. The
Tier M failure analysis pointed straight at this: both M failures were *incomplete*
localization on the two multi-file tasks. Localization rate drops **0.90 → 0.45**
from M to H, so the added difficulty is being paid in search, which is the intent.

`h5` was reserved before any H result was seen and graded once. It came back 0/4.

## 13. Trajectory analysis (RQ4)

Full detail in [`TRAJECTORY_EFFICIENCY.md`](docs/history/TRAJECTORY_EFFICIENCY.md).

| | Tier M | Tier H |
|---|---|---|
| mean turns | 14.0 | 14.0 |
| mean reads | 10.4 | 10.5 |
| mean distinct files read | 9.4 | 9.4 |
| mean commands | 2.2 | 2.2 |
| ran a training command | 90 % | 90 % |
| **submitted voluntarily** | **1/20** | **0/20** |
| cost / trajectory | $0.0158 | $0.0192 |

**Answer to RQ4, and it is mostly negative.** Grouped by outcome at Tier H:

| label | n | reads | patches | distinct files | first patch turn | prompt tokens |
|---|---|---|---|---|---|---|
| FULL_FIX | 8 | 10.5 | **1.5** | 9.5 | 13.1 | 100,090 |
| PARTIAL_FIX | 4 | 10.5 | **1.5** | 9.0 | 13.3 | 102,476 |
| WRONG | 8 | 10.5 | **0.5** | 9.5 | 13.7 | 97,176 |

Reads are **identical to one decimal place** across all three outcomes. The only
discriminator is whether a patch was written at all. The strategy is **exhaustive
enumeration, not hypothesis-driven search**: read nearly every module, then commit
or do not. Trajectory metrics are therefore useless as a cheap reward proxy here,
and an 8-module package is small enough to read exhaustively inside 14 turns — to
make localization the bottleneck, the repository must be too large to read, which
is a scale change rather than a bug-difficulty change.

**39 of 40 flash episodes ended on budget exhaustion, not by choice.** Grading runs
on final state so nothing is invalidated, but every difficulty number in this report
is difficulty **at a 14-turn budget**, and some Tier H failures may be
budget-limited rather than capability-limited. This session cannot separate those.

## 14. Held-out evaluation

| kind | construction | status |
|---|---|---|
| held-out configurations | hidden checks sweep shapes/lengths/dtypes the visible fixtures never use | ✅ in every task |
| held-out probe inputs | independent oracle uses `SEED=424242424`, disjoint from all graded fixtures | ✅ |
| held-out instances | same family at different scope (F1: t1→m1→h1, etc.) | ✅ |
| **held-out task** | **`h5_masking_triple_defect`** reserved before results, graded once, no adjustment after | ✅ 0/4 |

These test whether the **verifier and benchmark** overfit their own fixtures. They
are not a claim about model generalization; no policy was trained.

## 15. Failure cases

Three defects in our own work, all found by measurement and all logged in
[`PROTOCOL_CHANGELOG.md`](PROTOCOL_CHANGELOG.md).

**R1a — a metric that measured nothing.** `files_changed` compared the workspace
against *gold*, so it reported where the bug was injected, not what the model
edited. Caught in the deliberate one-episode smoke test before the primary run.

**R1b — valid patches rejected on formatting.** The model produced the correct
causal-mask fix with a bare `@@` and no line numbers; the parser rejected it. This
is the second time our applier, not the model, caused an "invalid" verdict — Phase
0.5 found 3 of 4 invalids were ours. Fixed to accept headerless hunks located by
content, while still rejecting prose, non-existent removals and no-ops.

**R3 — the independent oracle was weaker than the verifier it audits.** This is the
important one. The first Tier H audit reported two hardened false negatives on
`h3`. Reading them showed `h3` e1 had `files_edited_by_model = []` — **the model
made no edits at all** — so `hardened=0` was right and *our oracle was wrong*. The
oracle compared logits, token counts, final loss and the LR trace; `h3`'s defects
live entirely in the accumulation denominator and the clip/step order, and none of
those four comparisons can see either. It was structurally blind to family F4.

Two invariants were added (accumulation equivalence, clipping effectiveness), all
audits regenerated, no trajectory re-sampled. The correction **lowered** our own
Tier H headline from 10/20 to 8/20 — a revision in that direction after seeing the
number is the opposite of tuning.

The generalizable lesson: G3 says a verifier is only a proxy for intent. The same
applies one level up. *Independent* bought independence from the verifier's
fixtures and bought nothing about coverage.

## 16. Threats to validity

- **n is small.** 4 episodes per task, 2 for the confirmatory run. A single rollout
  moves a task between difficulty classes. No p-values, no significance claims, no
  SOTA claims anywhere in this project.
- **The 14-turn budget confounds every difficulty number** (§13).
- **Single provider.** Both models are DeepSeek; a shared idiosyncrasy would look
  like a task property.
- **Fuzz probes are known-unknowns.** 13 probes drawn from the exploit taxonomy
  Phase 0.5 produced. A 0.00 hardened FPR means the shortcuts *we imagined* are
  blocked.
- **The 8-module repo is a scale model**, not a research codebase. What transfers is
  the shape of the work, not the scale.
- **Gap 1 is live.** The Tier M/H hardened suite has no L1 return-type check, so a
  semantically-correct contract violation passes.
- **No optimization pressure.** Every rate here is for frozen models. G3 predicts a
  fixed verifier degrades as the policy strengthens, and nothing here can see that.

## 17. Limitations

This phase measures **RL environment readiness**, not **RL training effectiveness**.
No policy was trained, no checkpoint exists, no training curve exists, and no
improvement metric is claimed. The `minimal_grpo.py` artifact is a CPU learning
exercise on a toy task, not a result about these environments.

## 18. GPU training as optional future work

Not executed. What Phase 2 could add that this phase provably cannot:

1. **The falsification the project was designed for.** Constructed exploits prove a
   naive/hardened gap *exists*; 50 trajectories show a frozen model does not use
   it. Only a policy under gradient pressure tests whether it *matters*.
2. **Reward-integrity A/B.** Two arms, matched seeds and budgets, only the reward
   differing; training reward rises in both while held-out performance diverges.
   Requires parameter updates.
3. **Held-out structure as a model claim.** Today it tests verifier overfitting;
   with training it would test policy generalization.
4. **G3's prediction, directly.** Re-run the fuzz audit against the trained policy
   and watch a fixed verifier degrade.

Gated on: closing Gap 1, a persistent grading container (§8), and a documented
FP/FN baseline — which this report now provides.

## 19. Conclusion

# PUBLIC_READY_LOCAL = YES

The Phase 0.6/1 blocker was a single missing measurement. It has been made, three
times over, and the result is a coherent research artifact rather than a
scaffold:

- **15 tasks**, all REAL or REAL-DERIVED, 12 fully source-audited, 0 rejected;
- a **measured difficulty gradient** — 4 easy, 4 in band, 2 too hard — obtained in
  one redesign round with the second left unused;
- **50 real multi-turn trajectories** across two models, every one independently
  audited;
- the headline empirical result: **naive FPR climbs 0.10 → 0.20 → 0.60 with
  difficulty while hardened holds at 0.00/0.00**;
- an adversarial audit with a **documented six-item verification horizon**, run
  before expansion, that found two real gaps in our own verifier;
- a container sandbox with a written threat model, verified running;
- three self-caught measurement defects, one of which **lowered our own headline
  number**;
- 148 tests, reproducible scripts, no credential in the repository.

What makes it publishable is not that the numbers are good. It is that the negative
findings are load-bearing: AdamW does not produce NaN from gradient-magnitude bugs;
trajectory shape carries no signal about outcome; our own audit oracle was blind to
a whole family. Those are the parts a reviewer cannot get from a README.

**Not published.** No GitHub push, per instruction. Readiness is a local assessment.

## 20. Reproduce

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install "verifiers==0.3.0" pytest tiktoken
python scripts/build_sandbox.py                     # Docker image + sandbox check

.\.venv\Scripts\python.exe -m pytest tests -q       # 148 passed
python scripts/audit_tasks.py                       # Tier E gold/buggy
python scripts/audit_repo_tasks.py                  # Tier M
python scripts/audit_repo_tasks_h.py                # Tier H
python scripts/audit_cheats.py                      # 7/7 exploits
.\.venv\Scripts\python.exe scripts/fuzz_verifier.py            # 13 probes
.\.venv\Scripts\python.exe scripts/source_alignment_audit.py   # 15 tasks
.\.venv\Scripts\python.exe scripts/measure_verifier_cost.py

# real-model evaluation (needs DEEPSEEK_API_KEY in the shell, ~$1.20 total)
.\.venv\Scripts\python.exe scripts/run_deepseek_repo_eval.py --tier M --episodes 4 `
    --model deepseek-v4-flash --work .sandbox_work --out artifacts/tier_m_primary.jsonl
$env:TRGYM_WORK='.sandbox_work'; $env:TRGYM_TAG='tier_m'
.\.venv\Scripts\python.exe scripts/audit_tier_m.py artifacts/tier_m_primary.jsonl
.\.venv\Scripts\python.exe scripts/analyze_results.py artifacts/tier_m_audit.json
```

## Artifact index

| file | contents |
|---|---|
[`LITERATURE_GUARDRAILS_2026.md`](LITERATURE_GUARDRAILS_2026.md) | 8 papers → 8 binding design rules, with enforcement locations |
[`EXPERIMENT_MANIFEST.json`](EXPERIMENT_MANIFEST.json) | frozen v1 config, 27 file hashes |
[`PROTOCOL_CHANGELOG.md`](PROTOCOL_CHANGELOG.md) | R1a, R1b, R3 with cherry-picking analysis |
[`SOURCE_ALIGNMENT_AUDIT.csv`](SOURCE_ALIGNMENT_AUDIT.csv) | 15 tasks × 4 gates |
[`VERIFIER_FUZZ_AUDIT.md`](VERIFIER_FUZZ_AUDIT.md) / `.csv` | 13 probes, five metrics, two gaps |
[`VERIFIER_QUALITY_MATRIX.md`](VERIFIER_QUALITY_MATRIX.md) | three axes + 6-item verification horizon |
[`DIFFICULTY_DISTRIBUTION.md`](DIFFICULTY_DISTRIBUTION.md) | per-task difficulty, all tiers |
[`TASK_CHAINS.md`](TASK_CHAINS.md) | E→M→H stepping stones, what each tier adds |
[`TRAJECTORY_EFFICIENCY.md`](docs/history/TRAJECTORY_EFFICIENCY.md) | RQ4, five findings |
[`SANDBOX_DESIGN.md`](SANDBOX_DESIGN.md) | threat model, container config, limitations |
`TIER_M_REAL_MODEL_AUDIT.csv`, `TIER_H_REAL_MODEL_AUDIT.csv`, `TIER_M_CONFIRMATORY_PRO_REAL_MODEL_AUDIT.csv` | per-trajectory labels |
`artifacts/*.jsonl` | full trajectories: every tool call, observation, patch, token count |
`artifacts/raw/*_final_sources/` | final state of every file each trajectory edited |
