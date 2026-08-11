# LITERATURE_GUARDRAILS_2026

Not a literature review. Only findings that **changed what this project does**,
each paired with the concrete design rule it produced and where that rule is
enforced in code.

All eight papers were read from arXiv abstract pages, not from blog summaries.
Verified 2026-08-10.

---

## G1 — Weak verifiers accept wrong patches at a measurable, high rate

**Source:** *Auditing Reward Hackability in Code RL Training Environments*
([arXiv:2606.16062](https://arxiv.org/abs/2606.16062))

**Finding, quoted:** *"On a 49-task sample of SWE-bench Verified, 28.5% of tasks
have test suites weak enough that a Docker-verified incorrect patch passes them.
On 20 R2E-Gym tasks across 6 repositories, the same pipeline at single-shot
exploit generation yields 25.0%."* They further measure model Pass@1 as
**+14.14 pp higher on hackable tasks** (95% CI [+11.80, +16.48]) across 134
frontier submissions.

**Implication for us:** A pass rate on a task set with unaudited verifiers is not
a capability measurement. Roughly a quarter of it could be exploitability. Our
Phase 0.5 report claimed a 0 % naive false-positive rate from 40 trajectories —
that number is now understood as "no exploit was attempted", not "no exploit
exists".

**Design rules:**
1. Every reported pass rate is accompanied by a verifier FP/FN measurement from a
   deliberate exploit suite, never reported alone. → §6 of this run,
   `VERIFIER_FUZZ_AUDIT.md`.
2. A gold-sanity gate runs on every task in CI: gold must pass, buggy must fail,
   asserted before any model sees it. → `scripts/audit_repo_tasks.py`,
   `tests/test_harness.py`.
3. The +14.14 pp result implies exploitable tasks look *easier*. So an
   unexpectedly high pass rate is treated as a **verifier alarm**, not a
   difficulty signal, until fuzzed.

---

## G2 — Fuzz the verifier *before* training, not after

**Source:** *Before the Model Learns the Bug: Fuzzing RLVR Verifiers*
([arXiv:2606.01066](https://arxiv.org/abs/2606.01066))

**Finding:** *"if the verifier is wrong, optimization can learn the bug."* The
framework generates adversarial completions and reports five quantities: false
positives, false negatives, disagreement between a buggy and a stricter verifier,
exploit patterns, and uncertainty indicators.

**Implication for us:** This is exactly our naive-vs-hardened pair, and it names
a metric we were not reporting: **disagreement rate** as a first-class number,
independent of which verifier is right.

**Design rules:**
1. Fuzz audit is a **precondition** for expanding the task set, not a follow-up.
   → run before any Tier H work this session.
2. Report all five: naive FPR, hardened FPR, naive FNR, hardened FNR, and
   naive↔hardened disagreement. → `VERIFIER_FUZZ_AUDIT.md`.
3. Ground truth for FP/FN comes from the independent equivalence probe, never
   from the other verifier. → `scripts/audit_real_model.py`, `PROBE_SEED`
   disjoint from every graded fixture.

---

## G3 — Faithfulness / robustness / scalability, and no fixed verifier survives

**Source:** *The Verification Horizon: No Silver Bullet for Coding Agent Rewards*
([arXiv:2606.26300](https://arxiv.org/abs/2606.26300))

**Finding, quoted:** *"every verifier we can build is only a proxy for human
intent, never the intent itself"*; *"no fixed reward function can remain effective
as policy capability continues to grow; and verification must co-evolve with the
generator."* Signals are characterised along **faithfulness, robustness,
scalability**, and getting all three at once is the central open problem.

**Implication for us:** Our reports have been implicitly framing the hardened
verifier as the correct one. It is a better proxy, not the truth. And a verifier
tuned against today's DeepSeek does not stay valid against a stronger policy —
which matters because Optional Phase 2 is exactly "make the policy stronger".

**Design rules:**
1. Score every verifier on all three axes explicitly, including measured
   per-verification cost. → `VERIFIER_QUALITY_MATRIX.md`.
2. Never write "the verifier is secure" or "reward is sound". Write the measured
   FPR/FNR and the known horizon. → enforced by review of every report.
3. State the verification horizon per task: the class of wrong-but-passing
   submissions we know we would miss.
4. Phase 2 must re-run the fuzz audit against the trained policy; a Phase 1 audit
   does not transfer. → recorded in the Phase 2 section of the final report.

---

## G4 — "From a real PR" does not mean the task is real

**Source:** *PAIChecker: Uncovering and Checking PR-Issue Misalignment in
SWE-Bench-Like Benchmarks* ([arXiv:2607.28587](https://arxiv.org/abs/2607.28587))

**Finding:** **13.6 % of SWE-bench Verified instances are misaligned** — the
linked PR does not actually address the linked issue — across five patterns in
eleven scenarios. Detection needs code-level validation, not link-following.

**Implication for us:** `REAL_BUG_EVIDENCE.md` cites issues and PRs as provenance
and stops there. By this result, roughly one in seven such citations would not
survive inspection. Our claim "10/10 REAL or REAL-DERIVED" is currently a
citation count, not an audit result.

**Design rules:**
1. Every REAL / REAL-DERIVED task passes a four-gate alignment audit before it
   counts: **(A)** issue intent matches patch intent, **(B)** pre-fix behaviour
   reproduces, **(C)** the gold fix resolves it, **(D)** the reduction to our
   tiny/repo scale preserves the same root cause. → `SOURCE_ALIGNMENT_AUDIT.csv`.
2. Any gate failing → `REJECTED_SOURCE`. No re-interpretation to save a task.
3. Distinguish **cited** from **audited** provenance in every count.

---

## G5 — Iterative self-verification, and hacking detection in the loop

**Source:** *SWE-Universe: Scale Real-World Verifiable Environments to Millions*
([arXiv:2602.02361](https://arxiv.org/abs/2602.02361))

**Finding:** Large-scale environment construction includes iterative
self-verification and in-loop hacking detection; auto-generated tasks are not
trained on directly.

**Implication for us:** Our task generation is already declarative-mutation
based, which is stronger than free generation. But we validate a task once at
construction and never again.

**Design rules:**
1. Gold-sanity and buggy-fails checks run on every task on **every** test run, so
   a later edit that silently breaks discrimination fails CI.
2. Hacking detection runs **in the evaluation loop**, not only post-hoc: every
   trajectory records gates fired and files edited at grade time.

---

## G6 — Difficulty and structure beat volume; stepping stones beat variety

**Source:** *A Deep Dive into Scaling RL for Code Generation with Synthetic Data
and Curricula* ([arXiv:2603.24202](https://arxiv.org/abs/2603.24202))

**Finding, quoted:** *"data diversity and structure, rather than volume alone,
become the limiting factor"*; *"stepping stones, i.e. easier and harder variants
of the same core task, support curriculum-based training"*; the multi-turn
generation approach *"substantially improves the yield of valid synthetic
problems"*.

**Implication for us:** This directly justifies the instruction to cap at 15
tasks. It also says our E→M pairing (same root cause, different scope) is the
*right* structure rather than a shortcut, and that difficulty should climb inside
a family rather than across unrelated ones.

**Design rules:**
1. Hard cap of **15 accepted tasks** this session. Quality gates over count.
2. New difficulty is added as a **stepping stone in an existing family**
   (E → M → H), never as an unrelated new bug. → `TASK_CHAINS.md` if Tier H
   is built.
3. Target the medium band: uniformly-easy and uniformly-hard sets are equally
   useless, which our own zero-advantage unit test states as an identity.

---

## G7 — Multi-turn rollouts become long-context problems

**Source:** *ProRL Agent: Rollout-as-a-Service for RL Training of Multi-Turn LLM
Agents* ([arXiv:2603.18815](https://arxiv.org/abs/2603.18815))

**Implication for us:** Confirmed empirically in our own smoke test before this
document was written: **one 10-turn episode consumed 52,328 prompt tokens** for
5,400 completion tokens. Prompt cost dominates and grows superlinearly with turns.

**Design rules:**
1. Hard budgets stay: 14 turns, 24 commands, 900 s wall, 180 s per command. →
   `trgym/harness/tools.py::Budget`.
2. Tool output is clipped to 6,000 chars so one `read_file` cannot blow the
   context. → `MAX_OUTPUT_CHARS`.
3. Record turns / reads / runs / patches / tokens / cost per trajectory and ask
   whether harder tasks need *more investigation* or merely *more tokens*. →
   `TRAJECTORY_EFFICIENCY.md`.

---

## G8 — Self-play SWE-RL is a Phase 2 concern, and it raises the bar for us

**Source:** *Toward Training Superintelligent Software Agents through Self-Play
SWE-RL* ([arXiv:2512.18552](https://arxiv.org/abs/2512.18552))

**Implication for us:** Read as a constraint on Optional Phase 2, not on this
session: a co-evolving generator/verifier setup requires a verifier whose FP/FN
are *known*, otherwise self-play amplifies the verifier's bugs. Reinforces G3.

**Design rule:** Phase 2 is gated on a documented FP/FN baseline from Phase 1. No
training on a verifier we have not audited.

---

## Anti-p-hacking rule adopted from this reading

G1 and G6 together create an obvious temptation: tune the tasks until DeepSeek
lands near 2/4, then report a "healthy dynamic range". That is fitting the
benchmark to one model.

**Rule, binding for this session:** at most **two** difficulty redesign cycles.
Every version's results are retained. If after two rounds the set is still
uniformly easy or uniformly hard, the result is written as
`DIFFICULTY_CALIBRATION_FAILED` and reported as a negative finding. A tuned pass
rate would be a worse artifact than an honest failure.

## Statistical honesty rule

n = 4 episodes per task. No p-values, no significance claims, no SOTA claims.
Counts and rates only, with the raw trajectory as the evidence of record. This
session measures **RL environment readiness**, not RL training effectiveness.

---

## Where each rule is enforced

| rule | enforced in |
|---|---|
| G1.2 gold-sanity gate | `scripts/audit_repo_tasks.py`, `tests/test_harness.py` |
| G2.1–2.3 pre-training fuzz, five metrics, independent ground truth | `scripts/fuzz_verifier.py`, `VERIFIER_FUZZ_AUDIT.md` |
| G3.1 three-axis scoring incl. cost | `VERIFIER_QUALITY_MATRIX.md` |
| G4.1 four alignment gates | `scripts/source_alignment_audit.py`, `SOURCE_ALIGNMENT_AUDIT.csv` |
| G5.1 re-validation every run | `tests/` (task audits run in CI path) |
| G6.1 task cap, G6.2 stepping stones | `TASK_CHAINS.md`, task registry |
| G7.1–7.3 budgets and cost accounting | `trgym/harness/tools.py`, `TRAJECTORY_EFFICIENCY.md` |
| anti-p-hacking | `PROTOCOL_CHANGELOG.md`, versioned task sets |
