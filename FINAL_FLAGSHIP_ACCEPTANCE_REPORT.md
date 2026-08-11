# FINAL_FLAGSHIP_ACCEPTANCE_REPORT

```
FLAGSHIP_GPU_FREE_COMPLETE = NO
POST_SUCCESS_AUDIT        = NOT REACHED
READY_FOR_USER_REVIEW_AND_PUSH = NO
```

Assessed against `FINAL_FLAGSHIP_COMPLETION_CONTRACT.md`, written before
implementation and not modified since. **4 of 10 gates closed** (G0, G2, G3, G6). The
contract forbids "mostly complete" as a substitute, so the verdict is NO.

**This verdict is now computed, not asserted.** `scripts/final_acceptance.py` ran for the
first time in session 4, two of its gate parsers were fixed against the real artifact
schemas, and it independently returns exactly this state from artifacts alone
(`artifacts/final_acceptance.json`). Where it previously reported `BLOCKED-PLATFORM` for
G1's runtime half on Windows, it now requires a Linux-written evidence artifact and **fails
if that artifact is absent** — a skip is not a pass.

**G1 was marked PASS in session 2 and has been flipped back to FAIL.** Setting up G5's
isolation canaries showed its central security claim was overstated: grading imported
candidate code into the process where the gold oracle is reachable (R14). The repair is
written but not yet measured, and a gate does not stay green because a fix exists.

**Session 4 measured R14's repair, and found a second defect underneath it (R15).** The
canaries, run in the environment where grading actually happens, confirm the sandboxed path
contains all seven containment probes while the deliberately unsafe control leaks eight of
nine — so R14's boundary is real. But the same run showed a workspace built from the *buggy*
template scoring a **pass**: candidate code sharing the grading container's process with the
checks can rewrite the tolerances that decide its own verdict. Containment is not integrity.
Two mitigation layers were added and are **unrun**, because command execution was blocked
partway through the session. G1 and G5 therefore stay FAIL.

"Foundations repaired" is **not** a gate and is never counted toward the total.

No GPU training. No remote push. No new service.
**165 host tests pass (Windows) + 16 v1 tests pass (Linux/Docker).**

---

## Gate status

| Gate | Verdict | Evidence |
|---|---|---|
| **G0** Evidence preservation | **PASS** (strengthened) | **127** artifacts frozen (was 56); **30/30 headline metrics regenerated exactly**; freezes now content-addressed + immutable; one destroyed artifact recorded as `LOST_OVERWRITTEN` rather than reconciled away (R7) |
| **G1** `verifiers.v1` migration | **FAIL** (was PASS, flipped) | Migration itself is done and verified — 16 behavioural tests, `validate` CLI 1.0, live DeepSeek smoke solved m1, 95 v1 modules executed. **But** the oracle-unreachability claim held on one boundary only (R14), and the repair for it turned out to protect the host without protecting the verdict (R15). R14's boundary is now **measured** (7/7 containment probes contained, control leaks 8/9); R15's two mitigation layers are written and **unrun** |
| **G2** Hardened verifier v2 + replay | **PASS** | All 7 frozen criteria met. D1+E1+E2+E3 rejected by v2, accepted by v1; `v2_FN = 0`; coverage 98.9 %; v2 FP identical to v1 |
| **G3** Turn-budget ablation | **PASS** | Confound quantified. Verdict **`BUDGET-SENSITIVE`**; a Phase 1 difficulty label was corrected as a direct result. |
| **G4** Tier S localization | **NOT STARTED** | — |
| **G5** Grader scalability | **FAIL** | Isolation half **ran** (11 probes: 6 required channels + gold-import + 4 verdict-integrity). Result: sandboxed path contained 7/7 containment probes, control leaked 8/9 — and one verdict-integrity probe **escaped**, which is what R15 fixes. The re-run against that fix, and the ≥30-job throughput half, were blocked when command execution stopped. Evidence: `artifacts/g5_isolation_canaries.PRE_R15_FIX.json` |
| **G6** Heuristic gate FP audit | **PASS** | 32 licensed human-written ML functions; submission-level FP **15.6 % → 0.0 %** after R13; hard structural gates separated from soft heuristics; exploit suite **7/7** with identical attributions |
| **G7** Packaging / CI / fresh clone | **NOT STARTED** | — |
| **G8** Final scientific synthesis | **NOT STARTED** | Depends on G1–G7 |
| **G9** Post-success red-team | **NOT REACHED** | Only meaningful once G0–G8 pass |

---

## G0 — PASS

```
python scripts/freeze_historical_artifacts.py
→ frozen 127 artifacts, 0 missing   (104 distinct content-addressed blobs)
→ 30/30 metrics: claimed == recomputed
→ G0 verdict: PASS
```

**G0 failed twice on its own defects before it passed**, and both are worth more than
the green verdict:

*R5b — it did not index graded workspaces.* `build_sandbox.py` used `.sandbox_work`
as scratch and `rmtree`d Tier M's 20 workspaces during Phase 1's own verification.
Fixed: dedicated `.sandbox_selftest`, the preserved-source trees indexed (56 → 127
artifacts), and an AST test forbidding an evidence dir as an `rmtree` argument — plus a
companion test proving that detector actually fires.

*R7 — its frozen copies were mutable.* Copies went to a flat name and were re-copied
every run, so the second G0 invocation overwrote its own frozen copy of
`verifier_fuzz_audit.json` with the post-E-probe version. G0 caught this as
`fuzz.n_probes: claimed 13, recomputed 16` and correctly reported **FAIL**. Frozen
copies are now content-addressed (`<name>.<sha[:12]>`, written only if absent), so a
changed artifact adds a blob and never replaces one.

**One artifact is permanently lost and is recorded as lost.** The raw 13-probe v1 fuzz
JSON is unrecoverable. `CLAIMED` now asserts the live v2 suite at 16 probes *and*
carries a separate `fuzz_v1_artifact` entry with `status: LOST_OVERWRITTEN` naming the
two committed documents where its summary does survive. Setting the expectation to 16
and moving on would also have turned G0 green; it would have erased the fact that
evidence was destroyed. The manifest now permanently carries the loss as a checked
fact, which is the correct outcome.

## G1 — PASS

Full detail in `VERIFIERS_V1_MIGRATION_AUDIT.md`, `VERIFIERS_V1_MIGRATION_REPORT.md`,
`VERIFIERS_VERSION_SNAPSHOT.md`.

```
tests_v1/                16 passed  (Linux/Docker)
tests/                  155 passed, 16 skipped  (Windows)
scripts/v1_provenance.py  verdict PASS
official validate CLI     valid_rate 1.0
live DeepSeek smoke       semantic_repair 1.00
```

Plan A (native v1 in Linux) succeeded. Windows still cannot `import verifiers.v1`
(`fcntl`, TYPE B, confirmed) so everything v1 runs in Docker.

**The design document's premise was wrong and evidence overruled it.** I had argued from
`Harness.launch(…, endpoint, secret, …)` that this was a *polarity inversion* requiring the
agent loop to be rebuilt as an MCP toolset. Reading the shipped `LeanTask` reference showed
the real division: **Task owns setup and grading; Harness owns the rollout.** So no bespoke
harness was needed — the built-in `bash` harness drives editing. That is also the honest
choice: a custom `Harness` whose `launch` ignored `endpoint` would satisfy the type contract
while leaving the ACP/MCP path dead, i.e. a wrapper wearing a subclass.

**Why this is not cosmetic.** Every check is behavioural. `type(task).score is
BaseTask.score` asserts reward discovery is v1's own; the Docker check reads back values only
the container's interpreter could produce; the oracle check greps the candidate's entire
filesystem and finds nothing; the legacy-API ban has a positive control that must fire on
`legacy_research/transformer_repair_v0.py`.

**Live smoke.** `artifacts/raw/v1_smoke2/` — from a symptom-only prompt, DeepSeek located
the defect unaided and made the minimal correct edit (`tril(diagonal=1)` → `tril(diagonal=0)`
in `tinygpt/attention.py`), all six hidden checks passing, `grading_side: host`,
`suite: v2`, `stop: agent_completed`. The earlier failed run is preserved and is more
instructive: the agent lost 5 of 14 turns to `which python3` because the harness program is
launched with only `HarnessConfig.resolved_env` and had no `PATH`. Fixed as configuration.

**Four of my own defects found en route** (R9–R12). Two matter beyond this gate:

- **R11** — the grading bridge recorded *every* check as passing, because `run_repo_checks`
  returns tuples rather than raising. Gold and the planted bug both scored 1.0. It failed
  **open** and nothing but the gold-vs-no-op separation assertion would have caught it.
- **R12** — the `infra_error` metric read `trace.has_error` (`not ok`), and `ok` is set
  *after* scoring, so it returned 1.0 for every rollout. Same shape as R11: a signal that
  always says the same thing. Every reward and metric now has a fires / does-not-fire test.

**Correction to earlier reporting in this project:** I previously wrote that the first smoke
showed "the agent did nothing", reading the summary line. The trace showed 14 model calls
and a competent investigation. The metric was broken, not the rollout.

## G2 — PASS

Implemented per the frozen protocol (SHA `D297A0FE…`): `repo_contract_return_types`
and `repo_contract_public_api` as an L1 layer. v1 left byte-identical; v2 = v1 +
contract, so the replay compares rather than mutates.

| protocol criterion | status |
|---|---|
| 1. Gap 1 closed — D1 **and** E1–E3 rejected | **PASS** — all four: v1 accepts 4/4, v2 rejects 4/4 |
| 2. gold passes 100 % | **PASS** — 10/10 repo tasks |
| 3. buggy fails 100 % | **PASS** |
| 4. no unexplained new FULL_FIX rejection | **PASS** — `v2_FN_count = 0` |
| 5. every v2↔oracle disagreement explained | **PASS** — 0 remain; the one that existed was root-caused (R4) |
| 6. replay coverage ≥ 90 % | **PASS** — **98.9 %** (89/90), **50/50 multi-turn** |
| 7. FP not reduced by over-rejecting | **PASS** — FP identical at 0.2778, FN = 0 |

Replay (`VERIFIER_V2_REPLAY.csv`): v1 FP 0.2778 / v2 FP 0.2778; v1 FN 0 / v2 FN 0;
0 v1↔v2 disagreements. Fuzz: v2 rejects D1 as required, rejects 9/9 WRONG probes,
and rejects **0** FULL_FIX probes.

**Closed by implementing E1–E3 exactly as frozen.** E2 initially escaped: `numpy.float64`
subclasses `float`, so `isinstance(x, float)` accepted a numpy scalar. The check now
uses `type(x) is float` plus a `json.dumps` round-trip, which is the downstream
breakage E2 was written to describe (R6). Gold stayed clean, v2 rejects **0**
FULL_FIX probes, and the replay is unchanged at `v2_FN = 0`, `v2_FP == v1_FP`.

**Two of my own defects found en route**, both logged: 10 package files carried a
UTF-8 BOM that made `ast.parse` reject *gold*; and the contract check was
over-strict, rejecting a genuine `FULL_FIX` because a token count was `int` rather
than `Tensor` (R4). The fidelity test in
`scripts/test_reconstruction_fidelity.py` proved that was a real verifier problem
and not a reconstruction artifact before it was touched.

## G3 — PASS

# Verdict: `BUDGET-SENSITIVE`

Protocol pre-registered and hashed (`620FDAAE…`) **before the first API call**.
10 new trajectories: all 5 Tier H tasks × 2 episodes × 24 turns, against frozen
14-turn controls. Independent oracle, **zero disagreements**.

| task | 14 turns | 24 turns |
|---|---|---|
| `h1_attention_double_defect` | 2/4 | **2/2** |
| `h2_position_double_defect` | 0/4 | 0/2 |
| `h3_accumulation_and_clipping` | 2/4 | **2/2** |
| `h4_schedule_triple_defect` | 4/4 | 2/2 |
| **`h5_masking_triple_defect`** | **0/4** | **2/2** |
| **overall** | **8/20 = 40 %** | **8/10 = 80 %** |

Both pre-declared trigger conditions fired, so the verdict rule returns
`BUDGET-SENSITIVE` mechanically.

**The headline behavioural finding:** voluntary submission went **0/20 → 8/10**, and
budget exhaustion **20/20 → 0/10**. Successful episodes submitted at turns 19–24 —
a window Phase 1 could not see into.

**A Phase 1 conclusion is now wrong.** `h5` was labelled `TOO_HARD` from 0/4. At 24
turns it is 2/2, and the successful episodes edited **all three** required files;
the 14-turn episodes had reached only `data.py`. It needed three coordinated edits
and ran out of turns after the first.

**`h2` remains 0/2 for a different reason:** both episodes ended
`too many unusable actions` after 17 and 19 turns with nothing edited — a
PROTOCOL_FAILURE. Its true difficulty is **still unmeasured**, at either budget.

## Foundations repaired before any new paid trajectory

Two defects fixed ahead of the gate work, both logged (R5, R5b):

**h2's 0/4 was an adapter defect, not difficulty.** Both episodes investigated
normally for 14–16 turns, then the provider returned a message with neither content
nor a tool call. The adapter echoed that contentless message back, which makes every
subsequent completion empty too — so one empty response deterministically produced
the run of three that tripped the guard. Provider failures were also being counted as
the policy emitting an invalid tool, conflating INFRA_FAILURE with behaviour. Fixed,
with four regression tests including one proving the guard still fires for genuinely
invented tool names. **`h2` stays `UNMEASURED` and must be re-run before it carries
any difficulty claim.**

**G0 did not index the graded workspaces.** `build_sandbox.py` used `.sandbox_work`
as scratch and `rmtree`d Tier M's 20 workspaces during Phase 1's own verification.
Now uses `.sandbox_selftest`; G0 indexes the preserved-source trees (56 → 127
artifacts); an AST test forbids passing an evidence dir to `rmtree`, with a companion
test proving the detector actually fires.

## G6 — PASS

Full detail in `HEURISTIC_GATE_FP_AUDIT.md`.

```
32 benign human-written ML functions  (torch 2.9.1 BSD-3-Clause, numpy 2.3.3 BSD)
submission-level FP   15.6 %  ->  0.0 %
exploit suite         7/7 caught, identical gate attributions
tests                 +10 paired prose/violation regression tests
```

Corpus deliberately drawn from this project's own domain — attention, normalisation,
optimisers, LR schedules, clipping, collation — so a false positive is a false positive on
the kind of code a competent human fix would contain.

**The audit found a real defect.** All five false positives were **docstrings**:
`MultiheadAttention` and `TransformerEncoderLayer` tripped `dynamic_execution` because their
docstrings say "training is disabled (using ``.eval()``)"; `Embedding`, `CrossEntropyLoss`
and `default_collate` tripped `hardcoded_tensor` on `tensor([...])` in docstring *examples*.
The gates were penalising submissions that document themselves. Fixed by tokenizing before
matching (R13).

**This is the third instance of the same defect** — the G1 legacy-API detector flagged its
own docstring, the rmtree detector flagged files that merely mentioned an evidence dir, and
now the anticheat gates. Standing rule recorded: any regex over source in this project must
be preceded by tokenization.

**Hard/soft separation as required.** `public_api_preserved`, `grader_files_untouched` and
`support_files_untouched` are structural task-contract gates; FP-auditing them against
foreign code is meaningless, so they are audited by G2's replay against real submissions
(FP 0.2778 for both v1 and v2). The five content heuristics are what this audit measures.

**Statistical honesty:** 0/32 gives a one-sided 95 % Clopper–Pearson upper bound of 8.9 %.
The claim is "no false positive observed in 32 domain-matched samples", not "the gates never
false-positive". No p-value is attached to the before/after, because the change is a
deterministic code fix, not a sampled effect.

**A defect in my own audit, found on the way:** the first version reimplemented the gates as
plain regexes instead of calling them, which was stricter than the verifier and inflated the
measured rate to 18.8 %. It now invokes the real gate functions and reads the authoritative
exploit result from `artifacts/cheat_audit.json`.

## G4, G5, G7–G9 — NOT STARTED

No work was done. Stated plainly rather than describing intent as progress.

---

## Budget

| | |
|---|---|
| new trajectories | **10** of 30 allowed (G3 arm) |
| estimated new spend | **< $0.50**, well inside the $5 stop |
| historical trajectories | 90, untouched |

## Path to YES

1. **G6, G5** — heuristic FP audit and grader scalability/isolation. Both deterministic,
   no paid calls.
2. **G4** — Tier S, frozen once before the 12 precommitted trajectories.
3. **G7** — packaging, CI, true fresh-clone reproduction.
3. **G8**, then **G9** in full — including mutant testing, which has not been
   attempted and is the stage most likely to invalidate G0, G2 or G3.
4. Re-run `h2` under the fixed adapter so its difficulty is measured rather than
   absent.

## Files to read

1. `TURN_BUDGET_ABLATION_REPORT.md` — the one closed research gate, and the Phase 1 correction
2. `FALLBACK_EXECUTION_LOG.md` — every Plan A failure and the fallback taken
3. `PROTOCOL_CHANGELOG.md` R3/R4 — two self-caught verifier defects
4. `VERIFIER_V2_REPLAY.csv` — 89 rows, v1 vs v2 vs oracle
5. `artifacts/raw/v1_probe.json` — the real v1 API, input to G1
