# FINAL_FLAGSHIP_COMPLETION_CONTRACT

Machine-verifiable acceptance contract for the GPU-free flagship. Written
**before** implementation, 2026-08-10, so that the finish line cannot move.

Final state is binary. `scripts/final_acceptance.py` computes it from artifacts,
not from prose:

```
FLAGSHIP_GPU_FREE_COMPLETE = YES   iff  G0..G9 all PASS
FLAGSHIP_GPU_FREE_COMPLETE = NO    otherwise
```

Forbidden as substitutes for a failed gate: *conditional*, *mostly complete*,
*basically done*, *95 %*, *ready except*, *only X remaining*, *minor gap*.
**NO is a legitimate result. Fake completion is not.**

GPU RL training is **Optional Phase 2** and is not part of any gate.

---

## G0 — Evidence preservation

| | |
|---|---|
| **PASS iff** | every load-bearing Phase 0/0.5/1 headline metric can be regenerated from frozen artifacts, verified by re-running the analysis scripts and diffing against the reported numbers |
| **artifact** | `FROZEN_PHASE1_MANIFEST.json`, `artifacts/frozen_phase1/` |
| **machine check** | `final_acceptance.py` recomputes ≥6 headline metrics and diffs them against the manifest's recorded values; any mismatch → FAIL |
| **FAIL if** | any Phase 1 number cannot be reproduced, or any historical artifact was overwritten |

## G1 — Full migration to current `verifiers.v1`

| | |
|---|---|
| **PASS iff** | the public package path is 100 % v1 (`import verifiers.v1`), and all ten of: fresh install; Taskset loads; tasks enumerate; TaskData immutable; Docker Runtime works; reward computed from Trace; grading artifacts unreadable by the candidate; setup/finalize clean; no cross-rollout state leak; v1 CLI dry-run — plus **one real DeepSeek smoke through the v1 lifecycle** |
| **artifact** | `VERIFIERS_V1_MIGRATION_AUDIT.md`, `VERIFIERS_V1_MIGRATION_REPORT.md`, `VERIFIERS_VERSION_SNAPSHOT.md`, `environments/transformer_repair/` |
| **machine check** | import `verifiers.v1`, load the taskset, enumerate tasks, run a dry-run; assert zero `import verifiers as vf` / `vf.Environment` / `SingleTurnEnv` / `Rubric` references on the public path |
| **known constraint** | `verifiers.v1` imports `fcntl` and **cannot load on Windows**. Plan B (Linux container) is mandatory, not optional |
| **FAIL if** | evaluation still runs through a v0 path with a v1 wrapper that is not actually exercised |

## G2 — Hardened verifier v2 + historical replay

| | |
|---|---|
| **PASS iff** | the confirmed return-type/API-contract hole is closed; gold passes 100 %; buggy fails 100 %; no unexplained new rejection of a historical FULL_FIX; every v2↔independent-oracle disagreement explained individually; replay coverage reported; and the hole is **not** closed by rejecting more legitimate fixes |
| **artifact** | `VERIFIER_V2_PROTOCOL.md` (hashed before code changes), `VERIFIER_V2_REPLAY.csv` |
| **machine check** | replay ≥90 % of the 50 historical multi-turn trajectories through v1, v2 and the independent oracle; assert `v2_FPR ≤ v1_FPR` **and** `v2_FNR == 0` on FULL_FIX |
| **FAIL if** | coverage < 90 % without replacement trajectories labelled `REPLACEMENT_NOT_HISTORICAL`, or FPR is reduced by blanket rejection |

## G3 — Turn-budget ablation

| | |
|---|---|
| **PASS iff** | the 14-turn confound is *quantified*: a pre-registered 24-turn arm on a frozen Tier H subset, compared against frozen 14-turn controls, with the difficulty interpretation updated to `BUDGET-SENSITIVE` or `BUDGET-ROBUST WITHIN TESTED RANGE` |
| **artifact** | `TURN_BUDGET_ABLATION_PROTOCOL.md` (frozen first), `TURN_BUDGET_ABLATION_REPORT.md`, `artifacts/tier_h_24turn.jsonl` |
| **machine check** | assert the artifact contains both budgets, ≥1 shared task, and a stated verdict token |
| **note** | **no result direction is required.** PASS is about measurement, not about 24 turns helping |
| **budget** | ≤10 new trajectories |

## G4 — Tier S: repo-scale localization

| | |
|---|---|
| **PASS iff** | 3 tasks, each 20–50 plausible files with real import relationships and 1–3 truly relevant files; gold PASS / no-op FAIL on all 3; source audit complete; localization metrics computable; a real multi-turn evaluation run; and the agent demonstrably **cannot** exhaustively read the repo inside budget |
| **artifact** | `LOCALIZATION_SCALE_REPORT.md`, `artifacts/tier_s_primary.jsonl`, `TIER_S_REAL_MODEL_AUDIT.csv` |
| **machine check** | file count per task in [20,50]; relevant files ≤3; every non-relevant file imported or referenced somewhere; `fraction_repo_inspected < 1.0` for every episode |
| **forbidden** | `dummy_001.py` padding; redesign after seeing results |
| **budget** | 12 new trajectories, one frozen version only |

## G5 — Grader scalability

| | |
|---|---|
| **PASS iff** | either **(A)** a safe persistent/batched grader measurably reduces startup overhead with a 100 %-passing isolation suite, or **(B)** benchmark data shows the official v1 runtime already provides acceptable warm execution and the custom optimization is deleted |
| **artifact** | `VERIFIER_SCALABILITY_REPORT.md`, isolation canary tests |
| **machine check** | ≥30 sequential grading jobs benchmarked; mean/p50/p95 recorded for cold and final path; all state-leak canaries (file, env, module global, child process, temp dir, grader secret) pass |
| **FAIL if** | deferred to future work, or isolation traded for speed |

## G6 — Heuristic gate false-positive audit

| | |
|---|---|
| **PASS iff** | ≥20 genuinely benign human-written ML code/patch snippets from license-friendly sources are run through the heuristic gates; the FP rate is measured; and hard structural gates are separated from soft telemetry signals, with the exploit suite re-run to prove no protection was lost |
| **artifact** | `HEURISTIC_GATE_FP_AUDIT.md` |
| **machine check** | ≥20 samples with source/commit/license recorded; gate decisions tabulated; exploit suite still catches 7/7 |

## G7 — Packaging, CI, fresh-clone reproduction

| | |
|---|---|
| **PASS iff** | a **fresh clone in a clean environment** reproduces the GPU-free core from the README verbatim, with a source-isolation check proving imports come from the fresh tree and not `E:\RL` |
| **artifact** | `REPRODUCIBILITY.md`, `FINAL_TEST_RUN.log`, `.github/workflows/ci.yml`, `pyproject.toml`, `uv.lock`, `LICENSE`, `CITATION.cff`, `SECURITY_MODEL.md`, `LIMITATIONS.md` |
| **machine check** | fresh-clone log exists with exit 0; recorded `module.__file__` does not contain the original working tree; CI command list executed locally |
| **FAIL if** | tested commands differ from README commands |

## G8 — Final scientific synthesis

| | |
|---|---|
| **PASS iff** | `FINAL_FLAGSHIP_RESEARCH_REPORT.md` answers RQ1–RQ6 with **script-generated** numbers only, and every headline metric matches a regenerated summary |
| **machine check** | metric hash in the report equals the hash of the freshly generated summary |
| **forbidden claims** | SOTA, statistical significance, policy improvement, RL improvement, generalization from training |

## G9 — Post-success red-team audit *(mandatory even if G0–G8 all pass)*

| stage | PASS iff |
|---|---|
| **A** independent code review | a reviewer given only the contract and the repo — told to assume the implementation is wrong — finds no gate that should fail. `POST_SUCCESS_CODE_REVIEW.md` |
| **B** result integrity | every reported number recomputed from raw artifacts matches the report. `POST_SUCCESS_RESULT_AUDIT.json` |
| **C** spot checks | 3 successful + 3 failed trajectories, 3 fuzz exploits, 3 gold, 3 no-op read by hand end to end. `POST_SUCCESS_SPOTCHECK.md` |
| **D** mutant testing | ≥5 deliberate breakages (disable hidden oracle, expose grading file, corrupt return contract, reintroduce causal-mask bug, create cross-job state leak) each cause the corresponding test to **FAIL**; if tests stay green they do not protect real behaviour and must be fixed. `POST_SUCCESS_MUTATION_TEST.md` |
| **E** clean-room final run | `scripts/final_acceptance.py` passes in a fresh clone with no reuse of the original venv, PYTHONPATH, or Docker temp state |

If G9 finds a defect: the affected gate flips to FAIL, the defect is fixed, and
**the whole of G9 is re-run.** Declaring success once does not close it.

---

## Failure taxonomy (drives fallbacks, never lowers the bar)

| type | meaning | response |
|---|---|---|
| **A** implementation bug | parser, path, import, fixture, race | fix root cause, regression, retry — max 3 distinct root causes |
| **B** platform incompatibility | Windows POSIX, Docker mount, shell | Plan B: WSL2/Linux. Plan C: Docker dev container. Do **not** keep patching Windows internals |
| **C** architecture approach failure | e.g. persistent worker cannot be isolated safely | Plan B: batched. Plan C: official runtime. **Criterion unchanged** |
| **D** transient API failure | timeout, 429, 5xx | exponential backoff, ≤5 retries, do offline gates meanwhile, retry after |
| **E** permanent provider failure | model retired, no tool-calling | Plan B: another available model on the same provider, recorded as a model change, never silently mixed with historical data |
| **F** research hypothesis fails | 24 turns still fail; Tier S all 0/4 | **not an engineering failure.** Keep the negative result. Never tune tasks, prompts, sample selection or success definitions to fix it |

Every Plan A failure and the fallback taken is recorded in
`docs/history/FALLBACK_EXECUTION_LOG.md`. "Needs user input" is not a permitted response
except for the four blockers below.

## The only permitted stopping blockers

1. DeepSeek credential actually invalid/revoked (offline gates still completed first).
2. Provider unavailable after ≤5 retries plus all offline work.
3. Docker engine unusable on Windows **and** WSL2 **and** container fallback.
4. Continuing would require payment, a new service, a GPU, or a public push.

## Budget

≤ **30** new DeepSeek trajectories total (G3 ≤10, G4 = 12, remainder reserve).
Stop paid calls before an estimated **$5**. The 50 historical trajectories do not
count.

## Scope freeze

No CUDA, Triton, Kubernetes, RAG, vector DB, multi-agent architecture, second
benchmark, new provider, SFT or RL training. This round is
**close → harden → scale realistically → reproduce → audit.**
