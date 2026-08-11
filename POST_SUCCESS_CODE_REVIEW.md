# POST_SUCCESS_CODE_REVIEW

G9 stage A. The brief: assume the implementation is wrong, read the contract and the
repository, and find any gate that should fail but does not.

This is written against the code and artifacts as they stand, not against the narrative in
the reports. Where a finding does not flip a gate, that is stated explicitly rather than
implied, and where a headline is weaker than it sounds, the weaker reading is recorded.

---

## Findings that change what may be claimed

### A1. Verifier v2 is unexercised on the replay population — G2 PASSES anyway

`artifacts/verifier_v2_replay.json` reports:

```
v1_FP_rate            0.2778
v2_FP_rate            0.2778
v1_v2_disagreements   0
contract_only_rejections 0
```

**v2 never once decided differently from v1 across 89 replayed trajectories.** The gate's
machine check is `v2_FPR <= v1_FPR` and `v2_FNR == 0`, and "v2 behaves exactly like v1"
satisfies both trivially. G2 therefore legitimately PASSES under the frozen contract — the
criterion was fixed in advance and is met — but the gate does **not** demonstrate that the
hardened verifier hardens anything on this data.

Consequence, and it is binding on G8: no report may say v2 reduced false positives,
improved precision, or caught anything v1 missed. The honest statement is that v2 adds
contract-level checks which **this replay population never triggers**, and that its
false-positive rate against the independent oracle remains 27.78 %.

Not a gate failure. A claim restriction.

### A2. 19 of 23 predicates are forgeable — already documented, restated here because it is the largest caveat

`predicates.FORGEABLE` lists every check whose ground truth is internal to the candidate.
A submission that fabricates its own gradient norms passes `repo_gradients_reach_optimizer`.
R16 moved the *decision* out of the candidate process; it did not make an untrusted
process's self-report true.

Only four checks are unforgeable, because only they compare against gold:
`repo_matches_gold_logits`, `repo_supervised_token_count`, `repo_lr_schedule_matches_gold`,
`repo_contract_public_api` (which never asks the candidate anything — the trusted side
reads the submitted sources and compares ASTs).

`tests/test_check_surfaces_agree.py::test_gold_dependent_predicates_are_declared_unforgeable`
fails if a gold-backed check is ever mislabelled, or if a new predicate is added without
being classified either way.

### A3. Two parallel check surfaces exist — now tested, previously not

R16 left `trgym/repo/checks.py` (in-process; gold preflight, `fallback=True`, most of the
host suite) and `trgym/repo/predicates.py` (the trusted comparator; **what production
actually grades with**). Nothing forced them to agree.

Found while building G9 stage D: a mutation aimed at the return-type contract in
`predicates.py` would have SURVIVED, because `tests/test_repo_tasks.py` drives
`run_repo_checks` and never reaches the sandboxed predicate. That is the R11 shape exactly
— a suite that is green about code it does not execute.

Closed by `tests/test_check_surfaces_agree.py` (28 tests) and by splitting the mutation
case in two, so both surfaces are broken and checked independently.

### A4. Tier S weakens `repo_contract_public_api` relative to Tier M

The check compares AST signatures of `tinygpt/*.py` — top level only. In Tier S those eight
modules are thin facades, so most public symbols are re-exported rather than defined, and
the comparison covers the facade functions but not the ~40 modules beneath. Gold and
candidate are both built from the same template, so the check is *consistent*, not wrong —
but it inspects a smaller surface than the file count suggests.

Not a gate failure: the contract's G4 machine checks are file count, relevant-file count,
reference connectivity and `fraction_repo_inspected`, none of which depend on this. Worth
knowing before anyone cites Tier S as evidence about API contracts.

---

## Gates examined and found sound

**G0.** `FROZEN_PHASE1_MANIFEST.json` records `verdict PASS, n_missing 0, n_unparseable 0`.
The one destroyed artifact (`verifier_fuzz_audit.json`, R7) is recorded as
`LOST_OVERWRITTEN` rather than reconciled by editing the expectation — the honest handling.
Independently confirmed: `scripts/build_final_metrics_summary.py` recomputes tier_m,
tier_h and tier_m_pro from raw trajectories without consulting the manifest, and
`manifest_crosscheck.agrees == true`.

**G1.** Static assertions in `final_acceptance.py` now match the R16 architecture, and two
of them were themselves wrong on first run — they searched raw text for `/grader` and
`pickle` and matched the comments *documenting the fix*. Both read the AST now. The
runtime half rejects `v1_runtime_evidence.json` whose `grading_sha256` no longer matches
`grading.py`, so stale evidence cannot satisfy it.

**G3.** Measurement-only gate; no result direction required. The 24-turn arm exists at
`artifacts/tier_h_24turn.jsonl` with 10 rows.

**G5.** The strongest evidence in the project: 0/17 canary leaks with **17/17 probes
executing and 0 refused**, against an unsafe control that still leaks 12/17. Three earlier
versions of this suite reported clean containment while measuring nothing, and the artifact
now carries `contained_by` per probe plus `channels_without_an_executing_probe` so a vacuous
run cannot pass.

**G6.** 0/32 false positives at submission level on benign human-written ML code.

**G7.** The fresh-clone check interrogates the clone's own interpreter for each module's
`__file__` and fails if any resolves back into `E:\RL`, and strips `PYTHONPATH`,
`PYTHONHOME` and `VIRTUAL_ENV` from the child environment. That is the failure mode a naive
fresh-clone test misses.

---

## Attempted attacks that did not work

| attack | result |
|---|---|
| read the hidden checks at `/grader/trgym/repo/checks.py` | contained — no repository is mounted into the candidate container (R16) |
| `gc.get_objects()` walk to `gold_repo()` | contained — the module is not in that process |
| forge `<<<TRGYM_OBS>>>` on stdout | rejected — marker is nonced per job and the last authenticated block wins |
| assemble the marker at runtime to evade a literal scan | rejected — the nonce, not the literal, is the secret |
| oversized / malformed observation document | rejected before allocation; 31 tests in `tests/test_obs_protocol.py` |
| declare `shape: [1e6, 1e6]` to exhaust trusted-process memory | rejected on arithmetic before any buffer is touched |
| mutate `ATOL` on the grader module from candidate code | contained — the grader module is not in the candidate's process |

---

## Verdict

No gate should fail that currently passes. Three claim restrictions (A1, A2, A4) and one
structural risk that has been closed (A3). A1 is the one that matters for the write-up:
**G2 must not be described as an improvement over v1**, because on the replayed population
it made no different decision even once.
