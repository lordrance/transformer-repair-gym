# POST_SUCCESS_MUTATION_TEST

G9 stage D. Break each defence deliberately and require the corresponding check to go RED.

A green suite tells you the code does what the tests check. It does not tell you the tests
check anything. This project has shipped that failure twice — R11's verifier scored gold and
a planted bug identically, and R15's denylist refused every canary *before it executed*, so
"contained" meant "never happened". Mutation testing is the difference, and a **surviving
mutant is the finding**: it means the defence is not tested and the gate resting on it is
not earned.

Artifact: `artifacts/post_success_mutation_test.json`. Harness:
`scripts/post_success_mutation_test.py` (`--dry-run` verifies every anchor resolves exactly
once before anything is touched).

---

## Result

**8 cases. 0 survivors. All restored. Post-restore suite green.**

| mutation | file | disabled defence | verified by | RED | time |
|---|---|---|---|---|---|
| `hidden_oracle_protection` | `harness/sandbox.py` | candidate container holds no oracle (R16) | canary `oracle_source_read_in_container` | ✅ | 907 s |
| `trusted_comparator_boundary` | `repo/predicates.py` | comparator never imports candidate code (R16) | `tests/test_grading_isolation.py` | ✅ | 21 s |
| `verdict_protocol_authentication` | `repo/obs_protocol.py` | observation block nonced, last-wins (R15) | `tests/test_obs_protocol.py` | ✅ | 4 s |
| `grading_artifact_isolation` | `transformer_repair/grading.py` | grading never runs candidate code beside gold (R14) | `tests/test_grading_isolation.py` | ✅ | 21 s |
| `return_type_contract` | `repo/checks.py` | `accumulate_gradients` returns exactly `float` | `tests/test_contract_checks.py` | ✅ | 6 s |
| `sandboxed_predicate_enforcement` | `repo/predicates.py` | production predicates actually enforce (R16) | Tier S freeze `noop_fails` | ✅ | 47 s |
| `causal_mask` | `repo_template/tinygpt/attention.py` | reference implementation is causally correct | `tests/test_repo_tasks.py` | ✅ | 21 s |
| `cross_job_state_isolation` | `harness/sandbox.py` | no job sees another job's temp state | canary `temp_dir_persistence` | ✅ | 703 s |

```
survived                      []
all_restored                  True
post_restore_tests_pass       True    (248 passed, 16 skipped)
canaries regenerated clean    n_leaked = 0
freeze regenerated clean      frozen  = True
```

All five contract-required mutations are present (`hidden_oracle_protection`,
`grading_artifact_isolation`, `return_type_contract`, `causal_mask`,
`cross_job_state_isolation`), plus the two PROTOCOL_CHANGELOG additions for R15 and R16 and
one extra covering the production predicate surface.

---

## What the two expensive cases actually proved

**`hidden_oracle_protection`** re-adds `-v {REPO_ROOT}:/grader:ro`. With the repository back
in the candidate container, `oracle_source_read_in_container` leaked — candidate code read
`/grader/trgym/repo/checks.py` — and so did `gold_template_read_in_container` and
`grader_mount_absent`. That is the R16 defect reproduced on demand, and it confirms the
canary suite detects it rather than assuming it.

**`cross_job_state_isolation`** swaps the `/tmp` tmpfs for a shared bind mount. Three probes
leaked: `temp_dir_persistence` (state survived into the next job), `child_process` and
`reference_bundle_unreachable` (both write evidence into `/tmp`, which became host-visible).
One mutation, three independent detectors — the cross-job property is over-determined rather
than resting on a single probe.

---

## Two cases were wrong before they were right

Both were caught by reasoning about what *would* turn red, not by watching a run finish.
Recording them because a mis-aimed mutation and an untested defence produce identical
output — "SURVIVED" — and only one of them is a real finding.

### `trusted_comparator_boundary` — the first version disabled nothing

It originally re-added the `/grader` mount and checked `gold_oracle_import_evasive`. Case 1's
artifact showed that probe returning `leaked=False` under exactly that mutation, so the run
was stopped mid-flight. The reason is structural: mounting the repository does not put
`trgym.repo.checks` into the *probe's* process, and the R16 probe never imports it, so the
object-graph walk finds nothing either way.

The case now mutates what the defence actually is — the trusted comparator opening the
candidate's workspace with `RepoModules`, which is the R14 defect reintroduced from the
other side — and is verified by the static test written for precisely that.

### `return_type_contract` — the check was dead coverage

Aimed initially at `tests/test_repo_tasks.py`. It would have survived, and the reason is the
sharpest finding of this audit: **no Tier E/M/H/S task plants a return-type defect, and no
test called `repo_contract_return_types` or `repo_contract_public_api` directly.** Both
checks were added by verifier v2 to close a confirmed hole (VERIFIER_V2_PROTOCOL.md H1), and
both could have been deleted from the codebase with the whole suite still green.

That is R11's shape one level up. R11 was a check that always passed; this was a check that
was never asked anything, which is harder to see because it produces no suspicious result —
it produces none at all.

The contract's instruction for a green mutation is explicit: *"if tests stay green they do
not protect real behaviour and must be fixed."* So `tests/test_contract_checks.py` was
added — 8 tests exercising both checks against gold and against four separate contract
violations (Tensor-valued `accumulate_gradients`, non-JSON-serialisable history, an added
parameter, a removed public symbol). The mutation now turns them red in 6 seconds.

This also bears on G2: the replay reports `contract_only_rejections: 0`, and there is now a
second independent reason to read that as "the contract checks were never exercised on that
population" rather than as evidence they are permissive.

---

## Restoration

Every case restores the file byte-for-byte in a `finally`, and `restored` is asserted per
case rather than assumed — all 8 report `restored=True`. After the last case the harness
regenerates the two artifacts the mutations overwrite (`g5_isolation_canaries.json` and
`tier_s_spec.json`), because leaving mutant results on disk would hand G4 and G5 a
contaminated artifact reporting leaks and a failed freeze. Both came back clean, and the
full host suite passes.

## Reproduce

```bash
python scripts/post_success_mutation_test.py --dry-run      # anchors only, mutates nothing
python scripts/post_success_mutation_test.py --keep-going   # ~30-40 min; longer under load
```

`--keep-going` continues past a survivor so one failure does not hide the rest. Without it
the run stops at the first, which is the right default when the expectation is zero.
