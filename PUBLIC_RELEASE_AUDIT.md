# PUBLIC_RELEASE_AUDIT

Pre-publication audit of the exact tree being pushed. **Status: PASS.**

No secret value, token, or personal address is reproduced anywhere in this document.

---

## 1. Secret scan — PASS

Content search across the full working tree.

| category | result |
|---|---|
| provider API keys (`sk-…`, OpenAI / DeepSeek / Anthropic shape) | none |
| GitHub tokens (`ghp_`, `gho_`, `github_pat_`) | none |
| AWS access keys (`AKIA…`) | none |
| private key material (`BEGIN … PRIVATE KEY`) | none |
| `Authorization: Bearer …` headers | none |
| `.env` / `.env.*` files | **none exist anywhere in the tree** |

Five key-shaped matches were inspected individually and are all **documentation
placeholders** — `"sk-..."`, `"<key>"`, `"<rotated key>"` — in
`PHASE_0_FEASIBILITY_REPORT.md`, `PHASE_1_GPU_FREE_REPORT.md`,
`scripts/run_deepseek_repo_eval.py` and `scripts/run_api_baseline.py`.

By design nothing in this project reads a credential from a file. `DEEPSEEK_API_KEY` is
taken from the shell environment only, and the runner exits with an instruction rather than
falling back to any file. The optional cross-model check verified provider variables by
**name only**; no value was read, printed, or logged (`artifacts/cross_model_smoke.json`).

## 2. Personal data — RESOLVED

**File contents:** clean. No email address, phone, or postal address in any tracked file.

**Commit metadata:** the repository was initialised with a university email, so every
commit's author and committer line carried it. `.gitignore` cannot remove that, and commit
metadata is fully public on a public repository.

Fixed before any push, while the history was still unpublished:

- `user.email` / `user.name` set to the GitHub noreply identity for the authenticated
  account (`lordrance`).
- All six existing commits rewritten via `git filter-branch --env-filter` so both author
  and committer fields carry the noreply address.
- Backup refs under `refs/original/` deleted, reflogs expired, `git gc --prune=now` run, so
  the old objects are unreachable and not transferred by a clone.
- Verified: zero occurrences of the university address across `git log --all` author,
  committer, and name fields.

This document deliberately describes the address rather than quoting it — an audit file that
prints the value it just scrubbed would republish it.

## 3. Machine-specific paths — accepted, low severity

Several artifacts contain Windows temp paths recorded as run provenance
(`artifacts/fresh_clone_run.json`, the trajectory `.jsonl` files, `artifacts/final_acceptance.json`).
These are provenance, not secrets, and reported numbers depend on the artifacts containing
them. `.claude/` — the one genuinely unwanted machine-specific entry — is untracked and
ignored.

## 4. Files excluded from the repository

`.gitignore` covers virtualenvs, byte-code and tool caches, `.env*` and credential files,
agent/IDE state (`.claude/`, `.cursor/`, `.idea/`, `.vscode/`), model caches, scratch
workspaces, disposable clones, and `outputs/`.

Already-tracked offenders were removed from the index with `git rm --cached` (they remain on
disk locally):

| path | reason |
|---|---|
| `.claude/settings.json`, `.claude/settings.local.json` | agent session state, machine paths |
| `outputs/transformer_repair--validate/…` | disposable CLI run output |

**Deliberately kept:** everything under `artifacts/` that a reported number depends on, and
`artifacts/frozen_phase1/`, which G0 hashes. `artifacts/g5_isolation_canaries.PRE_R15_FIX.json`
is retained because it substantiates the R15 narrative in `PROTOCOL_CHANGELOG.md`; it is
evidence, not clutter.

## 5. Licence — PASS

MIT, `LICENSE` at the repository root, copyright Rance. `pyproject.toml` declares
`license = { file = "LICENSE" }` and the MIT classifier. `CITATION.cff` declares `license:
MIT`. Dependencies are declared, not bundled.

**A correction.** An earlier version of this section asserted that no third-party source
was redistributed. That was false when written: `artifacts/raw/v1_study/src_*.py` held ten
files of verbatim `verifiers` library source, dumped during the G1 migration to answer API
questions from the real implementation rather than by guessing. Publishing another
project's code in an MIT repository without its licence is a genuine problem, however small
the excerpt.

Those files are removed. Nothing depended on them: `VERIFIERS_VERSION_SNAPSHOT.md` already
documents `docker/study_v1.py` as the way to regenerate them, and no gate reads them. The
derived notes that *are* this project's own work — `v1_study.json` and `v1_tree.txt` —
remain, because the G1 findings cite them.

## 6. Large files — PASS

Nothing approaching GitHub's limits. Largest tracked items are trajectory `.jsonl` artifacts
(≈0.5–0.9 MB each) and the total `artifacts/` tree at ≈6 MB. No model weights, no `.pt`
checkpoints, no Docker layers, no virtualenv.

## 7. Verification state at the time of this audit

```
acceptance          G0-G9 PASS, FLAGSHIP_GPU_FREE_COMPLETE = YES
host suite          258 passed, 16 skipped
tests_v1 (Linux)    16 passed
isolation canaries  0/17 leaked, 17/17 executed; unsafe control 12/17
Tier S freeze       frozen=True, gold PASS / no-op FAIL on all three tasks
mutation suite      8/8 RED, 0 survivors, all restored
result audit        45 values recomputed independently, 0 mismatches
```

## 8. CI, and one observed flake

All five jobs are green on the release commit: host suite on {ubuntu, windows} ×
{py3.11, py3.12}, and a sandboxed job that builds both images and runs the isolation
canaries, `tests_v1` through the official v1 runtime, the Inspect smoke in both
directions, the adversarial verifier replay, and the acceptance gate.

**An intermittent CI failure turned out to be a real grading defect (R18).** It is recorded
here in full, including the fact that the first diagnosis was wrong.

`tests/test_harness.py::test_a_policy_that_fixes_the_bug_passes_the_hidden_suite` failed on
windows/py3.11, passed on a rerun of the identical commit, and was written up as a flake.
It then failed again on **both** ubuntu jobs. Three occurrences across platforms is not a
flake, and the "rerun went green" evidence had been given more weight than it deserved.

Root cause: `RepoModules` cleared `sys.modules` and called `importlib.invalidate_caches()`
before importing a candidate tree, which resets *finder* caches but does not decide whether
a cached `__pycache__/*.pyc` is reused. Python decides that by comparing the source's
(mtime, size) against the pair in the `.pyc` header. The repair in these tasks — for
example `tril(diagonal=1)` → `tril(diagonal=0)` — leaves the file **exactly the same
size**, so if it lands in the same mtime second as the preceding write, the old bytecode
runs and the patch is invisible.

The consequence is worse than a flaky test: **a correct repair could be graded as still
broken.** A false negative against a working submission, in the grading path itself. The
telling detail was that `repo_strict_causality`, which never consults gold, was among the
failures — the candidate's own code was not what executed.

Fixed by purging `__pycache__` under the tree on entry, in both `RepoModules`
implementations (`trgym/repo/visible_runtime.py` and the probe's copy inside the candidate
container). `tests/test_bytecode_staleness.py` forces the collision deterministically —
pinning mtime and asserting a same-size repair is graded as fixed — so a regression fails
every time rather than once in a while.

## 9. Release checklist

- [x] Secret scan clean
- [x] No `.env` present
- [x] No personal data in file contents
- [x] Commit-metadata email rewritten to noreply and verified absent from history
- [x] `.gitignore` hardened; `.claude/` and `outputs/` untracked
- [x] Licence present and declared
- [x] No oversized or binary junk
- [x] Acceptance re-verified after v0.2 enhancements
- [x] Repository URLs point at the authenticated account
- [x] Remote created (private) and pushed
- [x] Remote tree verified: 0 ignored paths, 0 personal addresses in commit metadata
- [x] CI green on all five jobs (ubuntu + windows × py3.11/3.12, plus sandboxed)
- [x] Remote clone verified: cloned from GitHub, release tree present, no `.venv`,
      `.claude`, `outputs` or `.env`
- [x] Licence detected as MIT by GitHub's own endpoint
- [x] `v0.1.0` (pre-enhancement baseline) and `v0.2.0` tags pushed; release published
- [x] Visibility flipped to **public** only after the above

### One limitation of this release's verification

The clean-room *execution* check (`uv sync` + full suite inside a clone) was last run
locally before v0.2 and is recorded in `artifacts/fresh_clone_run.json`. For v0.2 itself the
equivalent evidence is CI: five jobs that start from `actions/checkout` of the pushed
commit on machines that have never seen the development tree, install from the committed
lockfile, and run the host suite, the isolation canaries, `tests_v1` through the official v1
runtime, the Inspect smoke and the acceptance gate. That is a stronger independence
guarantee than a local clone, but it is not the same artifact, and the distinction is worth
stating rather than blurring.
