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
MIT`. No vendored third-party source is redistributed; dependencies are declared, not
bundled.

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

**Recorded rather than glossed over:** on one run,
`tests/test_harness.py::test_a_policy_that_fixes_the_bug_passes_the_hidden_suite` failed on
windows/py3.11 only — `repo_strict_causality` and `repo_matches_gold_logits` still failing
after a patch that had applied successfully — while the same commit passed on the other
three matrix entries and locally. A rerun of the identical commit passed.

So it is non-deterministic, not a regression, and its frequency is unmeasured (one
occurrence). It is a *test* flake, not a grading-path flake: the isolation canaries, the
Tier S freeze, and the gold/no-op separation were all stable across every run. Still, a
grading-adjacent test that is not reproducible is a real if minor defect, and chasing it
belongs in a follow-up rather than in a release commit. Do not treat a single green run of
this test as proof it is deterministic.

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
- [x] CI green on all five jobs
- [ ] Remote fresh-clone reproduction
- [ ] Visibility flipped to public
