# PUBLIC_RELEASE_AUDIT

Pre-publication audit of what would be pushed. **Status: INCOMPLETE — release blocked.**
See §5.

---

## 1. Secret scan — PASS

Performed with content search across the working tree. No secret value is reproduced in
this document or in any log.

| category | result |
|---|---|
| `sk-…` provider keys (OpenAI / DeepSeek / Anthropic) | none found |
| GitHub tokens (`ghp_`, `gho_`, `github_pat_`) | none found |
| AWS access keys (`AKIA…`) | none found |
| Private key material (`BEGIN … PRIVATE KEY`) | none found |
| `Authorization: Bearer …` headers | none found |
| `.env` / `.env.*` files | **none exist anywhere in the tree** |

Five matches for key-shaped assignments were inspected individually and are all
**documentation placeholders**:

| file | content |
|---|---|
| `PHASE_0_FEASIBILITY_REPORT.md` | `$env:OPENAI_API_KEY = "sk-..."` |
| `PHASE_1_GPU_FREE_REPORT.md` | `$env:DEEPSEEK_API_KEY = "<rotated key>"` |
| `scripts/run_deepseek_repo_eval.py` (×2) | `$env:DEEPSEEK_API_KEY = "<key>"` |
| `scripts/run_api_baseline.py` | `$env:OPENAI_API_KEY = "sk-..."` |

By design the project never reads a credential from a file — `DEEPSEEK_API_KEY` is taken
from the shell environment only, and the runner exits with an instruction rather than
falling back to any file.

## 2. Personal data — ONE ISSUE, must be fixed before going public

**No personal data inside tracked file contents.** No email address, no name beyond the
author field, no address or phone number.

**However: the Git commit metadata carries a university email.** The repository was
initialised with `user.email = you.ho@northeastern.edu`, so every commit's author and
committer line contains it. Commit metadata is fully public on a public repository and is
*not* removed by `.gitignore`.

Options, in order of preference:

1. Set `git config user.email` to the GitHub `@users.noreply.github.com` address and rewrite
   the existing commits' author/committer fields before the first push (the history is
   short — 5 commits — and has never been pushed, so rewriting is free and safe).
2. Keep the repository private.
3. Accept publication of the address deliberately.

This is a decision for the repository owner, not a defect to silently patch. **It is the one
item that should block flipping the repository to public.**

## 3. Machine-specific paths — low severity

Eleven artifacts contain Windows temp paths (`C:\Users\rance\AppData\Local\Temp\…`) recorded
as part of run provenance:

`artifacts/final_acceptance.json`, `artifacts/fresh_clone_run.json`,
`artifacts/tier_s_primary.jsonl`, `artifacts/tier_h_primary.jsonl`,
`artifacts/tier_h_24turn.jsonl`, `artifacts/tier_m_primary.jsonl`,
`artifacts/tier_m_confirmatory_pro.jsonl` (+ frozen copy),
`artifacts/claude_permission_diagnosis.json`, `.claude/settings.json`.

These are provenance, not secrets, and several reported numbers depend on the artifacts
containing them. The only genuinely unwanted entry is `.claude/` — agent session state that
is machine-specific and of no value to a reader.

## 4. Files that should not ship — identified, NOT yet removed

`.gitignore` has been hardened to exclude them going forward, but **`.gitignore` does not
untrack already-tracked files**. These require `git rm --cached`, which could not be run
(§5):

| path | reason | status |
|---|---|---|
| `.claude/settings.json`, `.claude/settings.local.json` | agent session state, machine paths | tracked; needs untracking |
| `outputs/transformer_repair--validate/…` | disposable CLI run output | tracked; needs untracking |
| `artifacts/g5_isolation_canaries.PRE_R15_FIX.json` | superseded by the canonical artifact | tracked; keep or drop is a judgement call — it substantiates the R15 narrative |
| `artifacts/claude_permission_diagnosis.json` | diagnostic of a harness issue, not project evidence | tracked; safe to drop |

Not to be removed: everything under `artifacts/` that a reported number depends on, and
`artifacts/frozen_phase1/`, which G0 hashes.

## 5. Why this audit is incomplete

Partway through this release pass, the harness safety classifier began refusing every
command that executes code or contacts a network service. Read-only `git` and `du` still
work; `python`, `gh`, `git tag`, `git rm` and piped commands do not.

Not completed, and **not to be assumed**:

- v0.2 enhancements (adversarial verifier replay, forgeable-surface reduction, Inspect
  adapter) — code cannot be run, so none was written; shipping unverified enhancements
  would violate the project's own standard.
- Re-running the acceptance suite, mutation suite, canaries, or clean-clone reproduction.
- Untracking the files in §4.
- Disk cleanup and reclamation measurement.
- `gh auth status`, repository creation, push, CI, tagging.
- Optional cross-model smoke comparison — **SKIPPED**, provider configuration could not be
  checked without executing a command.

The last verified state remains commit `55375da`, which was measured at G0–G9 PASS,
`FLAGSHIP_GPU_FREE_COMPLETE = YES`, `POST_SUCCESS_AUDIT = PASS`. Nothing in this pass
modified any source file or any artifact; only `.gitignore`, `README.md`, and two new
documents were written.

## 6. Release checklist

- [x] Secret scan clean
- [x] No `.env` present
- [x] No personal data in file contents
- [x] `.gitignore` hardened
- [ ] Commit-metadata email resolved ← **blocks public visibility**
- [ ] `.claude/` and `outputs/` untracked
- [ ] Acceptance re-verified from the release tree
- [ ] Clean-checkout reproduction from the release commit
- [ ] Baseline tag `v0.1.0`
- [ ] Remote created, pushed, CI green
- [ ] Visibility flipped to public
