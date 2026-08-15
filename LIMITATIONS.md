# LIMITATIONS

What this project does **not** establish. Written so that a reader who only skims the
reports still cannot mistake its scope. Every entry below is sourced from an artifact in
this repository, not from an impression.

Required by `FINAL_FLAGSHIP_COMPLETION_CONTRACT.md` (G7).

---

## 1. No training. No RL. No policy improvement.

Nothing here trains a model. There is no GPU run, no GRPO, no LoRA, no SFT. `minimal_grpo/`
is a CPU demonstration of the algorithm's mechanics on a toy problem; it is not an experiment
and its output supports no claim about this environment.

Consequently, **every number in this project describes an off-the-shelf model's behaviour on
a fixed benchmark.** No claim of policy improvement, RL improvement, or generalization from
training is made or supported anywhere.

## 2. One model family, one provider, small n

All real-model measurements come from DeepSeek (`deepseek-chat`, plus one `flash` arm in
Phase 0.5). Trajectory counts are in the tens, not thousands. That is enough to characterise
*this* environment's difficulty and to catch verifier defects; it is not enough for
statistical significance, and none is claimed. Differences between arms should be read as
*directional observations on small samples*.

## 3. Difficulty labels are budget-sensitive, and one historical label was wrong

G3 established the verdict **`BUDGET-SENSITIVE`**: at a 14-turn budget, voluntary submission
was 0/20 and budget exhaustion 20/20; at 24 turns that inverted. A Phase 1 difficulty label
was corrected as a direct result.

Any difficulty statement in this repo is therefore conditional on the turn budget it was
measured at, and the two budgets tested (14, 24) do not span the interesting range.

## 4. Some historical measurements are protocol-invalid and are kept, not deleted

The `h2` result was produced by an adapter that entered an empty-response / non-action
failure loop, so its episodes measure the adapter rather than the task. It is labelled
protocol-invalid and retained; it must not be read as a capability result, and it has **not
yet been re-measured** under the repaired adapter.

The general rule this project follows: an invalid measurement is labelled and kept, never
quietly replaced by a better-looking one.

## 5. The security model is a research boundary, not a production one

`SECURITY_MODEL.md` is the authoritative statement; the short version:

- the threat modelled is **a policy optimising against a reward**, not a determined human
  adversary with a kernel exploit;
- **in-process verdict integrity is mitigated, not guaranteed** (R15). Grading must execute
  the code it grades, so candidate code shares a process with the checks inside the grading
  container. Two layers make tampering expensive — a static refusal before execution and a
  post-run integrity check with a nonced result protocol — and neither is a boundary;
- docker-out-of-docker, used for CI convenience on Windows, mounts the host socket and hands
  effective host control to that process. It is not a posture to reproduce for untrusted
  code;
- no side-channel analysis, no supply-chain verification, no penetration testing.

## 6. The verifier is hardened against observed exploits, not against unknown ones

The v2 verifier closes one confirmed contract hole and is exercised by a 16-probe fuzz suite
and a 7-cheat exploit suite. `v2_FP_rate == v1_FP_rate` — v2 rejects strictly more *invalid*
submissions without rejecting any additional legitimate fix — and `v2_FN_count == 0` on
`FULL_FIX`. That is evidence of no regression on the population that was replayed; it is not
a claim of completeness against exploits nobody has constructed yet.

`v2_vs_oracle_disagreements = 20` on the replay: these are individually explained in
`VERIFIER_V2_REPLAY.csv` and `VERIFIER_QUALITY_MATRIX.md`, and they are the honest measure of
how much interpretation still sits between "the checks passed" and "the bug is fixed".

## 7. The heuristic gates were measured on 32 samples

G6's false-positive audit uses 32 licensed, human-written ML functions. The submission-level
FP rate went 15.6 % → 0.0 % after R13 (matching tokenized code rather than prose). 32 samples
cannot establish a low FP *rate*; it establishes that the specific failure mode that was
observed no longer occurs on that sample, and that the exploit suite still catches 7/7.

## 8. The task suite is small, synthetic in origin, and single-domain

Ten repo-level tasks over one small transformer training codebase, plus the Tier E function
tasks. The defects are planted from real-world bug patterns (documented in
`REAL_BUG_EVIDENCE.md`), but they are planted, not harvested from a project's history. No
claim is made that performance here transfers to a different codebase, language, or defect
distribution.

## 9. Reproduction is verified on one platform pair

Windows host + Docker Desktop (Linux containers). `verifiers.v1` cannot import on Windows at
all (`fcntl`), so every v1 result comes from the Linux image. Nothing has been tested on
macOS, on native Linux without Docker Desktop, or on ARM.

## 10. What was left unfinished

Tracked live in `FINAL_WORKLOG.md`; at the time of writing G1, G4, G5, G7, G8 and G9 are not
closed. `docs/history/FINAL_FLAGSHIP_ACCEPTANCE_REPORT.md` carries the computed gate state, and
`scripts/final_acceptance.py` recomputes it from artifacts rather than from any of this prose.
