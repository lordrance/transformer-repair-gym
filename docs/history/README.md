# Historical record

These documents are kept because the project's claims rest on them, not because they are
current. They were moved out of the repository root so a reader arriving at the top level
sees the results and the design, rather than a wall of forty-odd files.

**Nothing here was deleted.** Every one of these is cited by a document that is still live,
and several record mistakes that later findings depend on. Deleting the evidence of an error
would make the correction unverifiable, which is the opposite of what this project is for.

| file | why it is kept |
|---|---|
| `G1_MIGRATION_DESIGN.md` | The design whose central premise was **wrong**. It argued the v1 migration was a "polarity inversion" requiring a custom harness; reading the installed source showed Task owns setup and grading while Harness owns the rollout, so no custom harness was needed. `VERIFIERS_V1_MIGRATION_REPORT.md` cites it as the thing evidence overturned. |
| `FALLBACK_EXECUTION_LOG.md` | Every Plan-A failure and the fallback taken, required by the completion contract. Also the record of the sessions where command execution was blocked and work stopped rather than being faked. |
| `FINAL_FLAGSHIP_ACCEPTANCE_REPORT.md` | The computed gate state at flagship completion, cited by `LIMITATIONS.md`. |
| `PHASE_0_5_REAL_MODEL_REPORT.md` | First real-model measurements; the source of the difficulty calibration that reshaped the task set. |
| `PHASE_1_GPU_FREE_REPORT.md` | Phase 1 write-up, cited by the release audit for its credential-handling instructions. |
| `DIFFICULTY_CALIBRATION.md` | Per-task difficulty analysis and the redesign options considered. |
| `REWARD_BASELINE_V2.md` | The rebuilt naive verifier in detail — the "weak grader" the headline comparison is measured against. Without it, "naive" is an unexamined word. |
| `TRAJECTORY_EFFICIENCY.md` | RQ4 evidence, cited by `PHASE_1_FINAL_RESEARCH_REPORT.md` and the literature guardrails. Source of the finding that trajectory shape carries no signal about outcome. |

Start instead with [`../../README.md`](../../README.md), [`../../RESULTS_SUMMARY.md`](../../RESULTS_SUMMARY.md),
and [`../../FINAL_FLAGSHIP_RESEARCH_REPORT.md`](../../FINAL_FLAGSHIP_RESEARCH_REPORT.md).
