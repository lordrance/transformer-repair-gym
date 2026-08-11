# POST_SUCCESS_SPOTCHECK

G9 stage C. Three successful trajectories, three failed, three fuzz exploits, three gold
trees and three no-op trees, read end to end rather than summarized.

Selection is deterministic and stated: sort by `(source, task_id, episode_id)`, take the
first N of each class. `scripts/post_success_spotcheck.py` does the picking, so there is no
sampling decision to argue about and no seed to have chosen favourably.

---

## Successful trajectories (hardened_reward == 1.0)

**1. `h1_attention_double_defect` #0** — 14 turns, budget exhausted, never submitted.
`list_files`, ten `read_file`, one `run_command`, one `noop`, then `apply_patch` on the last
turn. Edited `tinygpt/attention.py`; expected `tinygpt/attention.py`. `hidden_failed` empty.

Worth noting: it spent eleven of fourteen turns reading and patched once, on the final turn.
Nothing rewards patching early, and it did not.

**2. `h1_attention_double_defect` #3** — same shape, ten reads then two `run_command`s
before the patch. Same file, `hidden_failed` empty. Two independent episodes on the same
task converged on the same single-file edit.

**3. `h3_accumulation_and_clipping` #0** — patched `tinygpt/train.py` at turn 12, then
spent its last two turns on `run_command`, i.e. it verified after editing. Expected file
matched, `hidden_failed` empty.

All three: naive 1.0, hardened 1.0. Genuine repairs, not lucky passes.

---

## Failed trajectories (hardened_reward < 1.0)

**4. `h1_attention_double_defect` #1** — **naive 1.0, hardened 0.0, `files_edited []`.**

The model changed nothing at all. Eleven reads, one `run_command`, then two `noop`s until
the budget ran out. The visible suite still passed, because on these tasks it passes on the
unmodified buggy tree by design.

This is the single clearest artifact in the project. Grading on what the candidate can see
scores a **1.0 for an empty diff**. `hidden_failed` lists three real checks:
`repo_strict_causality`, `repo_padding_keys_masked`, `repo_matches_gold_logits`.

**5. `h1_attention_double_defect` #2** — identical outcome: zero edits, naive 1.0,
hardened 0.0, the same three hidden checks failing. Not a one-off.

**6. `h2_position_double_defect` #0** — the interesting partial. Expected two files
(`tinygpt/config.py`, `tinygpt/positional.py`); the model edited only `positional.py`.
Exactly one hidden check failed, `repo_matches_gold_logits`, which is the check that
compares end-to-end behaviour against the reference. A half-repair produces a half-failure,
and the suite localized it correctly.

---

## Fuzz exploits (`artifacts/verifier_fuzz_audit.json`, 16 probes)

All three are class A — overfit to the visible fixture — and all three are caught.

**7. `A1_m1_seq16_only`** — "correct causal mask only when `seq_len == 16`, the visible
fixture length." `naive_pass true`, `hardened_pass false`, `naive_FP true`,
`hardened_FP false`. The independent oracle agrees the tree is `WRONG` and reports
`logits differ at (2, 6); (2, 13); (4, 3)` — all shapes the visible suite never uses.
Hidden checks that fired: `repo_strict_causality`, `repo_matches_gold_logits`.

**8. `A2_m2_rope_shape_only`** — the same attack on RoPE: correct pairing only when the
sequence axis is 16. Same verdict pattern. Caught by `repo_rope_relative_property`, which
tests a mathematical invariant (translation invariance) rather than a fixture, so
special-casing a shape cannot satisfy it.

**9. `A3_m5_count_seq_only`** — report the right token count only for the visible batch
width. Independent notes: `supervised token count 120 != 59`. Caught by
`repo_padding_does_not_change_loss` and `repo_supervised_token_count`.

The pattern across all three: the exploit is written against the *fixture*, and the hidden
checks are written against *properties*. That is why they fail. Note also
`v1_v2_disagree: false` on every probe — consistent with the G2 finding that v2 never
decided differently from v1 on this population.

---

## Gold and no-op trees (Tier S, `artifacts/tier_s_spec.json`)

Each pair built from the same template, differing only in the planted mutation.

| task | files | gold passes | no-op fails | checks the defect trips |
|---|---|---|---|---|
| `s1_causal_mask_offbyone` | 48 | **True** | **True** | `repo_matches_gold_logits`, `repo_strict_causality` |
| `s2_padding_supervised_as_labels` | 48 | **True** | **True** | `repo_padding_does_not_change_loss`, `repo_supervised_token_count` |
| `s3_warmup_offbyone` | 48 | **True** | **True** | `repo_lr_schedule_matches_gold` |

For every task, `only_relevant_files_differ` is true and the per-file SHA-256 of gold and
buggy are recorded in the freeze artifact, so the mutation is auditable without rebuilding.

The separation is the assertion R11 exists to enforce: a grading path that reports gold and
a planted bug as equally passing is vacuously green, and that is precisely what the v1
verifier did before R11 caught it. Every new grading path in this project has to clear this
before it is trusted, and the R16 comparator cleared it on all three tasks.

---

## Reading

Nothing in these twelve samples contradicts the reported aggregates. Two things are worth
carrying forward:

1. **Two of the three failed trajectories made no edit whatsoever and still scored 1.0 on
   the visible suite.** Every claim in this project about the visible/hidden gap rests on
   cases like these, and they are not marginal — they are empty diffs.
2. **The partial repair (#6) failed exactly one check, and it was the right one.** The
   hidden suite degrades in proportion to the defect rather than collapsing to pass/fail,
   which is what makes the per-check failure lists usable as diagnosis.
