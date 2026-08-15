# RESULTS_SUMMARY

Every figure below is regenerated from raw trajectories and gate artifacts by
`scripts/build_final_metrics_summary.py`. Canonical file:
`artifacts/final_metrics_summary.json`, sha256
`07e069a6843d24087484ebdca40c4a9fe04dfd26914114452c67c6a89013227e`, independently
recomputed with separate arithmetic by `scripts/post_success_result_audit.py` (45 values,
0 mismatches). A reader-facing subset is generated at
`artifacts/public_results_summary.json`.

---

## 1. Visible-only grading is not a capability measurement

| family | n | visible pass | hidden pass | naive FPR |
|---|---|---|---|---|
| Tier M (14 turns) | 20 | 20 / 20 | 18 / 20 | 0.10 |
| Tier M pro | 10 | 10 / 10 | 8 / 10 | 0.20 |
| Tier H (14 turns) | 20 | 20 / 20 | 8 / 20 | **0.60** |
| Tier H (24 turns) | 10 | 10 / 10 | 8 / 10 | 0.20 |
| Tier S (48 files) | 12 | 12 / 12 | 9 / 12 | 0.25 |
| **total** | **72** | **72 / 72** | **51 / 72** | — |

**21 of 72 episodes scored 1.0 on the visible suite without repairing the defect.** Two of
them had an *empty diff* — the model changed nothing and still received full marks. The
hidden suite's own false-positive and false-negative rates are 0.00 across every family.

## 2. Candidate isolation (R16 trusted comparator)

| grader path | leaked | executed | refused before execution |
|---|---|---|---|
| sandboxed (production) | **0 / 17** | 17 / 17 | 0 |
| in-process (unsafe control) | 12 / 17 | 17 / 17 | 0 |

Both columns are required. The control is the deliberately unsafe pre-R14 path; if it
contained everything, the probes would be measuring nothing. `17 / 17 executed` matters as
much as `0 leaked` — three earlier versions of this suite reported clean containment while
measuring nothing.

Probe coverage: `/grader` absent, hidden-check source unreadable, gold/template unreadable,
gold unreachable by import and by object-graph walk, no serialized reference bundle present,
verdict/protocol unforgeable, and cross-job state isolation (env, module globals, temp dir,
child process, grader secret).

## 3. Mutation testing of the verifier itself

**8 deliberate breakages, 0 survivors**, all restored byte-for-byte, post-restore suite
green (248 passed, 16 skipped).

Two mutations were themselves wrong before they were right, and both are recorded:
one disabled nothing and would have "survived" for an uninteresting reason; the other
exposed that `repo_contract_return_types` was **dead coverage** — no task and no test
exercised it, so a check added by verifier v2 could have been deleted with the suite still
green. `tests/test_contract_checks.py` (8 tests) closes it.

## 4. Repo-scale localization (Tier S)

48 files, 89 import edges, 0 orphan modules, frozen before any paid call.

```
structural ceiling on reading   0.50   (one tool call per turn, 24 turns)
fraction of repo inspected      mean 0.181, max 0.271
relevant file located           12 / 12
full fix                         9 / 12
```

Localization is not the bottleneck at this scale. The informative failure is
`s3_warmup_offbyone`: all four episodes opened the correct file and three still failed, two
repairing `_train/loop.py` instead of `_optim/schedule.py` — the symptom fits a
`scheduler.step()`/`optimizer.step()` ordering bug just as well.

## 5. Grading throughput and the price of isolation

| phase | mean | p50 | p95 | failures |
|---|---|---|---|---|
| cold | 7.384 s | 7.245 s | 8.543 s | 0 |
| final | 6.948 s | 7.028 s | 7.815 s | 0 |
| in-process reference | 0.668 s | 0.629 s | 0.749 s | 0 |

**10.4×** overhead, 90 jobs, zero failures. The in-process reference is not a candidate
path; it is the pre-R14 design, priced only for comparison.

Re-measured after v0.2-B, and the cost is real: grading went from 5.23 s to **6.95 s**
per job (+33 %) because five predicates now recompute gold's answer on a trusted-supplied
fixture. The earlier figure was measured before that change and no longer described the
code, so it was replaced rather than left standing. Part of the increase was avoidable and
was removed — `_gold_pure_outputs` is now cached, where before it rebuilt gold and
re-entered `RepoModules` once per predicate, five times per job.

That is the trade v0.2-B makes: five checks moved from forgeable to gold-anchored, at
+1.7 s per grading job. Nothing here is throughput-bound.

## 6. Reproducibility

A fresh `git clone` + lockfile install reproduces the full suite and the acceptance run, with
`PYTHONPATH`/`PYTHONHOME`/`VIRTUAL_ENV` stripped. No module and no `sys.path` entry resolves
back into the original working tree.

This check earned its place: its first run failed because `numpy` had never been declared as
a dependency — the original virtualenv was built with `--system-site-packages` and inherited
it, so everything worked locally and nine tests failed in a clean environment.

## 7. v0.2 — what the hardened verifier actually rejects

G2's ordinary replay shows `v1_v2_disagreements == 0` across 89 real trajectories: agents
essentially never produce a semantically-correct tree that violates the documented
interface, so the contract layer is never exercised. A targeted adversarial population was
constructed to separate the two verifiers.

```
ordinary replay, v1/v2 disagreements       0 / 89
adversarial replay, v1 accepted           10 / 11
adversarial replay, v2 rejected           11 / 11
distinguishing (v1 accept, v2 reject)     10 / 11
pre-registered expectations met           all
```

Controls behaved correctly: unmodified gold accepted by both; a real causal-mask defect
rejected by both. The one non-distinguishing case was *predicted* to also break v1, and did.

**This is not a base rate.** All 11 cases were built specifically to separate the verifiers.
It says what v2 rejects, not how often such violations occur. See
`VERIFIER_ADVERSARIAL_REPLAY.md`.

## 8. v0.2 — forgeable predicate surface reduced

| | before | after |
|---|---|---|
| forgeable | 19 / 23 | **14 / 23** |
| gold-anchored (cannot be faked) | 4 / 23 | **9 / 23** |

Five checks were converted by moving their fixture generation to the trusted side, so the
trusted process can recompute gold's answer and require a match:
`repo_padding_keys_masked`, `repo_rope_relative_property`, `repo_rope_norm_preserved`,
`repo_visible_single_token_attention`, `repo_visible_rope_position_zero`.

Sound only because these are pure functions with a unique correct answer. The
training-dynamics checks were deliberately **not** converted: a buggy candidate's loss curve
is supposed to diverge from gold's, so anchoring them to gold would be wrong rather than
stricter. The two sets must partition the predicate space, and a test fails if a new check
is added unclassified.

## 9. v0.2 — Inspect compatibility

A thin adapter exposes the native tasks through UK AISI Inspect's dataset/solver/scorer
interface. It defines no checks of its own and grades through the same R16 trusted
comparator with `fallback=False`, so an Inspect run cannot score more permissively than the
native harness.

```
inspect eval inspect_adapter/transformer_repair_inspect.py --model mockllm/model
    -> accuracy 1.000   (gold)
inspect eval ... -T mode=noop --model mockllm/model
    -> accuracy 0.000   (planted defect)
```

Deterministic, no API key, no model call. Both directions are checked because a scorer that
cannot fail is not a scorer. Eight drift tests pin the adapter to the native suite.

## 10. Budget

24 / 30 precommitted trajectories (10 turn-budget ablation, 12 Tier S, 2 runtime smokes).
Lifetime spend across all phases: 112 trajectories, **$1.2986**. No GPU training.

---

## Not claimed

No comparison against published systems. No inferential statistics — samples are 10–20 per
arm. No policy trained. No generalization beyond these tasks. No deployment-grade security.

**Verifier v2 is unexercised on the replay population**: `v1_v2_disagreements == 0` across 89
replayed trajectories, so it passes its frozen criterion while demonstrating nothing about
hardening. **19 of 23 predicates remain forgeable** — the trusted comparator stops a
candidate deciding its verdict, not lying about its own numbers.
