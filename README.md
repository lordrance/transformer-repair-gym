# transformer-repair-gym

**An evaluation environment for LLM code repair, built to answer the question that comes
before "what score did the model get?" — can the grader be trusted?**

A model is given a small transformer training repository with a planted defect, a
user-style symptom report, and no indication of which file is wrong. It must localize and
fix the defect. The interesting engineering is not the tasks; it is keeping the *answer key*
away from the thing being graded, which took three attempts to get right.

> No RL training was performed. No gradient was ever taken against these rewards. This is
> an evaluation environment and a verifier-integrity study, not a training result.

---

## Why grader integrity is the bottleneck

Grade a code-repair task by running the tests the candidate can see, and you measure
whether the visible suite passes — not whether the bug is fixed. The gap is where reward
hacking lives, and it is large:

**Across 72 episodes, the visible suite passed 72 times. The hidden suite passed 51.**
Twenty-one episodes scored a perfect 1.0 without fixing the defect — **including episodes
with an empty diff**, where the model changed nothing at all.

Any pass rate reported on an unaudited verifier is measuring the verifier.

## Architecture: the candidate never sees the answer key

```
   ┌─────────────────────────┐                      ┌──────────────────────────────┐
   │   CANDIDATE CONTAINER   │   validated          │     TRUSTED COMPARATOR       │
   │                         │   observations       │                              │
   │  • candidate's repo     │ ───────────────────► │  • gold reference repo       │
   │  • standalone probe     │   (nonced, typed,    │  • hidden checks             │
   │  • public task inputs   │    size-bounded,     │  • all pass/fail predicates  │
   │                         │    no pickle)        │                              │
   │  NO gold. NO oracle.    │                      │  never imports candidate code│
   │  NO template. NO repo.  │ ◄─── public inputs ─ │                              │
   └─────────────────────────┘      only            └──────────────────────────────┘
        --network=none                                    runs after the container exits
        --cap-drop=ALL                                    decides the verdict here
        --read-only, uid 10001
```

The candidate computes **observations** — its own logits, its own loss values — and returns
them. It never returns a verdict, because a verdict computed inside the candidate's process
is a verdict the candidate can forge.

Getting here took three iterations, and the first two *measured clean while being wrong*:

| | what was believed | what was actually true |
|---|---|---|
| **R14** | grading is isolated | grading imported candidate code into the process holding gold |
| **R15** | the container is the boundary | inside it, candidate code could set `ATOL = 1e9` and pass a broken tree |
| **R16** | a pattern denylist stops oracle access | the whole repo was mounted at `/grader`; `open('/grader/trgym/repo/checks.py')` was enough |

R16's fix is structural rather than defensive: remove the oracle from the room.

## Measured results

| result | number |
|---|---|
| Visible-only grading accepts non-fixes | **21 / 72** episodes scored 1.0 without repairing the defect, incl. empty diffs |
| Candidate-container isolation | **0 / 17** probes leaked, **17 / 17 executed**; unsafe control still leaks **12 / 17** |
| Mutation testing of the verifier itself | **8 / 8** deliberate breakages turned the right check RED; **0 survivors** |
| Repo-scale localization (48-file package) | defect file located **12 / 12**, repaired **9 / 12**, while reading **≤ 27 %** of the repo |
| Clean-room reproducibility | fresh `git clone` + lockfile install reproduces every result; no module resolves back into the original tree |

`17 / 17 executed` is load-bearing. Three earlier versions of the isolation suite reported
clean containment while measuring nothing at all — once because the package could not import
on the host, once because the "unsafe" control was secretly calling the safe grader, and once
because a denylist refused every probe *before it ran*, so "contained" meant "never happened".

## Architecture

```
tinygpt/       8 modules -- config, norm, positional, attention, model, data, optim, train
tests/         the visible suite; regenerated before grading
SYMPTOM.md     what a user reported: no file named, no root cause
```

Five tools, no shell: `list_files`, `read_file`, `run_command` (4 named commands),
`apply_patch` (unified diff, fuzzy), `submit`. Budgets: 14 turns, 24 commands,
900 s, 6 kB tool output.

Three tiers, as stepping stones inside the same bug families
([`TASK_CHAINS.md`](TASK_CHAINS.md)):

| tier | n | what changes |
|---|---|---|
| **E** | 5 | one file, location and failing property both disclosed |
| **M** | 5 | 8-module package, location withheld, symptom-only, multi-turn |
| **H** | 5 | 2–3 **interacting** defects — fixing one is not enough |
| **S** | 3 | **48-file** package; the eight public modules are facades and the defect sits 3–4 levels beneath the one the symptom names |

Tier S exists to separate *localization* from *repair*. With one tool call per turn and a
24-turn budget, reading all 48 files is impossible by construction (ceiling: 0.5 of the
repo). Observed: the model read a mean of 18 % and found the right file every time — so at
this scale, localization is not the bottleneck. The failures are more interesting: on
`s3_warmup_offbyone` all four episodes opened the correct file and three still failed, two
of them repairing the training loop instead of the scheduler. Opening the right file is not
the same as reading it correctly.

## An example task

`m5_masking_interaction`. The model is given the package and this report, and is
told nothing else:

> *"The reported loss for the same examples changes depending on what else is in
> the batch, so our numbers are not comparable between runs. Sorting the dataset by
> length made the curve jump visibly. At inference the model puts a lot of
> probability on token 0."*

Two defects, in different modules: `collate()` writes `pad_token` into padded label
positions, and the objective counts `numel()` instead of unmasked positions. Both
models tested fix `data.py` and miss `model.py` — the same failure, every time.

## Naive vs hardened

The comparison only means something if the weak grader is *realistic*.

| | naive | hardened |
|---|---|---|
| oracle | `tests/test_visible.py` **inside the workspace** | gold repo, **outside every mount** |
| runs | `pytest` in the candidate's own directory | 3–4 hidden checks across L1/L2/L3 |
| inputs | one fixed public fixture per property | 5 hidden shapes, unseen step counts |
| anti-tamper | none, deliberately | grader files regenerated; gold unreachable |

The naive grader is not a straw man: **the unmodified buggy repo fails it on all 15
tasks**, and it tests the right properties. It is weak the way real first-pass
graders are weak — one fixture, and it trusts a file the candidate can edit.

The load-bearing design decision is that **the gold repo is never mounted**. Phase
0.5 built an exploit that leaves the bug alone and rewrites the oracle in the same
file; no static gate catches it. It fails only because the oracle is unreachable.
*An oracle the candidate can reach is not an oracle.*

## What this is, and what it is not

**Controlled and synthetic.** The repository is a scale model of ML-engineering debugging,
not a research codebase. Defects are *planted* — derived from real documented bugs
(`REAL_BUG_EVIDENCE.md`) but injected, not naturally occurring. This is not SWE-bench: no
real pull requests, no real issue threads.

**Measured.** Every number above is regenerated from raw trajectories and artifacts by
`scripts/build_final_metrics_summary.py`, whose output digest is pinned in the research
report and re-verified by an independently written recomputation.

**Not claimed.** No comparison against published systems or leading agents. No inferential
statistics — samples are 10–20 episodes per arm and no test or interval is reported. No
policy was trained or improved. No generalization beyond these tasks. No security guarantee
suitable for deployment.

## Limitations

- **No RL.** Despite the domain, no policy was trained. This measures environment and
  verifier readiness.
- **Small n.** 10–20 episodes per arm; Tier S is n = 4 per task. The s1/s2 versus s3 split
  (8/8 against 1/4) is large and deliberately uninterpreted.
- **One model family.** All trajectories are DeepSeek. A shared idiosyncrasy would look
  like a task property.
- **14 of 23 predicates are forgeable** (reduced from 19 in v0.2). Moving verdicts to the
  trusted comparator stops a candidate *deciding* its result; it does not make an untrusted
  process's self-report true. Nine checks are now anchored to gold and cannot be faked. The
  remainder are training-dynamics checks, which cannot be anchored: a buggy candidate's loss
  curve is *supposed* to differ from gold's. See `predicates.FORGEABLE` /
  `predicates.GOLD_ANCHORED` and `SECURITY_MODEL.md`.
- **Verifier v2 does not change any decision on real trajectories.**
  `v1_v2_disagreements == 0` across 89 replayed rollouts — agents essentially never produce
  a semantically-correct tree that violates the documented interface. Targeted adversarial
  replay shows what v2 *does* reject when the population contains such cases (10 of 11
  constructed violations), but that rate is a property of the construction and is **not** a
  base rate. v2 still may not be described as reducing false positives on real data. See
  [`VERIFIER_ADVERSARIAL_REPLAY.md`](VERIFIER_ADVERSARIAL_REPLAY.md).
- **One Tier S symptom names its subsystem.** `s1` mentions the team "refactored the
  attention internals", which narrows the search space. Noticed after the task was frozen
  and disclosed rather than retuned.
- Budget-bound difficulty: most episodes end on budget exhaustion, not by the model
  choosing to stop.

Six known miss-classes are enumerated in
[`VERIFIER_QUALITY_MATRIX.md`](VERIFIER_QUALITY_MATRIX.md). There is no claim anywhere that
the reward is sound.

## Reproduce

Dependencies are pinned in `uv.lock`. These four commands are exactly what
`scripts/fresh_clone_repro.py` executes against a clean clone, and
`artifacts/fresh_clone_run.json` records the result — so this block is verified rather
than merely documented.

<!-- REPRO-BEGIN (parsed by scripts/fresh_clone_repro.py; keep one command per line) -->
```bash
uv sync --extra dev
uv run python scripts/build_sandbox.py
uv run python -m pytest -q
uv run python scripts/final_acceptance.py
```
<!-- REPRO-END -->

`pytest -q` collects `tests/` and `tests_v1/`. On Windows the 16 `tests_v1` tests skip —
`verifiers.v1` imports `fcntl` — and that skip is deliberately left visible rather than
excluded, because a suite that silently shrinks is worse than one that reports what it
did not run. In Linux/Docker they execute:

```bash
docker run --rm -v "$PWD:/w" -w /w -e PYTHONPATH="/w:/w/environments" \
  -v /var/run/docker.sock:/var/run/docker.sock -v /tmp:/tmp \
  trgym-v1:latest python -m pytest tests_v1/ -q
```

Real-model evaluation needs `DEEPSEEK_API_KEY` **in the shell** (never in a file).
All 50 trajectories in this report cost **$1.29** total; commands are in §20 of
the report.

## Read first

1. [`PHASE_1_FINAL_RESEARCH_REPORT.md`](PHASE_1_FINAL_RESEARCH_REPORT.md) — the whole study
2. [`VERIFIER_FUZZ_AUDIT.md`](VERIFIER_FUZZ_AUDIT.md) — the central measurement
3. [`PROTOCOL_CHANGELOG.md`](PROTOCOL_CHANGELOG.md) — three self-caught measurement defects
4. [`DIFFICULTY_DISTRIBUTION.md`](DIFFICULTY_DISTRIBUTION.md) — per-task difficulty
5. [`LITERATURE_GUARDRAILS_2026.md`](LITERATURE_GUARDRAILS_2026.md) — 8 papers → 8 binding rules

Large logs live in `artifacts/raw/`; `artifacts/*.jsonl` holds every tool call,
observation, patch and token count. Nothing is deleted, including superseded runs.

## Licensing and credentials

No third-party code is vendored; the reference implementation was written for this
project. External repositories are cited as **evidence** only, with per-task license
notes in [`SOURCE_ALIGNMENT_AUDIT.csv`](SOURCE_ALIGNMENT_AUDIT.csv). `llm-speedrunner`
(CC BY-NC 4.0) and RE-Bench (no license file) were inspected and **excluded** — see
[`LICENSE_AUDIT.md`](LICENSE_AUDIT.md).

No credential appears in any tracked file; keys are read from the environment only.

## GPU training — optional Phase 2, not executed

What it could add that this phase provably cannot: the reward-integrity A/B under
optimization pressure; held-out generalization as a *model* claim; and a direct test
of the prediction that a fixed verifier degrades as the policy strengthens
([arXiv:2606.26300](https://arxiv.org/abs/2606.26300)). Gated on closing horizon
item H1 and on a persistent grading container — container startup is currently
**16× the cost of the checks it runs**.
