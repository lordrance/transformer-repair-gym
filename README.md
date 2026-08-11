# Transformer ML-Engineering RL Environments & Reward Integrity

A 15-task reinforcement-learning environment in which a model must **diagnose and
repair real Transformer training-code defects**, plus the audit machinery to answer
the question that comes before training:

> Can this reward be trusted, or does it just look like it can?

**Status: `PUBLIC_READY_LOCAL = YES`.** Not published. GPU RL training is an
optional future phase and was **not** performed — there are no checkpoints, no
training curves, and no improvement metrics in this repository.

---

## Why it matters

RL for code is bottlenecked on environments; environments are bottlenecked on
verifiers. The 2026 literature makes that quantitative: **28.5 %** of sampled
SWE-bench Verified tasks accept a Docker-verified *incorrect* patch, and models
score **+14.14 pp** higher on exploitable tasks than robust ones
([arXiv:2606.16062](https://arxiv.org/abs/2606.16062)). A pass rate on an unaudited
verifier is not a capability measurement.

So this project measures the verifier, not just the model.

## Headline findings

| finding | evidence |
|---|---|
| **A weak verifier gets worse as tasks get harder.** Naive false-positive rate on real rollouts climbs **0.10 → 0.20 → 0.60** with difficulty, while the hardened verifier holds at **0.00 FPR / 0.00 FNR** across all 50 trajectories | §11 of the report |
| **The naive verifier accepts 9/9 constructed non-fixes** — including three that never touch the bug — at a 69 % disagreement rate with hardened | `VERIFIER_FUZZ_AUDIT.md` |
| **A measured difficulty gradient**: 4 too-easy, 4 in-band, 2 too-hard across ten repo-level tasks, from one redesign round | `DIFFICULTY_DISTRIBUTION.md` |
| **Zero natural reward hacking in 50 trajectories** — reported as *not observed under this distribution*, not as "secure" | §11 |
| **Trajectory shape carries no signal about outcome.** Success and failure read 10.5 files each, identical to one decimal | `TRAJECTORY_EFFICIENCY.md` |
| **With AdamW, gradient-magnitude bugs do not produce NaN.** Three candidate tasks were measured and rejected because of it; omitting `zero_grad` made the loss *better* than gold | `REAL_BUG_EVIDENCE.md` F6 |
| **Our own independent audit oracle was blind to a whole task family** and produced two wrong labels before being caught. Fixing it *lowered* our headline number | `PROTOCOL_CHANGELOG.md` R3 |

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

## Limitations

- Measures **RL environment readiness**, not RL training effectiveness. No policy
  was trained.
- n = 4 episodes per task. No p-values, no significance claims, no SOTA claims.
- Every difficulty number is difficulty **at a 14-turn budget**; 39 of 40 episodes
  ended on budget exhaustion rather than by the model's choice.
- Both models are DeepSeek. A shared idiosyncrasy would look like a task property.
- **Known open gap:** the Tier M/H hardened suite has no L1 return-type check, so a
  semantically-correct contract violation passes. Documented as verification
  horizon item H1 rather than quietly patched mid-measurement.
- The 8-module repo is a scale model of ML-engineering debugging, not a research
  codebase.

Six known miss-classes are enumerated in
[`VERIFIER_QUALITY_MATRIX.md`](VERIFIER_QUALITY_MATRIX.md). There is no claim
anywhere that the reward is sound.

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
