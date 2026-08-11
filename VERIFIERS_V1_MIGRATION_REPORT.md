# VERIFIERS_V1_MIGRATION_REPORT

What moved, why it moved that way, and what the migration cost.

Per-check evidence is in `VERIFIERS_V1_MIGRATION_AUDIT.md`; the pin and platform facts are
in `VERIFIERS_VERSION_SNAPSHOT.md`.

---

## Why this was not optional

Prime Intellect deprecated the v0 API. The Phase 0/0.5/1 environment was a
`SingleTurnEnv` with a `Rubric`, so the public path was built on an interface that no
longer exists going forward. A wrapper would have preserved the appearance of currency
while leaving the real work in deprecated code — which is why the frozen contract fails G1
on exactly that: *"evaluation still runs through a v0 path with a v1 wrapper that is not
actually exercised."*

## The finding that reshaped the plan

`G1_MIGRATION_DESIGN.md` — written before reading the installed source — argued the
migration was a **polarity inversion**. The reasoning was that `Harness.launch` receives an
`endpoint` and a `secret` and returns a `ProgramResult`, and that the package ships an `ACP`
class with `mcp_urls`; so v1 launches the agent *inside* the runtime and has it dial out,
whereas my harness ran the loop host-side and dispatched tools inward. On that reading the
migration meant rebuilding the agent loop as an MCP toolset.

Reading `LeanTaskset`/`LeanTask` — the shipped reference for a container-graded task —
showed the division of labour is different:

> **`Task` owns setup and grading. `Harness` owns the rollout.** A `Task` plants its files
> with `runtime.write()`, exposes `@reward` methods that read back with `runtime.read()`,
> and "lets any container-capable harness edit it".

So there was no polarity to invert and no bespoke harness to write. The correct migration is
a `Taskset` + `Task` pair plus the **built-in** `bash` harness. That is both less code and
more honest: a custom `Harness` subclass whose `launch` ignored `endpoint` would have
satisfied the type contract while leaving the ACP/MCP path dead — a wrapper wearing a
subclass, failing G1 on the same grounds as any other wrapper.

Evidence beat the design document, which is the outcome the instruction to "study official
examples/tests before implementing each abstraction" exists to produce.

## The five open API questions, answered from source

| | answer |
|---|---|
| Q1 how `load()` yields tasks | subclass hook, may be a generator; `__iter__` is the read path applying system prompt + `head`/`shuffle` views |
| Q2 does `score` return or mutate | mutates `Trace`, and **must not be overridden** — the base already does discovery, invocation and weighting |
| Q3 what `@reward` decorates | `Task` methods declaring `task`/`trace`/`runtime` **by name**; `rollout.py:195` injects them, which is why `setup(self, runtime)` without `trace` is legal. Weight rides on `_vf_weight` |
| Q4 where budgets live | **not** `runtimes/limiters.py` — that is a `fcntl` `CreationLimiter` for concurrent runtime creation. Turn budgets are `AgentConfig.max_turns` |
| Q5 a reference to copy | yes: `LeanTaskset`/`LeanTask`, plus `BashHarness`/`NullHarness` and `DockerRuntime` |

Q4 mattered twice: it explains why v1 is POSIX-only (`fcntl` is load-bearing for
concurrency, not an incidental import) and it stopped me building a budget layer that
already existed.

## What was built

```
environments/transformer_repair/          the public path, 100% verifiers.v1
  configs.py     TransformerRepairConfig / TransformerRepairTaskConfig
  task.py        TransformerRepairTask(Task[...]), NEEDS_CONTAINER = True
                 setup · validate · @reward semantic_repair
                 @metric files_changed / touched_the_defective_files
                         / infra_error / hit_turn_limit
  taskset.py     TransformerRepairTaskset(Taskset[...]), load() over 10 repo tasks
  grading.py     host-side bridge; the oracle is ABSENT from the runtime
  configs/*.toml official CLI configs (validate dry run, DeepSeek smoke)

legacy_research/transformer_repair_v0.py  the v0 SingleTurnEnv, off the public path
```

Internal imports are **relative**, so the package works both as
`environments.transformer_repair` and as the CLI's top-level `transformer_repair` without
creating two module objects for the same files (which would break `issubclass`).

The v0 environment is **kept, not deleted**: it produced the 90 frozen historical
trajectories, and removing it would leave that evidence with no runnable provenance. A test
asserts nothing under `environments/` imports it.

## The security property survived the migration

The frozen requirement is that the gold oracle is **unreachable, not merely protected**.
Under v1 that is preserved structurally rather than by permissions:

- the candidate's code lives in a `DockerRuntime`;
- the hidden checks and gold live only in the rollout process;
- grading reads candidate sources **out** via `runtime.read()` and grades them in a
  host-side temp dir;
- nothing is bind-mounted, so there is no `..` from the workspace to the grader.

A candidate that greps its entire filesystem finds nothing to tamper with, because nothing
is there. Verified from inside the container, not asserted.

One consequence stated plainly: grading trusts the **bytes read out of** the runtime, never
the runtime's own report of success. Only files in the declared editable set are copied
out, so a smuggled `conftest.py` or `sitecustomize.py` cannot enter the graded tree.

**This is also where R9 was found.** The planted visible-smoke runner did
`from trgym.repo.checks import run_repo_checks` — importing the module that also defines the
hidden L2/L3 checks and `gold_repo()`. On Windows this was invisible because `trgym`
happened to be importable; it failed the moment the workspace lived somewhere `trgym` did
not. The honest reading is not "the container lacks a dependency" but that the dependency
should never have existed. The visible layer now lives in
`trgym/repo/visible_runtime.py`, contains no ground truth, and is planted beside the runner.

## Cost of the migration

| | |
|---|---|
| new tests | 16 (`tests_v1/`), Linux-only |
| host tests | 155 pass, 15 skip — unchanged in number, so nothing was traded away |
| defects found in my own code | 4 (R9, R10, R11, R12) |
| paid trajectories spent | 2 (one failed smoke, one successful) |
| platform fallbacks needed | 1 (TYPE B, docker-out-of-docker proxy mount) |

The two most valuable outcomes were not the migration itself:

**R11** — the grading bridge recorded every check as passing, because `run_repo_checks`
returns `(name, ok, message)` tuples rather than raising and I wrapped it in `try/except`.
Gold and the planted bug both scored 1.0. It failed **open**, made everything look correct,
and nothing but the gold-vs-no-op separation assertion would have caught it.

**R12** — the `infra_error` metric added by R10 read `trace.has_error`, which is
`not trace.ok`, and `ok` is set *after* scoring. It therefore returned 1.0 for every
rollout. R11 and R12 are the same shape: a signal that always says the same thing. Neither
was caught by asserting the expected value on the expected input; both needed a test
asserting the signal *changes*. Every reward and metric here now has a fires /
does-not-fire pair.

## A correction to my own earlier reporting

After the first smoke I wrote that "the agent did nothing", reading `infra_error 1.00` and
`files_changed 0.00` off the summary line. The trace record showed `ok: true`,
`is_completed: true`, `errors: []` and **14 model calls** — a competent investigation that
read the guilty file and then lost 5 turns to `which python3` because the harness program's
env had no `PATH`. The environment was misconfigured and my metric was broken; the model
was fine. Summary lines are derived values, and I should have read the trace first.

## What is still not proven

- Only `m1_attention_regression` has been through a live v1 rollout. The other nine tasks
  are enumerated, validated (gold passes) and graded in tests, but not model-driven.
- The v1 CLI dry run and smoke both run under docker-out-of-docker with `-v /tmp:/tmp`. A
  native Linux host would not need that mount; the requirement is a property of this
  Windows + Docker Desktop setup, not of the environment.
- No claim is made about model capability from a single successful episode. It is an
  integration proof, not a measurement. G4 is where measurement happens, on precommitted
  tasks and trajectory counts.
