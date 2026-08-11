# VERIFIERS_VERSION_SNAPSHOT

Pinned target of the G1 migration. Regenerate the raw facts with
`docker/study_v1.py` (writes `artifacts/raw/v1_study/`).

## Pin

| | |
|---|---|
| package | `verifiers` |
| version | **0.3.0** (exact pin, not `>=`) |
| resolved from | `/usr/local/lib/python3.12/site-packages/verifiers/v1/__init__.py` |
| python | 3.12 |
| torch | 2.5.1+cpu (CPU wheels only; no GPU in this project by construction) |
| numpy | 2.5.2 |
| image | `trgym-v1:latest`, built by `docker/Dockerfile.v1` |

The pin is exact because 0.2.x → 0.3.x was a rewrite and the v1 API is still moving. A
version bump invalidates the migration until `tests_v1/` is re-run;
`test_pinned_verifiers_version_imports` fails loudly rather than silently adapting.

## Platform constraint (hard)

`verifiers.v1` **cannot be imported on Windows.**
`verifiers/v1/runtimes/limiters.py` imports POSIX-only `fcntl` for its `CreationLimiter`,
which serialises concurrent runtime creation. Confirmed twice; classified TYPE B.

Consequence: all v1 development, tests and evaluation run in Linux/Docker. The host test
suite stays green on Windows because `tests_v1/` is guarded by
`pytest.mark.skipif(sys.platform == "win32")` — 155 host tests pass, 15 skip.

Note this is not an incidental portability wart: budgets and concurrency limits are where
v1 touches the OS, which is why `fcntl` is load-bearing rather than accidental.

## numpy major-version note

The container resolves **numpy 2.5.2** because verifiers 0.3.0 requires `numpy>=2`, while
the Windows host venv carries `numpy<2`. That divergence is deliberate and useful: the E2
contract probe turns on `numpy.float64` subclassing `float`, so the check has now been
verified across a numpy major boundary. Confirmed in-container:

```
numpy float64 is float subclass: True   |   type(numpy.float64(1.0)) is float: False
```

`type(x) is float` (not `isinstance`) holds under both majors — see PROTOCOL_CHANGELOG R6.

## v1 surface actually used

Recorded in `artifacts/raw/v1_provenance.json` from a real scored rollout, not from
imports. 95 `verifiers.v1.*` modules loaded; 0 `verifiers.envs.*` modules loaded.

| v1 API | how this project uses it |
|---|---|
| `Taskset` | `TransformerRepairTaskset.load()` yields the 10 repo tasks |
| `Task` | `TransformerRepairTask`, `NEEDS_CONTAINER = True` |
| `TaskData` | `TransformerRepairData` — carries no gold and no expected values |
| `TaskResources`, `TaskTimeout` | per-task cpu/memory/disk and per-phase timeouts |
| `Runtime` / `DockerRuntime` | the candidate container; `write`/`read`/`run` |
| `Trace` | `record_reward`/`record_metric` via the inherited `Task.score` |
| `@reward` / `@metric` | `semantic_repair`; `files_changed`, `touched_the_defective_files`, `infra_error`, `hit_turn_limit` |
| `BashHarness` (built-in) | drives the agent rollout — deliberately not a bespoke harness |
| CLI `validate` | model-free dry run, `valid_rate 1.0` |
| CLI `eval` | the live DeepSeek smoke |

## v0 modules that load regardless

`verifiers/__init__.py` lines 29–36 **eagerly** import `Parser` and `Rubric`, so any
`verifiers.v1` import loads 8 v0 modules (`verifiers.parsers.*`, `verifiers.rubrics.*`).
This is the library's packaging and cannot be avoided by us.

`verifiers.envs.*` is different — a **lazy** import map (`"SingleTurnEnv":
"verifiers.envs.singleturn_env:SingleTurnEnv"`). Those modules load only if something
reaches for a v0 environment, so their absence is real evidence that the v0 execution
path was never entered. That is the only v0 claim this project makes, and it is the one
that is actually checkable.

## Gotchas that cost time, recorded so they cost it once

- `vf-eval` is the **legacy v0** CLI. The v1 entry points are `eval`, `validate`, `debug`,
  `replay`. `eval` collides with the shell builtin — invoke `/usr/local/bin/eval`.
- `pydantic_config` cannot build a list or tuple from a bare CLI scalar; list-valued
  config must come from TOML/JSON. CLI-facing fields here are `list`, not `tuple`.
- `EvalConfig` layout: `model`/`client`/`sampling` are top level, the taskset is under
  `[env.taskset]`, and `SingleAgentEnv`'s one seat is `agent`, so runtime/harness/budget
  live under `[env.agent.*]`. `client` is a discriminated union needing `type = "eval"`.
- `TaskData.timeout` is a `TaskTimeout` model (`setup`/`agent`/`finalize`/`scoring`), not
  an int.
- `trace.rewards[k]` is a `Reward(score, weight)`; `trace.metrics[k]` is a plain float.
- `Trace` requires a `TraceTask` **and** an `AgentInfo`; there is no bare
  `Trace(id=..., task="name")`. See `make_trace` in `tests_v1/conftest.py`, copied from
  `verifiers/v1/cli/validate.py`.
- The harness program is launched with only `HarnessConfig.resolved_env`, and its `bash`
  tool inherits that env — so **`PATH` must be supplied explicitly** or the agent cannot
  find `python`.
- v1's docker runtime bind-mounts a `/tmp/vf-proxy-*` path for its egress proxy. Under
  docker-out-of-docker that path must resolve on both sides: mount `-v /tmp:/tmp`.
