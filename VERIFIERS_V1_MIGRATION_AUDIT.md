# VERIFIERS_V1_MIGRATION_AUDIT

The ten frozen G1 contract checks, each with the command that produces its evidence.

Every check is **behavioural**: the frozen FAIL condition for G1 is "evaluation still runs
through a v0 path with a v1 wrapper that is not actually exercised", so a test that merely
imports a v1 symbol proves nothing here. Each row below states what would break if the
legacy harness were still the real engine.

```
tests_v1/test_v1_migration.py   16 passed   (Linux/Docker)
tests/ (host suite)            155 passed, 15 skipped   (Windows)
scripts/v1_provenance.py       verdict PASS
```

## Reproduce

```bash
MSYS_NO_PATHCONV=1 docker build -f docker/Dockerfile.v1 -t trgym-v1:latest docker

# -v /tmp:/tmp is REQUIRED: v1's docker runtime bind-mounts /tmp/vf-proxy-* for its
# egress proxy, and under docker-out-of-docker that path must resolve on both sides.
MSYS_NO_PATHCONV=1 docker run --rm -v "e:/RL:/work" -w /work \
  -e PYTHONPATH="/work:/work/environments" -e PYTEST_ADDOPTS="-p no:cacheprovider" \
  -v "/var/run/docker.sock:/var/run/docker.sock" -v "/tmp:/tmp" \
  trgym-v1:latest python -m pytest tests_v1/test_v1_migration.py -q
```

---

## The ten checks

### 1. Fresh install / import of the pinned version — PASS
`test_pinned_verifiers_version_imports`. Asserts `verifiers.__version__ == "0.3.0"` and
that `v1.__file__` is under `site-packages`, so a local shim shadowing the name would fail.

### 2. The public environment imports `verifiers.v1` — PASS
`test_public_environment_subclasses_native_v1_bases`,
`test_public_path_imports_v1_submodules_explicitly`. `issubclass` against the **installed
class objects**, plus an AST check that the public path reaches into
`verifiers.v1.task/taskset/trace/runtimes/utils.decorators/configs.*` by name.

### 3. Taskset loads and tasks enumerate — PASS
`test_taskset_loads_and_enumerates`. 10 tasks, unique ids, and the official view machinery
(`head(3)`, `shuffle(seed=0)`) exercised rather than reimplemented. Also asserts the
system prompt is applied — which only happens via `Taskset.__iter__`, so a subclass that
bypassed the read path would fail.

### 4. `TaskData` is copy-on-write and carries no gold — PASS
`test_taskdata_is_copy_on_write_and_carries_no_gold`. `with_system_prompt` must not alias,
and `model_dump_json()` must not contain `def check_repo_`, `gold_repo`, or expected
values. `TaskData` is serialised into run records, so anything in it is effectively public.

### 5. Docker Runtime actually executes candidate code — PASS
`test_docker_runtime_actually_executes_candidate_code`. Reads back values only the
container's own interpreter could produce (`platform.system() == "Linux"`, `6*7`), runs the
planted repo's visible checks **inside** the runtime, and round-trips a sentinel edit
through `runtime.write`/`runtime.read`. A decorative container fails all three.

### 6. Reward computed from `Trace` — PASS
`test_reward_is_recorded_through_trace`. Asserts `"score" not in vars(type(task))` **and**
`type(task).score is BaseTask.score` — the inherited function object — so reward discovery
and weighting are v1's, not ours. Then checks `trace.rewards["semantic_repair"].weight ==
1.0`, i.e. the `@reward(weight=1.0)` declaration verifiably reached the `Trace`.

**Separation proven, not asserted:** `test_gold_scores_one_and_buggy_scores_zero` writes
gold sources through the runtime and requires 1.0, while the planted bug requires 0.0. This
is the check that caught R11, where the grading bridge recorded every check as passing and
both scored 1.0. No grading path should be trusted before this assertion exists.

### 7. Grading artifacts unreadable by the candidate — PASS
`test_grading_artifacts_are_not_candidate_readable`. The oracle is **absent**, not merely
protected: a filesystem-wide `grep -rl` for `check_repo_matches_gold_logits`, `def
gold_repo` and `build_gold` from inside the candidate returns nothing, no gold directory
exists under any obvious name, and `import trgym.repo.checks` fails in there.

Structural, not a permission bit: candidate sources are read **out** through
`runtime.read()` and graded in a host-side temp dir. Nothing is bind-mounted, so there is
no `..` from the workspace to the grader.

### 8. setup / validate / finalize / cleanup lifecycle — PASS
`test_setup_finalize_cleanup_lifecycle`, `test_validate_preflights_gold`. `setup` plants a
complete tree; `finalize` is callable and leaves the runtime alive; `validate` confirms gold
passes its own hidden suite — the guard against shipping a broken task.

### 9. No cross-rollout state leakage — PASS
`test_no_cross_rollout_state_leak`. Two runtimes for the same task: an edit, a `/tmp`
marker and an exported env var in rollout A must all be invisible in rollout B. Covers the
file, temp-dir and env-var channels in one pass.

### 10. Official v1 CLI dry-run — PASS
`test_official_v1_cli_validate_dry_run`. Runs the real `validate` CLI, which resolves the
taskset **as a plugin** (importing a top-level module named after the id), provisions a
Runtime, and runs `setup` then `validate`. Result: `valid_rate 1.0`, `error 0`.

This one found two genuine packaging defects: the package had to work under both
`environments.transformer_repair` and the CLI's top-level `transformer_repair` (fixed with
relative imports, avoiding duplicate class identities), and list-valued config fields had
to become `list` because `pydantic_config` cannot build a tuple from a CLI scalar.

### Legacy v0 absent from the public path — PASS
`test_public_path_is_free_of_legacy_verifiers_api` scans `environments/` for
`vf.Environment`, `SingleTurnEnv`, `MultiTurnEnv`, `Rubric`, and any `import verifiers` /
non-`v1` `from verifiers import`. Comments and string literals are stripped first, because
a raw substring scan flags this project's own prose about the ban.

`test_the_legacy_detector_actually_detects` is the positive control:
`legacy_research/transformer_repair_v0.py` is genuine `SingleTurnEnv`/`Rubric` code and the
detector **must** flag it. A ban that cannot fire is decoration.

---

## Live DeepSeek smoke through the v1 lifecycle — PASS

`artifacts/raw/v1_smoke2/` (`m1_attention_regression`, 24 turns, bash harness, docker
runtime).

```
env      transformer_repair  ·  bash harness  ·  docker runtime
model    deepseek-chat  via https://api.deepseek.com/v1
reward   semantic_repair 1.00
metrics  files_changed 1.00 · touched_the_defective_files 1.00
         hit_turn_limit 0.00 · infra_error 0.00
usage    11.9K/5.1K tokens (198.5K cached)
time     boot 0.9s · setup 11s · agent 1m 27s · scoring 3s
stop     agent_completed        ok: true       errors: []
```

The model located the defect unaided from a symptom-only prompt and made the minimal
correct edit, in the one file that was actually mutated:

```python
# tinygpt/attention.py
- causal = torch.ones(...).tril(diagonal=1)
+ causal = torch.ones(...).tril(diagonal=0)
```

All six hidden checks pass, with `grading_side: host` and `suite: v2` recorded on the trace.

**Two prior runs are preserved, and the earlier one is the more instructive.**
`artifacts/raw/v1_smoke/` scored 0.00 with `files_changed 0.00`: the agent explored
competently, read `attention.py`, then spent 5 of 14 turns on `which python3` /
`/usr/local/bin/python3`. The harness program is launched with only
`HarnessConfig.resolved_env` and its `bash` tool inherits that, so with no `PATH` the bare
`python` my prompt recommended did not exist. Fixed as configuration
(`[env.agent.harness.env] PATH = …`), not by rewriting the prompt around a broken
environment.

## Which v1 code actually executed

`artifacts/raw/v1_provenance.json`, from a real scored rollout resolved through the
official plugin loader:

```
all_bases_installed      true     (Taskset/Task/Trace/DockerRuntime under site-packages)
score_not_overridden     true     (type(task).score is BaseTask.score)
reward_discovered_by_v1  true     (via verifiers.v1.utils.decorators.discover_decorated)
no_v0_execution_path     true     (0 verifiers.envs.* modules loaded)
buggy_scores_zero        true
PASS                     true

verifiers.v1 modules loaded: 95
```

The v0 claim is deliberately narrow. `verifiers/__init__.py` eagerly imports `Parser` and
`Rubric`, so 8 v0 modules load on any `verifiers.v1` import — asserting their absence would
test the library's packaging, not this code, and would fail forever. `verifiers.envs.*` is
lazily mapped, so **its** absence is real evidence. That is the only v0 claim made here.

## Defects found during the migration

| | what | where |
|---|---|---|
| R9 | the candidate's visible smoke imported the module holding the hidden oracle | `PROTOCOL_CHANGELOG.md` |
| R10 | copying `LeanTask`'s `if trace.has_error: return 0.0` would have re-created the h2 INFRA/capability conflation | " |
| R11 | the grading bridge recorded **every** check as passing — the verifier was vacuously green | " |
| R12 | `infra_error` read `has_error` (`not ok`), which is still False during scoring, so it returned 1.0 for every rollout | " |

R11 is the one worth remembering: it failed **open**, made everything look correct, and
only the gold-vs-no-op separation assertion caught it.
