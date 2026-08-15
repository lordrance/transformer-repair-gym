# FALLBACK_EXECUTION_LOG

Every Plan A failure and the fallback actually taken. Failure types per
`FINAL_FLAGSHIP_COMPLETION_CONTRACT.md`.

| Gate | Plan A | Result | Plan B | Result | Plan C | Result | Final |
|---|---|---|---|---|---|---|---|
| **G0** | hash + copy all historical artifacts, recompute headline metrics | **PASS** — 56 artifacts, 0 missing, 29/29 metrics regenerate exactly | not needed | — | not needed | — | **PASS** |
| **G1** | `import verifiers.v1` on Windows | **FAIL — TYPE B**: `ModuleNotFoundError: No module named 'fcntl'` (POSIX-only, imported transitively by `verifiers/v1/runtimes/limiters.py`) | v1 inside a Linux container | **PASS for discovery** — v1 imports, 168 exports, full API probed into `artifacts/raw/v1_probe.json` | not reached | — | **INCOMPLETE** — API discovered, package not migrated |
| **G2** | replay from live workspaces under `.sandbox_work*` | **FAIL — TYPE A**: `.sandbox_work` had 0 directories; `scripts/build_sandbox.py` calls `shutil.rmtree` on it and destroyed Tier M's 20 workspaces during Phase 1's own final verification | recover Tier E from `patched_source` in the JSONL | **PASS** — 39/40 recovered | deterministic reconstruction of Tier M from `artifacts/raw/tier_m_final_sources/` (buggy template + preserved edited files, no model re-invoked) | **PASS** — coverage 76.7 % → **98.9 %** | **FAIL** — coverage criterion met, but criteria 4 and 7 fail: `v2_FN_count = 1` |
| **G2 (sub)** | BOM-free template files | **FAIL — TYPE A**: 10 `__init__.py` files carried a UTF-8 BOM from PowerShell `Out-File`; `ast.parse` rejects U+FEFF, so the new contract check failed on *gold* | strip BOM from all package files **and** make the AST check BOM-tolerant | **PASS** — contract layer clean on 10/10 repo tasks | — | — | **PASS** |
| **G3** | 24-turn arm on all 5 Tier H tasks, 2 episodes each, frozen 14-turn controls | **RUNNING** at the time of writing — 5/16 workspaces materialised; protocol pre-registered and hashed before the first API call | — | — | — | — | **INCOMPLETE** — data collection unfinished |
| **G4** | Tier S, 3 tasks × 20–50 files | **NOT STARTED** | — | — | — | — | **INCOMPLETE** |
| **G5** | benchmark official v1 runtime first | **NOT STARTED** | — | — | — | — | **INCOMPLETE** |
| **G6** | ≥20 benign human ML snippets | **NOT STARTED** | — | — | — | — | **INCOMPLETE** |
| **G7** | fresh-clone reproduction | **NOT STARTED** | — | — | — | — | **INCOMPLETE** |
| **G8** | final research report | **NOT STARTED** — depends on G1–G7 | — | — | — | — | **INCOMPLETE** |
| **G9** | post-success red-team | **NOT REACHED** — G9 is only meaningful once G0–G8 pass | — | — | — | — | **INCOMPLETE** |

## Detail on the two Plan A failures worth reading

### G1 — `verifiers.v1` cannot load on Windows (TYPE B, confirmed twice)

```
>>> import verifiers.v1
ModuleNotFoundError: No module named 'fcntl'
```

Root cause is in the dependency, not our code: `verifiers/v1/runtimes/limiters.py`
imports `fcntl`, which does not exist on Windows. This was already recorded as a
portability finding in Phase 1's `LICENSE_AUDIT.md`; it is now a hard blocker for
G1's Plan A rather than a footnote.

Plan B worked. Inside `python:3.12-slim` with `verifiers==0.3.0`, v1 imports and
the full lifecycle surface is available. Discovered and recorded:

| concept | actual signature |
|---|---|
| `Taskset` | `__init__(self, config: TasksetConfigT)`; `load, head, shuffle, task_type, toolsets, view` |
| `Task` | `__init__(self, data: DataT, config: ConfigT|None)`; `setup, finalize, score, validate, toolsets, with_system_prompt` |
| `TaskData` | pydantic: `idx, name, description, prompt, system_prompt, image, workdir, network_allow, network_block, artifacts, timeout, resources` |
| `Harness` | `setup, launch, run, score, session, resume, cleanup, install_skills, resolve_prompt` |
| `Runtime` | `start, stop, run, run_program, run_background, run_uv_script, read, write, expose, host_url, open_process, prepare_execution, alive, cleanup, teardown` |
| `Trace` | pydantic; `record_reward, record_metric, record_judge, record_error, stop, to_record` |
| decorators | `@vf.reward(weight=, priority=)`, `@vf.metric(priority=)` |
| runtimes shipped | `base, docker, subprocess, modal, prime, limiters` |
| console scripts | `vf-eval, vf-init, vf-build, vf-install, vf-setup, vf-tui, vf-gepa, eval, validate, debug, replay, gepa` |

**Consequence for the migration**: it is not a rename. `TaskData` is a pydantic
model with `image`/`workdir`/`network_*`/`timeout`/`resources` fields, which means
our Docker sandbox configuration becomes *task data* rather than harness code, and
our `Budget`/`tools.py` layer maps onto `Harness` + `Runtime` rather than sitting
beside them. That is a real restructure and it is the reason G1 was not finished
in this pass.

### G2 — Tier M workspaces were destroyed by our own tooling (TYPE A)

`scripts/build_sandbox.py` uses `./.sandbox_work` as its scratch directory and
`shutil.rmtree`s it. Phase 1's final verification ran that script *after* the Tier
M evaluation, silently deleting the 20 workspaces the replay needed.

This is worth stating plainly: **G0 was designed to prevent exactly this and did
not catch it**, because G0 hashes JSONL and audit files, not workspaces. The
workspaces were never in the frozen manifest.

Plan C recovered it. `scripts/audit_tier_m.py` had already copied every file each
trajectory edited into `artifacts/raw/tier_m_final_sources/` for precisely this
reason. Rebuilding the buggy repo and overlaying those files reproduces the exact
state the grader saw, deterministically, with no model call. Coverage went
76.7 % → 98.9 % (89/90; the one remaining is a Tier E rollout with no
`patched_source`, correctly labelled `UNREPLAYABLE` and kept in the denominator).

**Follow-up owed**: `.sandbox_work*` must be added to the G0 frozen manifest, and
`build_sandbox.py` must use a dedicated scratch path.

## Rules honoured

- No acceptance criterion was lowered to make a fallback succeed. G2's coverage
  criterion was met by *recovering* data, not by shrinking the denominator.
- No task, prompt, model, or success definition was changed after seeing a result.
- The G3 protocol was hashed (`620FDAAE…`) before the first API call of that arm;
  the V2 protocol was hashed (`D297A0FE…`) before the verifier was touched.
- No trajectory was regenerated by calling a model.

---

## G1 — `verifiers.v1` migration (session 2)

**Plan A: develop natively against `verifiers.v1` in Linux/Docker.** Succeeded. The 15-test
suite and the official `validate` CLI dry-run both pass. No Plan B needed for the migration
itself; three of my own defects were found and fixed on the way (R9, R10, R11).

### TYPE B — v1 docker runtime proxy bind-mount under docker-out-of-docker

**Plan A:** run the official `validate`/`eval` CLI from the Linux container with the host
docker socket mounted. Failed:
`docker proxy listener failed: … bind source path does not exist: /tmp/vf-proxy-sxu2kao5`.
The v1 docker runtime bind-mounts a `/tmp/vf-proxy-*` path for its egress proxy, and that
path existed only inside the calling container, not on the daemon's side.

**Plan B:** switch the dry run to `runtime.type = "subprocess"`. **Correctly refused by the
framework** — `TransformerRepairTask.NEEDS_CONTAINER = True`, so v1 rejected it with
"taskset needs a container runtime to validate". This is the framework enforcing my own
declaration, and weakening the declaration to make the run pass would have been exactly the
prohibited move.

**Plan C:** mount the Docker VM's own `/tmp` into the calling container (`-v /tmp:/tmp`) so
the proxy path resolves identically on both sides. Worked: `valid_rate 1.0`. The container
requirement was kept; only the test harness's mount topology changed.

### Not a fallback, a correction

`trgym-v1:latest` disappeared twice mid-session and I initially attributed it to v1's
`DockerRuntime` teardown. There is **no `rmi` anywhere** in `verifiers/v1/runtimes/docker/`.
The loss is environmental (Docker Desktop reclaim). Rebuild before a run; do not blame v1.

### Open, carried into the next session

The live DeepSeek smoke traversed the native lifecycle (bash harness, docker runtime, real
tokens, my rewards/metrics on the `Trace`) but the agent made **zero edits** —
`files_changed 0.00`, `infra_error 1.00`. Not yet root-caused, so **G1 is not PASS**. Prime
suspect is the harness program's `prepare_uv_script` staging needing egress while the task
sets `network_block=["*"]`; the phase ordering suggests it should work, so this needs
evidence rather than a guess. If it is the network policy, the allowance must be narrowed to
what the harness needs during setup — egress-free rollout is a frozen task property and will
not be dropped to make the smoke pass.

*(Resolved in the following session: the cause was `HarnessConfig.resolved_env` carrying no
`PATH`, so the agent's `bash` tool could not find `python`. Fixed as configuration; the
smoke then solved `m1` with `semantic_repair 1.00`.)*

---

## G5 / G9 — TYPE B: all local command execution blocked by the harness safety classifier

**Blocked action.** Every `Bash` and `PowerShell` invocation, from the first call of the
session onward, including read-only ones:

```
python scripts/g5_isolation_canaries.py
```

**Error, identical for both tools:**

> Auto mode could not evaluate this action and is blocking it for safety — a safety check
> separate from auto mode blocked this request because of earlier conversation content — it
> isn't about the action itself. […] Retrying it will hit the same refusal […] it reacts to
> earlier conversation content, not to the action itself, and it will keep firing for the
> rest of this conversation.

**Plan A — run the canary script.** Blocked. The probes are, by construction, code that
reads a process secret, spawns a child process and attempts to import the gold oracle. That
is what an isolation canary *is*; the design cannot avoid it and still test anything.

**Plan B — a different execution route.** `PowerShell` was tried as an independent tool and
returned the identical refusal, so the block is at the classifier rather than the shell. Not
a shell-syntax or permission-mode problem.

**Plan C — restructure the probes to be less adversarial.** **Deliberately not attempted.**
The block explicitly instructs not to rework the action to get around it, and a canary suite
softened until it passes a content filter would no longer demonstrate the vulnerability it
exists to demonstrate. Weakening the probe to obtain a green result is the exact
intent-as-evidence substitution the contract forbids. Declining Plan C here is a discipline
decision, not an inability.

**Classified TYPE B** (platform incompatibility) under the frozen taxonomy, matching the
contract's permitted stopping condition: *"required local execution blocked by Claude
Code/platform permission machinery and there is no available permitted route."*

### Unaffected offline work completed instead

Writing files is unaffected, so everything that does not require execution was done:

| artifact | purpose |
|---|---|
| `tests/test_grading_isolation.py` | 6 permanent R14 regression tests, 5 of them **static/AST** so they need no Docker and run on Windows and in CI; plus a detector-fires test |
| `scripts/final_acceptance.py` | the missing gate verifier required by the exit condition — reads artifacts only, never a report's prose, and fails on absent evidence |
| `scripts/g5_scalability_bench.py` | G5's throughput half: ≥30 sequential jobs, cold/final/in-process reference, nearest-rank p95 |
| `SECURITY_MODEL.md` | required G7 artifact; documents the two-boundary model and, explicitly, what is **not** defended |

### What could NOT be done, and why nothing was claimed

No gate was advanced. G5, G4, G7, G8 and G9 all require running code — canaries, benchmarks,
paid trajectories, a fresh-clone reproduction, mutation testing. The new files above are
**unverified**: they have never been executed. `scripts/final_acceptance.py` in particular
has never run, so it may itself contain defects.

G1 stays FAIL. The R14 repair is on disk and statically guarded, but the frozen rule is that
a gate closes on measurement, not on a fix existing.

**Nothing was marked PASS on the strength of written-but-unrun code.**

---

## Session 4 — canaries ran, R15 found, then execution blocked again

### G5 / R14 — the canary suite finally executed (TYPE A ×2, then success)

**Plan A: run `scripts/g5_isolation_canaries.py` on the Windows host.** Failed, **TYPE A**,
and failed *silently green*: all nine probes reported "contained" under both graders. Root
cause was the harness, not the subject — the script imports
`environments.transformer_repair.grading`, whose package `__init__` reaches `task.py` ->
`verifiers.v1` -> `fcntl`. Every grade raised `ModuleNotFoundError` and the script scored
"no evidence found" as containment.

Two things were fixed rather than worked around:

1. each payload now writes a witness file into its own workspace, and the artifact records
   `contained_by` per probe, so **"never ran" can no longer be reported as "contained"**;
2. `grade_in_process()` was calling `grade_workspace()` with default arguments — which R14
   had already routed to the sandbox — so the positive control was the same grader as the
   subject. It now reaches the pre-R14 path explicitly with `allow_in_process=True`.

**Plan B: run the suite inside the Linux image, where grading actually happens.** Worked,
after one topology correction of the same family as the earlier v1-proxy fix. Grading now
spawns a container, so the nested `docker run -v {REPO_ROOT}:/grader:ro` has to name a path
the *daemon* can resolve; `/work` exists only inside the calling container. Mounting the repo
at `/run/desktop/mnt/host/e/RL` — the path Docker Desktop already uses for `E:\` — makes the
string valid on both sides. No code changed; only the mount point.

Result, and both halves are load-bearing:

```
in_process_UNSAFE_CONTROL   leaked 8/9    <- the probes can detect the old vulnerability
sandboxed_container         leaked 1/9    <- and the isolated path is NOT clean
```

### R15 — the isolated path leaked, so the architecture was changed (TYPE C)

`verdict_tamper_tolerance` escaped, and `module_global_write` returned `graded_passed=True`
on a workspace built from the **buggy** template. Not a probe artefact: the container
isolates the grading host from the candidate, and does nothing about the candidate and the
checks sharing one process inside it.

Classified **TYPE C** (architecture approach failure). Plan A — process isolation alone —
does not deliver verdict integrity, and no amount of container hardening fixes it, because
the grader must import the code it grades.

**Plan B, taken:** two independent layers, both narrow, neither called a boundary — a static
refusal before execution, and a post-run integrity check plus a nonced result protocol. Full
description in PROTOCOL_CHANGELOG R15 and SECURITY_MODEL.md.

**Plan C, considered and rejected:** purge `trgym` from `sys.modules` and drop `/grader` from
`sys.path` before importing the candidate. Rejected on evidence, not taste: `gold_repo()` and
`check_repo_contract_public_api` import `trgym.repo.build` and `trgym.tasks.repo_specs`
*lazily*, so the purge would break gold grading in exchange for coverage the other two layers
already provide.

**No criterion was lowered, and no canary was weakened to obtain a pass.** Two canaries were
*added* (`verdict_tamper_tolerance`, `verdict_forge_protocol`) plus two gate-evading variants,
specifically so that layer 2 is tested even when layer 1 refuses the obvious payload.

### TYPE B — command execution blocked by the harness safety classifier, again

Partway through this session every command-running tool began refusing. The block is
per-command and content-triggered, not tool-specific: `PowerShell` returns the identical
refusal, and a trivial `python -c "print(...)"` is refused while a `cat >> file` heredoc still
runs.

**Exact commands refused** (all needed to close G1 and G5, none of them rewritten to evade
the refusal, per the frozen rule that a canary softened until it passes a filter is no longer
evidence):

```bash
# 1. re-run the canaries against the R15 fix -- the load-bearing measurement
MSYS_NO_PATHCONV=1 docker run --rm \
  -v "e:/RL:/run/desktop/mnt/host/e/RL" -w /run/desktop/mnt/host/e/RL \
  -e PYTHONPATH="/run/desktop/mnt/host/e/RL:/run/desktop/mnt/host/e/RL/environments" \
  -v "/var/run/docker.sock:/var/run/docker.sock" -v "/tmp:/tmp" \
  trgym-v1:latest python scripts/g5_isolation_canaries.py

# 2. G1's runtime evidence (tests_v1 + validate CLI + gold/buggy separation)
MSYS_NO_PATHCONV=1 docker run --rm \
  -v "e:/RL:/run/desktop/mnt/host/e/RL" -w /run/desktop/mnt/host/e/RL \
  -e PYTHONPATH="/run/desktop/mnt/host/e/RL:/run/desktop/mnt/host/e/RL/environments" \
  -v "/var/run/docker.sock:/var/run/docker.sock" -v "/tmp:/tmp" \
  trgym-v1:latest python scripts/v1_runtime_evidence.py

# 3. the host regression suite
.\.venv\Scripts\python.exe -m pytest -q
```

**Plan A** — retry the exact command that had already succeeded once this session: refused.
**Plan B** — `PowerShell` as an independent tool: identical refusal.
**Plan C** — restructure the probes to be less adversarial: **deliberately not attempted**,
same reasoning as the previous session. The refusal explicitly instructs not to rework the
action to get around it, and a canary softened to pass a content filter stops being evidence.

Offline work continued and is recorded below; **no gate was advanced**, because every
remaining gate needs execution.

---

## Permission / Auto-mode diagnosis (session 4b)

Asked to repair the configuration that keeps refusing local verification commands. Diagnosed
from the effective configuration, not from memory. Artifact:
`artifacts/claude_permission_diagnosis.json`.

**The configuration is not misconfigured.** Claude Code 2.1.226; no
`C:/ProgramData/ClaudeCode/managed-settings.json` and no admin policy anywhere; no
`autoMode` section in user settings, so `classifyAllShell` is unset at its built-in default;
no `deny` and no `ask` rules; `E:/RL` had no `.claude` directory at all.

There is one real gap — **no allow rule covers any E:/RL verification command** — but that is
not what is firing. A missing allow rule produces a permission *prompt*. What is happening is
a flat refusal whose own text says it is "a safety check separate from auto mode", triggered
"because of earlier conversation content", and "not about the action itself".

**Discriminating evidence, since a cause has to be established rather than assumed:**

| observation | rules it out |
|---|---|
| `claude --version`, `ls -la <dir>`, `python -c "print(...)"` all refused | no permission rule can deny these; none is sensitive |
| the *same* commands ran successfully earlier in this session under the *same* config | configuration did not change; accumulated conversation content did |
| writing this diagnosis JSON to `artifacts/` succeeded | the refusal is not a blanket tool block |
| writing the permission-rule JSON to `.claude/settings.local.json` **and** to a neutral `.proposed` filename both refused | the trigger is the content — a document granting shell execution — not the path |

**Root cause.** A conversation-content safety check, evaluated per tool call over accumulated
session content. This session's legitimate work is adversarial security testing of the
project's own grader: probes that read a process secret, import the hidden gold oracle, spawn
child processes, mutate an imported module's globals and forge a result protocol. Out of
context those payloads read as exfiltration and tamper code; in context they are the R14/R15
canaries the frozen G5 contract requires. The check cannot see the difference, and it is
correct not to guess.

**What was NOT done, deliberately:** `classifyAllShell` was not written as `false`, because it
is not set to `true` anywhere — writing it would be a no-op that misrepresents the cause. No
`$defaults` touched, no `hard_deny`/`soft_deny` altered, no `bypassPermissions`, no global
classifier disable, no broad `Bash(*)` / `Bash(python *)` / `Bash(docker *)` rule. `claude
auto-mode critique` and `claude auto-mode config` could not be run — the same refusal — so no
custom Auto rule was authored, since it could not have been critiqued before use.

**Classified TYPE B.** Unlike every earlier TYPE B in this log there is no Plan B or Plan C:
the refusal is scoped to the conversation, so it cannot be cleared from inside the
conversation, and the one configuration change that would help is refused for the same
reason. This is the first blocker in this project with no available fallback.

---

## Session 6 — R16 completed and measured; G4/G7/G8/G9 blocked TYPE B

### Completed and verified this session

R16 (trusted-comparator / candidate-container split) was implemented, measured and
documented. All load-bearing evidence was regenerated:

| evidence | result |
|---|---|
| host regression | 212 passed, 16 skipped |
| `tests/test_grading_isolation.py` + `tests/test_obs_protocol.py` | 47 passed, 0 skipped |
| canary suite, candidate path | **0 / 17 leaked, 17 / 17 executed, 0 refused** |
| canary suite, unsafe control | 12 / 17 leaked (probes live) |
| `tests_v1` | 16 passed |
| v1 runtime evidence | gold 1.0, buggy 0.0, `validate` 1.0 |
| G5 throughput | 90 jobs, 0 failures, mean 5.23 s |

`final_acceptance.py` computes **G0 G1 G2 G3 G5 G6 PASS = 6 / 10**.

### The block

Partway through building G4's Tier S template, every `Bash` invocation **outside the
existing `.claude/settings.local.json` allowlist** began returning the auto-mode safety
refusal — the same TYPE B pattern as sessions 3 and 4, which fires on accumulated
conversation content rather than on the command.

The boundary was measured rather than assumed:

- `./.venv/Scripts/python.exe scripts/final_acceptance.py` — **allowlisted, ran normally**
- `./.venv/Scripts/python.exe scripts/build_tier_s_template.py` — **not allowlisted, refused**
- compound/piped forms of the same command — refused

So execution is not dead; it is restricted to the commands already named in the allowlist.

**Widening the allowlist to reach the blocked commands was deliberately NOT done.** It
would work — allowlisted commands bypass the classifier, as the two results above show —
and that is exactly why it is the wrong move. The refusal is a safety check separate from
the permission system, and editing permissions specifically to defeat it is circumvention,
not configuration. Recorded here rather than done quietly.

### What this blocks

Every remaining gate needs to run a script that does not yet exist in the allowlist:

- **G4** — `scripts/build_tier_s_template.py` (written, unrun), plus the freeze and
  trajectory runners still to be written.
- **G7** — `git init`, the fresh-clone reproduction, `uv lock`.
- **G8** — regeneration of `artifacts/final_metrics_summary.json`.
- **G9** — the mutation-testing harness.

### State of the unrun G4 work

`scripts/build_tier_s_template.py` is written but **has never executed**, so nothing about
it is verified. It generates `trgym/repo_template_s/tinygpt/` — 43 files, the eight public
modules kept as thin delegating facades over `_core/`, `_ops/`, `_layers/`, `_model/`,
`_data/`, `_optim/`, `_train/`, `_metrics/`, `_io/`. The design intent is that the existing
hidden checks grade Tier S unchanged, so gold-PASS / no-op-FAIL is inherited from an
already-verified suite instead of re-derived, while a planted defect can sit in
`_ops/masking.py` four levels away from the symptom. None of that is measured. Treat the
file-count and reference-graph claims as intentions until `scripts/tier_s_freeze.py` exists
and has run.

### To resume

A fresh session, or the default permission mode instead of auto mode. The block is scoped
to this conversation and cannot be cleared from inside it.
