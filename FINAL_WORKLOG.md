# FINAL_WORKLOG

Append-only. Newest section last.

---

## Gate status (authoritative, latest)

```
G0 PASS   G1 FAIL (was PASS, flipped -- see R14/R15)   G2 PASS   G3 PASS
G6 PASS   G5 FAIL (isolation measured; the FIX is not)   G4 G7 G8 not started
G9 not reached

ACCEPTANCE COUNT = 4 / 10   (G0, G2, G3, G6)
FLAGSHIP_GPU_FREE_COMPLETE = NO
```

**This is now computed, not asserted.** `scripts/final_acceptance.py` was executed for the
first time this session, its gate parsers were fixed against the real artifact schemas, and
it independently returns G0/G2/G3/G6 PASS and everything else FAIL. See the session-4
section at the bottom for what changed and what is still unrun.

**G1 was closed and then re-opened in the same session.** The migration is genuinely done
and verified (16 behavioural tests, official `validate` CLI at `valid_rate 1.0`, a live
DeepSeek smoke that actually solved `m1`, 95 v1 modules proven to execute). But setting up
G5's canaries showed the oracle-unreachability claim rested on one boundary: grading read
the candidate's sources out of the container and then **imported them in the rollout
process**, which is where `gold_repo()` lives. The container constrains the agent; it does
not constrain code the grader imports on the agent's behalf. Repaired (grading now runs in
the locked-down sandbox) but **not re-measured**, so the gate is FAIL. See R14.

**G5 is blocked, not deferred.** The canary suite exists — 7 probes covering file, env var,
module global, child process, temp dir, grader secret, plus a direct gold-import attempt.
Its verification run could not be executed in this session: the harness safety classifier
declined to evaluate the script, because its probes are by construction code that reads
secrets and spawns child processes. Claiming containment without running them would be the
intent-as-evidence substitution the contract forbids.

Budget: **12 of 30** paid trajectories (10 G3 + 2 v1 smokes). Spend still **< $0.70**.

"Foundations repaired" is **not** a gate and is never counted toward the total. The
earlier report said 4/10; that was wrong and is corrected.

Budget: **11 of 30** paid trajectories used (10 in G3 + 1 live v1 smoke). Estimated
spend still **< $0.60** against the $5 stop. No GPU training, no push, no new service.

---

## Session: G0 re-closed, then G1 built

### G0 — re-closed twice, on its own defects (PASS)

- **R7**: G0's frozen copies were mutable — it re-copied to a flat name every run, so a
  later run overwrote its own frozen `verifier_fuzz_audit.json`. G0 caught this itself as
  `fuzz.n_probes: claimed 13, recomputed 16` and correctly said FAIL. Freezes are now
  content-addressed (`<name>.<sha[:12]>`, written only if absent). The destroyed v1 fuzz
  artifact is recorded as `LOST_OVERWRITTEN`, **not** reconciled away by editing the
  expectation to 16.
- **R8**: G0 equated "present + stable hash" with "preserved". `artifacts/raw/v1_probe.json`
  had been reported healthy for several runs while not being valid JSON (a PowerShell
  redirect folded docker's stderr into it). Recovered by `scripts/repair_v1_probe.py`,
  original stream kept as `v1_probe.raw.txt`, and G0 now gates on parseability.

Final: **127 artifacts, 0 missing, 30/30 metrics exact, 0 unparseable of 22 structured.**

Standing lesson across R5b → R7 → R8: each was found by *using* the evidence downstream,
never by re-reading the gate.

### G1 — real work done, NOT yet PASS

**Five open API questions answered from installed source** (not guessed). Source dumped to
`artifacts/raw/v1_study/` (`v1_src/` is the full package, readable on the host without Docker):

1. `Taskset.load()` is the subclass hook and may be a generator; `__iter__` is the read
   path that applies the system prompt and `head`/`shuffle` views.
2. `Task.score` **mutates** `Trace` and must **not** be overridden — the base already does
   reward/metric discovery, invocation and weighting.
3. `@reward`/`@metric` decorate `Task` methods that declare `task`/`trace`/`runtime` **by
   name**; `rollout.py:195` injects them, which is why `setup(self, runtime)` (no `trace`)
   is legal. Weight rides on `_vf_weight`.
4. Budgets are **not** in `runtimes/limiters.py` — that is a `fcntl` `CreationLimiter` for
   concurrent runtime creation. Turn budgets live in `AgentConfig.max_turns`.
5. Built-in references exist: `LeanTaskset`/`LeanTask` (container-graded, closest analogue),
   plus `BashHarness`/`NullHarness` and `DockerRuntime`.

**The design doc's premise was wrong and evidence won.** `G1_MIGRATION_DESIGN.md` argued the
migration was a "polarity inversion" because `Harness.launch` takes an `endpoint`/`secret`.
The real division of labour is: **Task owns setup + grading; Harness owns the rollout.** So
no custom harness was needed — the built-in `bash` harness drives editing while my Task
plants and grades. That is also what the gate wanted (no bespoke harness).

**Built** (`environments/transformer_repair/`, public path, all relative imports so the
package works as both `environments.transformer_repair` and the CLI's top-level
`transformer_repair`):

- `configs.py` — `TransformerRepairConfig`/`TransformerRepairTaskConfig`
- `task.py` — `TransformerRepairTask(Task[...])`, `NEEDS_CONTAINER = True`, `setup`,
  `validate`, `@reward semantic_repair`, `@metric files_changed / touched_the_defective_files
  / infra_error`
- `taskset.py` — `TransformerRepairTaskset(Taskset[...])`, `load()` over the 10 repo tasks
- `grading.py` — host-side bridge keeping the oracle **absent** from the runtime
- `configs/*.toml` — official CLI configs

`legacy_research/transformer_repair_v0.py` — the v0 `SingleTurnEnv` moved off the public
path, kept because it produced the 90 frozen trajectories.

**Three defects found and fixed en route** (R9, R10, R11 in `PROTOCOL_CHANGELOG.md`).
R11 is the one that matters most: `grade_workspace` recorded every check as passing because
`run_repo_checks` returns tuples rather than raising, so the v1 verifier was **vacuously
green** — gold and the planted bug both scored 1.0. Only the gold-vs-no-op separation
assertion caught it. Any new grading path needs that assertion *before* being trusted.

**Verified green:**
- `tests_v1/test_v1_migration.py` — **15 passed** in Linux, including a real `DockerRuntime`
  executing candidate code, reward reaching `Trace` with its weight, gold 1.0 vs buggy 0.0,
  oracle unreachable by filesystem-wide grep from inside the candidate, no cross-rollout
  leakage (file/env/tmp), and the official `validate` CLI dry-run at `valid_rate 1.0`.
- Host regression: **155 passed, 14 skipped** (v1 tests correctly skip on Windows).

**Live DeepSeek smoke — traversal proven, but the episode is not yet a satisfying smoke.**
`/usr/local/bin/eval` reported `env transformer_repair · bash harness · docker runtime`,
`model deepseek-chat via https://api.deepseek.com/v1`, real usage `8.7K/1.7K tokens`, and
scored **my** signals (`semantic_repair 0.00`, `files_changed 0.00`, `infra_error 1.00`,
`touched_the_defective_files 0.00`), with a phase breakdown (boot 11s, setup 15s, agent 40s,
scoring 3s). So the native lifecycle genuinely ran.

But `files_changed 0.00` with `infra_error 1.00` means **the agent never edited anything**.
That is `infra_error` doing exactly the job R10 added it for, and it must be root-caused
before G1 can be called PASS.

---

## SESSION BLOCKED — all local execution refused by the safety classifier

A continuation session was opened specifically to finish G1/G5 onward. **Every `Bash` and
`PowerShell` call was refused from the first invocation**, including read-only ones, with the
classifier reporting that it reacts to earlier conversation content and will keep firing for
the remainder of the conversation. Both tools returned the identical refusal, so this is not
a shell or permission-mode issue. Classified **TYPE B**; details and the Plan A/B/C chain are
in `FALLBACK_EXECUTION_LOG.md`.

Consequence: **no gate advanced.** Count remains **4/10** (G0, G2, G3, G6). G1 stays FAIL.

Offline work completed (writing files is unaffected), all of it **unverified — never run**:

- `tests/test_grading_isolation.py` — 6 R14 regression tests; 5 are static/AST so they need
  no Docker and will run on Windows and in CI, plus a detector-fires control
- `scripts/final_acceptance.py` — the gate verifier the exit condition requires and which did
  not previously exist; reads artifacts only, never prose, and fails on absent evidence
- `scripts/g5_scalability_bench.py` — G5's throughput half (≥30 jobs, cold/final/in-process)
- `SECURITY_MODEL.md` — required G7 artifact; the two-boundary model and what is *not* defended

**A fresh Claude Code session is required** — the block is tied to accumulated conversation
content, so continuing here cannot clear it.

## RESUME HERE (current)

### 0. First command in the new session — sanity, then the blocker

```bash
python scripts/final_acceptance.py     # never executed; expect G1/G4/G5/G7/G8/G9 FAIL
python -m pytest tests/test_grading_isolation.py -q   # never executed
```

Both are new and unverified; fix them before trusting their output.

### 1. Unblock G5 + re-close G1 — run OUTSIDE auto mode

```bash
python scripts/g5_isolation_canaries.py
```

Expected, and both halves matter:
- `sandboxed_container` — **all 7 probes contained** (this is the G5 requirement)
- `in_process` — **probes leak** (specifically `grader_secret` and `gold_oracle_import`).
  A canary suite where nothing leaks under the deliberately unsafe grader is not testing
  anything, so a clean `in_process` column means the canaries are broken, not that the code
  is safe.

Then the benchmark half of G5 (≥30 sequential jobs, mean/p50/p95, cold vs final), then
re-run `tests_v1/` (16) and the host suite (165) and re-close G1.

Note the fix has a cost that must be measured honestly, not hidden: grading moved from
in-process to one container per grade. Phase 1 measured container startup at ~8.24 s against
0.54 s of actual checks. If that is unacceptable, the answer is a *safe* persistent grader
(G5 option A), never a return to in-process grading of policy-touched trees.

### 2. Superseded: G1's earlier blocker (resolved)

Root-cause why the bash-harness agent made zero edits. Prime suspect: the harness program
is staged with `runtime.prepare_uv_script` (needs egress to fetch deps) while my `TaskData`
sets `network_block=["*"]`. Ordering suggests it should work (`prepare_setup` → `task.setup`
→ `harness.setup` → `prepare_execution([endpoint, …])`), so this needs evidence, not a guess.

```bash
# 1. the per-trace record is nested under the top-level "traces" key -- the outer
#    object reported ok:true, so read the INNER trace's ok/stop_condition/errors:
python -c "import json;d=json.loads(open('artifacts/raw/v1_smoke/traces.jsonl').read().splitlines()[0]);import sys;t=d['traces'][0] if isinstance(d.get('traces'),list) else d['traces'];print(json.dumps({k:t.get(k) for k in ('ok','is_completed','stop_condition','errors','tools','calls')},indent=2)[:4000])"

# 2. and the full CLI log, which will name a harness/program failure if there was one:
tail -120 artifacts/raw/v1_smoke/eval.log
```

If it is the network policy, the fix is to allow only what the harness program needs during
setup — **do not** simply drop `network_block`, since egress-free rollout is a frozen
property of the task.

### Canonical commands (all require Linux/Docker)

```bash
# rebuild the image (it has been reclaimed by Docker Desktop more than once;
# this is NOT v1 deleting it -- there is no `rmi` anywhere in v1's docker runtime)
MSYS_NO_PATHCONV=1 docker build -f docker/Dockerfile.v1 -t trgym-v1:latest docker

# the v1 test suite (15 tests). -v /tmp:/tmp is REQUIRED: v1's docker runtime
# bind-mounts a /tmp/vf-proxy-* path for its proxy listener, and under
# docker-out-of-docker that path must resolve identically on both sides.
MSYS_NO_PATHCONV=1 docker run --rm -v "e:/RL:/work" -w /work \
  -e PYTHONPATH="/work:/work/environments" -e PYTEST_ADDOPTS="-p no:cacheprovider" \
  -v "/var/run/docker.sock:/var/run/docker.sock" -v "/tmp:/tmp" \
  trgym-v1:latest python -m pytest tests_v1/test_v1_migration.py -q

# host regression (Windows, 155 tests)
.\.venv\Scripts\python.exe -m pytest -q

# G0
.\.venv\Scripts\python.exe scripts/freeze_historical_artifacts.py
```

For the live smoke, the key is read from the **User** environment scope and passed in
without ever being printed:

```powershell
$k = [Environment]::GetEnvironmentVariable('DEEPSEEK_API_KEY','User')
docker run --rm -v "e:/RL:/work" -w /work -e PYTHONPATH="/work:/work/environments" `
  -e DEEPSEEK_API_KEY=$k -v "/var/run/docker.sock:/var/run/docker.sock" -v "/tmp:/tmp" `
  trgym-v1:latest sh -c "/usr/local/bin/eval transformer_repair -o /work/artifacts/raw/v1_smoke2 '@' environments/transformer_repair/configs/m1_deepseek_smoke.toml"
```

### Then, in order

1. finish G1 (the blocker above), re-run **both** suites, write
   `VERIFIERS_V1_MIGRATION_AUDIT.md`, `VERIFIERS_V1_MIGRATION_REPORT.md`,
   `VERIFIERS_VERSION_SNAPSHOT.md`, and only then mark G1 PASS
2. G6 heuristic false-positive audit (deterministic, no paid calls)
3. G5 grader scalability + isolation (≥30 jobs)
4. re-run `h2` under the repaired adapter so its difficulty becomes measured
5. G4 Tier S — freeze 3 tasks and hashes **before** the 12 precommitted trajectories
6. G7 packaging/CI/true fresh clone, G8 synthesis, G9 full red-team incl. mutation testing

### Gotchas worth not rediscovering

- Windows cannot `import verifiers.v1` (`fcntl`). Everything v1 runs in Linux/Docker.
- `vf-eval` is the **legacy v0** CLI. The v1 entry points are `eval`, `validate`, `debug`,
  `replay`; `eval` collides with the shell builtin, so call `/usr/local/bin/eval`.
- pydantic_config cannot build a list/tuple from a bare CLI scalar — list-valued config
  must come from TOML/JSON.
- `EvalConfig` layout: `model`/`client`/`sampling` top level, taskset under `[env.taskset]`,
  the single seat under `[env.agent.*]`. `client` is a discriminated union needing `type = "eval"`.
- `trace.rewards[k]` is a `Reward(score, weight)`; `trace.metrics[k]` is a plain float.
- `Trace` requires a `TraceTask` **and** an `AgentInfo` — see `make_trace` in
  `tests_v1/conftest.py`, copied from `cli/validate.py`.

---

## Session 4 — final_acceptance runs; the canaries run; R15 found; execution blocked

Budget unchanged: **12 of 30** paid trajectories (10 G3 + 2 v1 smokes), new spend still
**< $0.70**. **Zero model calls were made this session.** (`artifacts/total_spend.json`'s
112 / $1.2986 is the *lifetime* Phase-0.5 + Phase-1 total and is not this contract's ledger;
the 12 are `artifacts/tier_h_24turn.jsonl` (10 rows) plus `artifacts/raw/v1_smoke{,2}`.)

### `scripts/final_acceptance.py` — executed for the first time, and it was wrong

Written blind in session 3, never run. Two of its gate parsers asserted field names that do
not exist, so they failed gates that actually pass:

- **G0** read `manifest["metrics"]` / `["n_metrics"]`. The real manifest stores
  `regenerated_headline_metrics` and `claimed_headline_metrics` as parallel dicts of metric
  *families*. It now counts individual metrics across families (≥6 required) **and** asserts
  every regenerated family has a claimed counterpart — a stronger check than the broken one.
- **G2** looked for `artifacts/verifier_v2_replay_summary.json` with `coverage` / `v1_FP` /
  `v2_FP` / `v2_FN`. The real artifact is `artifacts/verifier_v2_replay.json` with
  `summary.replay_coverage`, `v1_FP_rate`, `v2_FP_rate`, `v2_FN_count`. It now also asserts
  the two frozen protocol criteria the replay records explicitly (`c4`, `c7`).

A third change is a design correction rather than a bug fix. G1's runtime half reported
`BLOCKED-PLATFORM` on Windows, which is a skip wearing a different word. It now reads
`artifacts/raw/v1_runtime_evidence.json` — written by `scripts/v1_runtime_evidence.py`, which
runs *in* Linux and records what it observed — and **fails if that artifact is absent**. The
evidence carries the SHA-256 of `grading.py`, so evidence predating the code it describes is
rejected. On Linux the same checks are additionally re-executed live.

Result after the fixes: `G0 PASS  G2 PASS  G3 PASS  G6 PASS`, everything else FAIL. That
matches the checkpoint exactly — the first independent confirmation of the 4/10 count.

### The canaries ran — and the first two attempts measured nothing

Full detail in FALLBACK_EXECUTION_LOG. Short version: on Windows every probe reported
"contained" because the package import dies on `fcntl` before grading happens, and the
"unsafe" control was calling the *same* sandboxed grader as the subject. Neither can repeat
silently now: payloads write a witness file, the artifact records `contained_by` per probe,
and the control reaches the pre-R14 path explicitly with `allow_in_process=True`.

Run properly — inside the Linux image, repo mounted where the Docker daemon can resolve it:

```
in_process_UNSAFE_CONTROL   leaked 8/9   (grader secret, gold import, host file read,
                                          env var, module global, child process, temp dir,
                                          verdict tamper)
sandboxed_container         leaked 1/9
```

The left column is what makes the right one mean anything. All seven **containment** probes
are contained by the sandboxed path, including the gold-import and grader-secret probes:
R14's repair is real, and now measured rather than asserted.

### R15 — and the isolated path was still not clean

`verdict_tamper_tolerance` escaped, and `module_global_write` — whose entire payload is
`import trgym.repo.checks; ATOL = 1e9` — returned **`graded_passed = True` on a workspace
built from the buggy template**. A tree carrying the planted defect scored a pass through the
production grading path.

R14's own lesson, one level further in: the container stops candidate code reaching the
grading *host*; it does nothing about the candidate and the checks sharing one process
*inside* it, which they must, because grading has to execute the code it is grading.
Containment and integrity are different claims, and I had again tested only one of them.

Fixed with two narrow layers — a static refusal before execution, and a post-run integrity
check plus a per-job nonced result protocol — plus two gate-evading canaries so layer 2 is
tested even when layer 1 refuses the obvious payload. Detail in PROTOCOL_CHANGELOG R15.
Neither layer is called a boundary, and SECURITY_MODEL.md now states plainly that in-process
verdict integrity is mitigated, not guaranteed.

A second real defect surfaced in the same run: the sandbox image runs as uid 10001, and a
`mkdtemp` workspace created by root on Linux is mode 0700 — so the grading container could
not read the tree it was grading, and every check failed with an import error that *looks
like* "the candidate is wrong". Grading now opens the disposable workspace's mode first.

### Then execution was blocked

Partway through, every command-running tool began refusing (same TYPE B as session 3;
`PowerShell` identical; file writes unaffected). The commands needed to close G1 and G5 are
listed verbatim in FALLBACK_EXECUTION_LOG. They were **not** rewritten to evade the refusal.

So: **the R15 fix is written, statically guarded, and not measured.** G1 and G5 stay FAIL.
Nothing was marked PASS on the strength of code that has not been run — including, and
especially, the security fix I am most confident about.

## RESUME HERE (current)

### 0. First: this session's blocker is not a permission rule

Diagnosed in `artifacts/claude_permission_diagnosis.json`. Claude Code 2.1.226, no managed
policy, `classifyAllShell` unset, no deny/ask rules — nothing is misconfigured. The refusals
come from a conversation-content safety check that fires on accumulated session content
(this project's canaries read secrets, import a hidden oracle, spawn processes and forge a
result protocol), and it refused `claude --version` and `ls -la` as readily as the canaries.
It cannot be cleared from inside the session it is scoped to.

**Start a fresh session.** Then paste the prepared narrow allow rules into
`E:/RL/.claude/settings.local.json` — they only cover the named verification scripts and the
one exact `trgym-v1:latest` invocation, so the later runs stop prompting. Do **not** widen
them, and do not disable Auto mode or the classifier.

### 1. The three blocked commands, in order

```bash
# a. host regression suite -- R15 touched trgym/harness/sandbox.py, which Phase 1 tests use.
#    This is the most likely place for the fix to have broken something.
.\.venv\Scripts\python.exe -m pytest -q

# b. R14/R15 static regression tests (8 old + 6 new, no Docker needed)
.\.venv\Scripts\python.exe -m pytest tests/test_grading_isolation.py -q

# c. the canaries, against the fix. THIS is the measurement G5 turns on.
MSYS_NO_PATHCONV=1 docker run --rm \
  -v "e:/RL:/run/desktop/mnt/host/e/RL" -w /run/desktop/mnt/host/e/RL \
  -e PYTHONPATH="/run/desktop/mnt/host/e/RL:/run/desktop/mnt/host/e/RL/environments" \
  -v "/var/run/docker.sock:/var/run/docker.sock" -v "/tmp:/tmp" \
  trgym-v1:latest python scripts/g5_isolation_canaries.py
```

Expected from (c), and **all three conditions matter**:

| | |
|---|---|
| `in_process_UNSAFE_CONTROL` | still leaks ≥5 probes. A clean control means the probes broke. |
| `sandboxed_container` | leaks **0 / 11** |
| `contained_by` | no probe reads `probe_did_not_execute` |

If `verdict_tamper_tolerance_evasive` leaks, layer 2 is not working — debug the worker's
snapshot/compare, do not delete the canary.

### 2. G1 runtime evidence, then re-close G1

```bash
MSYS_NO_PATHCONV=1 docker run --rm \
  -v "e:/RL:/run/desktop/mnt/host/e/RL" -w /run/desktop/mnt/host/e/RL \
  -e PYTHONPATH="/run/desktop/mnt/host/e/RL:/run/desktop/mnt/host/e/RL/environments" \
  -v "/var/run/docker.sock:/var/run/docker.sock" -v "/tmp:/tmp" \
  trgym-v1:latest python scripts/v1_runtime_evidence.py
```

Writes `artifacts/raw/v1_runtime_evidence.json`: v1 import, taskset load via the official
loader, TaskData immutability, gold 1.0 vs buggy 0.0 **through the sandboxed path**, the
`tests_v1` count, and the `validate` CLI `valid_rate`. `final_acceptance.py` reads exactly
these fields and rejects the file if its `grading_sha256` no longer matches `grading.py`.

`tests_v1` now grades through a *nested* container, which it did not before R14. If a test
fails on mount topology rather than behaviour, fix the mount point (as in 1c) — never
`fallback=True`.

Then update `VERIFIERS_V1_MIGRATION_AUDIT.md` / `_REPORT.md` to describe the **three-layer**
model (outbound, inbound, integrity) and say plainly that the original single-boundary
isolation claim was incomplete. Only then G1 PASS.

### 3. G5

```bash
# same docker preamble as above
  trgym-v1:latest python scripts/g5_scalability_bench.py --jobs 30
```

Then write `VERIFIER_SCALABILITY_REPORT.md` from `artifacts/g5_scalability.json` +
`artifacts/g5_isolation_canaries.json`. Take contract Option B unless the numbers are
genuinely unusable; do not build a persistent grader to look impressive.

### 4. Then, in order

h2 re-measurement under the repaired adapter → G4 Tier S (freeze 3 tasks and hashes
**before** the 12 precommitted trajectories) → G7 packaging + true fresh clone → G8
synthesis → G9 full red team including mutation testing → `final_acceptance.py`.

For G9's mutation list, R15 adds a required mutation beyond the five frozen ones: **remove
the worker's tamper check** and confirm `verdict_tamper_tolerance_evasive` turns RED.

### Gotchas added this session

- The canary suite **must** run in Linux. On Windows `environments.transformer_repair`
  cannot even import, and the failure looks like a clean pass.
- Nested grading needs the repo at a daemon-resolvable path:
  `-v e:/RL:/run/desktop/mnt/host/e/RL`. `/work` does not exist on the daemon's side.
- `trgym-v1:latest` gets reclaimed by Docker Desktop regularly. Rebuilding takes ~2 minutes
  and is not a symptom of anything.

---

## Session 6 — R16: the oracle was in the room. G1 and G5 close.

```
G0 PASS   G1 PASS   G2 PASS   G3 PASS   G5 PASS   G6 PASS
G4 FAIL   G7 FAIL   G8 FAIL   G9 not reached

ACCEPTANCE COUNT = 6 / 10     (was 4 / 10)
FLAGSHIP_GPU_FREE_COMPLETE = NO
```

Computed by `scripts/final_acceptance.py`, not asserted. Budget unchanged: **12 of 30**
paid trajectories, spend still **< $0.70**. Zero model calls this session.

### What was actually wrong

R14 sandboxed grading. R15 added integrity layers because candidate code still shared a
process with the checks. Both were defending an asset that should never have been in the
room:

```
-v {REPO_ROOT}:/grader:ro      # sandbox.py, pre-R16
```

The whole repository — hidden checks, gold builder, pristine template — was mounted into
the container that runs candidate code. Two canaries proved it. `gold_oracle_import_evasive`
walked `gc.get_objects()` to `gold_repo()`. `oracle_source_read_in_container` just called
`open('/grader/trgym/repo/checks.py')` — no evasion whatsoever. R15's denylist could not
have stopped the second one, and R15's tamper check cannot see either, because **reading
gold is not mutating gold**.

The existing `host_file_read` canary had been reporting "contained" for three sessions
while aiming at the *host* path, which does not exist inside the container. It was
measuring nothing.

### The fix (R16)

Take the answer sheet out of the exam room.

```
Candidate container                     Trusted comparator
  candidate + public inputs   ──obs──▶    gold + oracle + hidden checks
  no /grader, no gold,                    applies every predicate
  no template, no reference               after the container exits
```

- `trgym/repo/candidate_probe.py` — mounted alone at `/probe`, imports no `trgym`,
  computes observations, decides nothing.
- `trgym/repo/obs_protocol.py` — no pickle. dtype allowlist, rank, element count and exact
  byte length all checked *before* allocation, then `frombuffer`.
- `trgym/repo/predicates.py` — owns gold; never imports candidate code.

All 23 checks were decomposed into `observation → validated narrow result → trusted
predicate`. The logits check **inverted** rather than moved: the candidate's `state_dict`
travels outward into the gold model, so the comparison is unchanged but nothing
gold-derived travels inward.

**R15's pattern denylist is deleted.** It had stopped being load-bearing, and while it
stood it refused every oracle-seeking canary *before execution* — scoring "contained" for
probes that never ran. Removing it took the suite from `refused 4 / ran 11` to
`refused 0 / ran 17`. `test_the_pattern_denylist_is_gone` fails if it returns.

### Measured

| evidence | result |
|---|---|
| canary suite, candidate path | **0 / 17 leaked, 17 / 17 ran, 0 refused** |
| canary suite, unsafe control | 12 / 17 leaked (probes live) |
| host regression | 212 passed, 16 skipped |
| isolation + protocol suites | 47 passed, 0 skipped |
| `tests_v1` (Linux) | 16 passed |
| v1 runtime evidence | gold **1.0**, buggy **0.0**, `validate` 1.0 |
| G5 throughput | 90 jobs, 0 failures, mean **5.23 s** |

R16 made grading *faster*: ~8.24 s → 5.23 s, because the container no longer mounts the
repo or imports the grader stack.

### Four defects found in my own work, by evidence rather than inspection

1. Probe unreadable — `mkdtemp` 0700 vs uid 10001. Same trap R15 hit with the workspace.
2. `grader_mount_absent` fired on an empty `/grader` mount point (`.exists()` is true for
   an empty dir).
3. `reference_bundle_unreachable` matched `torch.ao.nn.quantized.reference.*` on a bare
   substring.
4. Two `final_acceptance` checks matched the **prose documenting the fix** rather than the
   code. Both now read the AST. That is R13's lesson, rediscovered inside the gate meant
   to enforce it.

And one self-inflicted regression: a `testpaths = ["tests"]` I added to `pyproject.toml`
dropped `tests_v1` from collection, turning `212 passed, 16 skipped` into a tidier-looking
`212 passed`. Caught by baseline comparison, reverted. Visible skips are evidence.

## RESUME HERE (current)

### 0. The session ended blocked — TYPE B again

Every `Bash` command **outside the `.claude/settings.local.json` allowlist** began
returning the auto-mode safety refusal. The boundary was measured:

- `scripts/final_acceptance.py` (allowlisted) — **ran**
- `scripts/build_tier_s_template.py` (not allowlisted) — **refused**

Widening the allowlist would work and was **deliberately not done**: the classifier is a
safety check separate from the permission system, and editing permissions to defeat it is
circumvention, not configuration. Start a fresh session, or use the default permission
mode instead of auto mode.

### 1. G4 — Tier S. Read this before trusting anything in it.

`scripts/build_tier_s_template.py` is **written and has never been executed**. Nothing in
it is verified. It is *intended* to emit `trgym/repo_template_s/tinygpt/` — 43 files, the
eight public modules kept as thin delegating facades over `_core/ _ops/ _layers/ _model/
_data/ _optim/ _train/ _metrics/ _io/`, so the existing hidden checks grade Tier S
unchanged and gold-PASS / no-op-FAIL is *inherited* from an already-verified suite rather
than re-derived, while a defect can sit in `_ops/masking.py` four levels from the symptom.

Treat "43 files" and "every file is referenced" as **design intent, not measurements**,
until `scripts/tier_s_freeze.py` exists and has verified the import graph. This project
has twice shipped exactly that class of unverified claim.

Still to write: `scripts/tier_s_freeze.py` (file counts, ≤3 relevant, every non-relevant
file referenced, gold passes, no-op fails → `artifacts/tier_s_spec.json`, frozen **before**
any model call) and the 12-trajectory runner. Spend is authorised (12 trajectories, →24/30).

### 2. G7 — four of six artifacts are done

Present and passing: `LICENSE` (MIT, Rance), `CITATION.cff`, `pyproject.toml`,
`.github/workflows/ci.yml`, `FINAL_TEST_RUN.log`.
Missing: `uv.lock` (needs `uv` + network) and `artifacts/fresh_clone_run.json` (needs a
real `git init` + clone; this directory is still not a git repo). Neither can be
hand-written without fabricating evidence.

### 3. G8 — needs a generator that does not exist

`artifacts/final_metrics_summary.json` has no producer. `scripts/compute_metrics.py`
writes `phase05_metrics.json`, which is a different artifact. The G8 report must contain
that summary's full sha256.

### 4. G9 — gated on G0–G8 green

Required mutations now include two beyond the frozen five: **remove the worker's tamper
check** (R15) and **restore the `/grader` mount** (R16) — the latter must turn
`oracle_source_read_in_container` and `gold_oracle_import_evasive` RED.

### Addendum — Tier S specs written (still unrun)

`trgym/tasks/tier_s_specs.py` now exists: three tasks, one mutated file each, as
`TierSTaskSpec` (deliberately NOT a `RepoTaskSpec` — Tier S builds from a different
template and declares a much larger editable set, and widening the Tier M `tier` Literal
would let `get_repo_task` return something the Tier M build path cannot materialize).

| task | mutated file | predicted failing checks |
|---|---|---|
| `s1_causal_mask_offbyone` | `_ops/masking.py` (`tril(0)` → `tril(1)`) | strict_causality, matches_gold_logits |
| `s2_padding_supervised_as_labels` | `_data/collate.py` (`ignore_index` → `pad_token`) | supervised_token_count, no_pad_probability_mass, padding_does_not_change_loss |
| `s3_warmup_offbyone` | `_optim/schedule.py` (`(step+1)/w` → `step/w`) | lr_schedule_matches_gold |

The `expected_failing_checks` column is a **prediction**. `tier_s_freeze.py` must confirm
each one actually fails, and that nothing else does. The mutation `find` strings were
cross-checked by eye against `build_tier_s_template.py`; that is a static check, not a
run.

**The integration still to be decided, and deliberately not written blind.** Grading a
Tier S workspace goes `grade_workspace → run_checks → predicates → gold_repo(task_id) →
build_gold(get_repo_task(task_id))`, which resolves against the **Tier M** template. Tier
S therefore needs either a `template` field threaded through `RepoTaskSpec`/`build_repo`,
or a separate Tier S gold builder. That is surgery on the grading core which currently
passes G1 and G5 on measured evidence — writing it unverified risks more than it saves,
so it is left as an explicit decision for a session that can actually run the tests.

---

## Session 7 — G4 and G7 close. 8/10.

Auto-mode was replaced with acceptEdits, which cleared the TYPE B command block. Work
resumed from the previous RESUME HERE.

### G4 — Tier S localization (PASS)

`build_tier_s_template.py` executed for the first time. It wrote **48** files, not the 43
the previous session predicted — which is why that number was labelled unverified.

Freeze preflight (`scripts/tier_s_freeze.py`, before any paid call): 48 files, **89 import
edges, 0 orphans**, only relevant files differ, gold PASSES and no-op FAILS on all three
tasks. `frozen=true`.

The anti-padding check is a graph traversal and it earned its place immediately: its first
run flagged `_util/seeding.py` as an orphan, which turned out to be **two** bugs — the
resolver mishandled relative imports inside a package `__init__.py` (resolving
`from .seeding import …` to `tinygpt.seeding`), and `seeded_generator` genuinely was
imported-but-never-called while `_data/batching.py` hand-rolled the same generator. Edge
count went 64 → 89 after the fix.

12 precommitted trajectories, `deepseek-chat`, 24 turns:

```
relevant file located     12/12      fraction of repo read: mean 0.181, max 0.271
full fix                   9/12      structural ceiling 0.5 (1 tool call per turn)
naive (visible) pass      12/12      naive false positives 3/12
```

Localization is not the bottleneck at 48 files. **s3 is the informative failure**: all four
episodes opened `_optim/schedule.py` and three still failed — two repaired `_train/loop.py`
instead, because "first step does nothing, LR curve shifted by one" is equally consistent
with a `scheduler.step()`/`optimizer.step()` ordering bug. Opening the right file is not
reading it correctly.

**R17, found after the run and fixed without re-running.** `repo_fingerprint` hashed only
`tinygpt/*.py`, so every Tier S subpackage edit was invisible: the first pass reported
`edited_a_relevant_file 0/12` beside `full fix 9/12`, an impossible pair, and that
contradiction is what exposed it. Rewards were never affected — they come from real
grading. The patch metrics were recomputed from the graded workspaces, which still existed,
spending **zero** trajectories, with originals kept in `*_ORIGINAL_BUGGY` fields.

Budget: **24 / 30** (10 G3 + 12 G4 + 2 v1 smokes).

### G7 — packaging and fresh clone (PASS)

`git init` + two commits, MIT LICENSE (Rance), CITATION.cff, pyproject.toml, `uv.lock`,
CI workflow, FINAL_TEST_RUN.log.

**The fresh clone found a real dependency bug.** First run: `build_sandbox` failed, nine
tests failed, all with `Numpy is not available`. `numpy` was never declared — the original
`.venv` was built with `--system-site-packages` and inherited it from the system
interpreter, so everything worked locally and only broke in a clean environment. Declared
in `pyproject.toml`; the re-run passes end to end.

The clone check interrogates the clone's own interpreter for each module's `__file__` and
strips `PYTHONPATH`/`PYTHONHOME`/`VIRTUAL_ENV` from the child environment. All four `trgym`
modules resolve inside the clone; none resolve back into `E:\RL`.

`environments.transformer_repair` cannot import on Windows (`verifiers.v1` → `fcntl`) and
is recorded as `platform_limited` rather than counted as a packaging defect.

### Also this session

- `tests/test_check_surfaces_agree.py` (28 tests). R16 left **two** implementations of
  every check — `checks.py` (in-process) and `predicates.py` (what production grades
  with) — and nothing forced them to agree. Found while building the mutation cases: a
  mutation aimed at the return-type predicate would have survived because
  `test_repo_tasks.py` never reaches it.
- `POST_SUCCESS_CODE_REVIEW.md`: **G2's verifier v2 is unexercised.** `v1_v2_disagreements
  == 0` across 89 replayed trajectories — v2 never once decided differently from v1. The
  gate passes its frozen criterion (`v2_FPR <= v1_FPR`) legitimately, but no report may
  describe v2 as an improvement, and none does.
