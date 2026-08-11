# REPRODUCIBILITY

How to re-run this project's verification from scratch, and — separately — which of these
steps have actually been executed and which have not.

Required by `FINAL_FLAGSHIP_COMPLETION_CONTRACT.md` (G7).

> **Status.** The commands below are the ones this project really runs; each was executed at
> least once and the notes record what it produced. **The fresh-clone reproduction (§5) has
> not been performed yet**, so G7 is not closed. This document is the procedure plus its
> current execution status, not a claim that G7 passed.

---

## 0. Platform facts that are not optional

- **`verifiers.v1` cannot import on Windows.** `verifiers/v1/runtimes/limiters.py` imports
  POSIX-only `fcntl`. Every v1 result in this repository comes from the Linux image. This is
  a hard constraint, not a preference.
- **Grading spawns a container.** Since R14, candidate-touched trees are graded inside
  `trgym-sandbox:latest`. When the *caller* is itself a container (as it is for all v1 work),
  that is docker-out-of-docker, and mount paths must resolve on the **daemon's** side.
- Host used for every measurement: Windows 11 + Docker Desktop 27.0.3, Python 3.12.4 in
  `.venv`, containers on `python:3.12-slim`.

## 1. Images

```bash
# the Linux rollout/grading host (verifiers 0.3.0 + torch 2.5.1 CPU + docker client)
MSYS_NO_PATHCONV=1 docker build -f docker/Dockerfile.v1 -t trgym-v1:latest docker

# the locked-down grading sandbox (torch 2.9.1 CPU, non-root uid 10001, no network)
./.venv/Scripts/python.exe scripts/build_sandbox.py
```

`trgym-v1:latest` is reclaimed by Docker Desktop fairly often. Rebuilding takes ~2 minutes
and is not a symptom of anything; there is no `rmi` anywhere in v1's docker runtime.

## 2. The container invocation, and why it looks like that

```bash
MSYS_NO_PATHCONV=1 docker run --rm \
  -v "e:/RL:/run/desktop/mnt/host/e/RL" \
  -w /run/desktop/mnt/host/e/RL \
  -e PYTHONPATH="/run/desktop/mnt/host/e/RL:/run/desktop/mnt/host/e/RL/environments" \
  -v "/var/run/docker.sock:/var/run/docker.sock" \
  -v "/tmp:/tmp" \
  trgym-v1:latest <command>
```

Three details are load-bearing, each learned from a failure:

| flag | why |
|---|---|
| repo mounted at `/run/desktop/mnt/host/e/RL` | grading runs `docker run -v {REPO_ROOT}:/grader:ro` from *inside* this container. A conventional `/work` mount makes `REPO_ROOT` a path that exists only in the caller, and the daemon silently creates an empty `/work`. Mounting at Docker Desktop's own host path makes the string valid on both sides. |
| `-v /var/run/docker.sock` | v1's `DockerRuntime` shells out to the docker CLI. **This grants effective host control to the container** — a CI convenience, not a posture for untrusted code (`SECURITY_MODEL.md`). |
| `-v /tmp:/tmp` | v1's docker runtime bind-mounts a `/tmp/vf-proxy-*` path for its egress proxy listener, which must resolve identically on both sides. |

## 3. Verification, in dependency order

```bash
# host regression (Windows) -- v1 tests correctly skip here
./.venv/Scripts/python.exe -m pytest -q

# R14/R15 grading-isolation regression: static, no Docker, runs on any platform
./.venv/Scripts/python.exe -m pytest tests/test_grading_isolation.py -q

# G0: re-freeze and re-derive every headline metric
./.venv/Scripts/python.exe scripts/freeze_historical_artifacts.py

# v1 behaviour + G1 evidence (Linux; use the §2 preamble)
  trgym-v1:latest python -m pytest tests_v1/test_v1_migration.py -q
  trgym-v1:latest python scripts/v1_runtime_evidence.py

# G5 isolation + throughput (Linux; use the §2 preamble)
  trgym-v1:latest python scripts/g5_isolation_canaries.py
  trgym-v1:latest python scripts/g5_scalability_bench.py --jobs 30

# the gate verdict, computed from artifacts only
./.venv/Scripts/python.exe scripts/final_acceptance.py
```

**The canary suite must run in Linux.** On Windows `environments.transformer_repair` cannot
import at all (`fcntl`), every probe fails before grading happens, and the run reports a clean
"contained" for all of them. A green result from that command on Windows is meaningless —
this is recorded because it happened, in PROTOCOL_CHANGELOG R14 (cont.).

## 4. Reading the canary output

Three conditions, all required. Two of the three are about the *suite*, not the code:

| | |
|---|---|
| `in_process_UNSAFE_CONTROL` leaks ≥5 probes | a control that contains everything means the probes are broken |
| `sandboxed_container` leaks 0 | the actual security property |
| no probe reports `contained_by: probe_did_not_execute` | a probe that never ran contained nothing |

## 5. Fresh-clone reproduction — NOT YET PERFORMED

The remaining G7 requirement. It must copy the tree to a clean directory and prove imports
come from there:

- no reuse of `E:\RL\.venv`, no `PYTHONPATH` pointing at the original tree, no stale
  `__pycache__`, `.pytest_cache` or `.sandbox_work*`;
- record `module.__file__` for `trgym`, `environments.transformer_repair` and `verifiers`,
  and assert none resolves inside the original working tree;
- run exactly the commands in §3 — the commands in this file and the ones executed must be
  the same strings, which is what `final_acceptance.py` checks via
  `artifacts/fresh_clone_run.json`.

## 6. Claude Code permission note

Several commands above are, by construction, adversarial: the canaries read a process secret,
import a hidden oracle, spawn child processes and attempt to forge the grader's result
protocol. A session that accumulates enough of that content can hit a conversation-content
safety refusal that is *not* a permission rule and cannot be cleared by one. Diagnosis and the
narrow allow-rule set are in `artifacts/claude_permission_diagnosis.json`. The rules are
scoped to these exact scripts and this exact container invocation; do not widen them to
`Bash(python *)` or `Bash(docker *)`.
