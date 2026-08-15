# VERIFIER_SCALABILITY_REPORT

G5. Two questions, and both had to be answered before either number meant anything: does
the grader isolate, and does it scale. Isolation first — a fast grader that leaks the
answer is not a grader.

Sources: `artifacts/g5_scalability.json`, `artifacts/g5_isolation_canaries.json`.
Task: `m1_attention_regression`, 6 checks per job, 30 sequential jobs per phase.

---

## 1. Isolation (the gating result)

| grader path | leaked | executed | refused |
|---|---|---|---|
| `sandboxed_container` (production) | **0 / 17** | 17 / 17 | 0 |
| `in_process_UNSAFE_CONTROL` | 12 / 17 | 17 / 17 | 0 |

Both columns are load-bearing. The control is the deliberately unsafe pre-R14 path, reached
through `allow_in_process=True`; if it contained everything, the probes would be measuring
nothing. It leaks 12 of 17, so they are live.

`executed 17/17` matters as much as `leaked 0`. Three earlier attempts at this suite each
reported clean containment while measuring nothing:

1. run on Windows, where the package cannot import — every probe "contained";
2. the unsafe control silently calling the *same* sandboxed grader as the subject;
3. R15's static denylist refusing every oracle-seeking payload before it executed, so
   `contained` meant "never ran".

The artifact now records `contained_by` per probe and
`channels_without_an_executing_probe` per column. Both are clean: no probe reads
`probe_did_not_execute`, and every channel has at least one payload that actually ran.

### What the 17 probes cover

| property | probes |
|---|---|
| `/grader` absent from the candidate container | `grader_mount_absent` |
| hidden check / oracle source unreadable | `oracle_source_read_in_container`, `host_file_read` |
| gold + template source unreadable | `gold_template_read_in_container` |
| gold unreachable by import or object-graph walk | `gold_oracle_import`, `gold_oracle_import_evasive` |
| no serialized reference bundle in the candidate's world | `reference_bundle_unreachable` |
| verdict / protocol unforgeable | `verdict_forge_protocol`(+`_evasive`), `verdict_tamper_tolerance`(+`_evasive`) |
| cross-job state isolation | `env_var_write`, `module_global_write`(+`_evasive`), `temp_dir_persistence`, `child_process`, `grader_secret` |

Malformed and oversized candidate output is covered separately by 31 host-side tests in
`tests/test_obs_protocol.py`, because those cases are cheaper to express directly against
the decoder than as container probes.

---

## 2. Throughput

30 sequential jobs per phase, wall-clock per job.

| phase | mean | p50 | p95 | min | max | stdev | failures |
|---|---|---|---|---|---|---|---|
| cold | 7.3838 s | 7.2449 s | 8.5432 s | 5.3484 s | 9.3984 s | 1.0677 | 0 |
| final | 6.9481 s | 7.0276 s | 7.8151 s | 5.4639 s | 7.9878 s | 0.6899 | 0 |
| in-process *(reference only)* | 0.6675 s | 0.6293 s | 0.7487 s | 0.5344 s | 1.5357 s | 0.1742 | 0 |

Percentiles by nearest-rank. **0 failures in 90 jobs.**

Cold and final agree to within 0.1 s, so there is no warm-up effect worth modelling and no
hidden per-run state accumulating across jobs — which is itself a small corroboration of the
cross-job isolation probes.

### The price of isolation

**6.28 s per job, a factor of 10.4** over the in-process reference. That reference is not a
candidate path and never will be: it is the pre-R14 design in which candidate code executes
in a process holding gold. It appears here only to price the boundary.

These figures were re-measured after v0.2-B. The previous run reported 5.23 s per job, and
anchoring five checks to gold added **1.7 s**: each now recomputes gold's answer on a
trusted-supplied fixture. The older number was not left in place, because it described code
that no longer exists. Some of the increase was avoidable and was removed —
`_gold_pure_outputs` is cached now, where it had been rebuilding gold and re-entering
`RepoModules` once per predicate, five times per job.

Worth recording: R16 made this *cheaper*, not dearer. The pre-R16 measurement was ~8.24 s of
container startup against 0.54 s of checks. The candidate container no longer mounts the
repository and no longer imports the grader stack, so it starts faster — an unusual case of
the more secure design also being the quicker one. The comparison is indicative rather than
controlled: the two figures come from different sessions on the same machine.

At ~5.2 s/job a 10-task sweep costs under a minute of grading. Nothing in this project's
workload is grading-bound.

---

## 3. Decision: contract Option B

Take the official isolated sandbox per grading job; ship no custom persistent grader.

The reasoning is recorded in the artifact and repeated here because it is the whole point of
the gate. R14 established that candidate-touched trees must not be graded in a process that
holds gold, and R16 established that they must not be graded in a *container* that holds it
either. A persistent grader would reintroduce exactly the shared-state surface that
`module_global_write`, `temp_dir_persistence` and `env_var_write` exist to detect, and it
would have to pass the full canary suite before it could be trusted. None is shipped, so
there is no custom optimisation to justify or to delete.

10.4× for a boundary that three separate measurement attempts were needed to verify is a
trade worth making, and 7 s is not a number that constrains anything here.

---

## 4. Reproduce

```bash
# isolation (must be Linux; on Windows the package cannot import and every probe
# reports "contained" -- the exact false-clean this suite is designed to avoid)
docker run --rm -v "e:/RL:/run/desktop/mnt/host/e/RL" -w /run/desktop/mnt/host/e/RL \
  -e PYTHONPATH="/run/desktop/mnt/host/e/RL:/run/desktop/mnt/host/e/RL/environments" \
  -v "/var/run/docker.sock:/var/run/docker.sock" -v "/tmp:/tmp" \
  trgym-v1:latest python scripts/g5_isolation_canaries.py

# throughput
  ... trgym-v1:latest python scripts/g5_scalability_bench.py --jobs 30
```

Accept only if `sandboxed_container` shows `leaked 0`, `refused 0`, every channel with an
executing probe, **and** the unsafe control still leaks. A clean control means the probes
are broken, not that the grader is safe.
