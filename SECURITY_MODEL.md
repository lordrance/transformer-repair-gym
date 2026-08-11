# SECURITY_MODEL

What this environment defends, how, and — specifically — what it does **not** defend.

The threat here is not a malicious human attacker. It is a policy being optimised against a
reward: it will find whatever shortcut the verifier permits, without intent, simply because
that shortcut scores. Every defence below exists because a concrete exploit was constructed
and observed, not because it seemed prudent.

> **Status note (R16 — supersedes the boundary model below).** The architecture changed.
> Read this before the "Two boundaries" section, which describes the *previous* design.
>
> R14 sandboxed grading; R15 added integrity layers because candidate code still shared a
> process with the checks. Both were defending an asset that should never have been in the
> room. The whole repository was bind-mounted into the grading container at `/grader`, so
> candidate code could reach gold two ways — `gc.get_objects()` to `gold_repo()`, or simply
> `open('/grader/trgym/repo/checks.py')`. The second evades nothing; it opens a file. No
> pattern gate can answer that, and R15's tamper check cannot see it, because reading gold
> is not mutating gold.
>
> R16 removes the premise. The candidate container mounts no repository: it gets its own
> workspace, a standalone probe, and public task inputs. It computes **observations** and
> returns them over a nonced, explicitly validated channel. Gold, the hidden checks and
> every predicate live in a trusted process that never imports candidate code.
>
> **Measured**, not asserted: `artifacts/g5_isolation_canaries.json` records
> `sandboxed_container: leaked 0/17, ran 17, refused 0` against an unsafe in-process
> control that leaks 12/17. Every probe executed — nothing is "contained" because it never
> ran, which is the failure mode the previous two attempts kept producing. Gold scores 1.0
> and the buggy tree 0.0 through the new path (`artifacts/raw/v1_runtime_evidence.json`).
>
> R15's static denylist is **deleted**, not retained as defence-in-depth. It had stopped
> being load-bearing, and while it stood it refused every oracle-seeking canary before
> execution — hiding the boundary from the measurement that was supposed to verify it.

---

## The asset

One thing must stay secret: the **gold repository** — the corrected reference source for
each task — together with the hidden check implementations that compare against it. If a
policy can read gold, the task collapses from "diagnose and repair a defect" to "copy a
file", and every reward number becomes meaningless.

Secondary assets: the hidden check *implementations* (their names are public and appear in
prompts; their code must not be), and any grading-side secret in the grading process
environment.

## Two boundaries, both required

The property claimed is that the oracle is **unreachable**, not merely protected. That takes
two boundaries in opposite directions, and an earlier version of this project claimed the
property from the first alone.

### Boundary 1 — outbound: the agent cannot reach gold

The candidate works inside a `verifiers.v1` `DockerRuntime`. Gold is never written into it.

- gold is built by `build_gold()` into a **process-local temp directory** on the grading
  side and is never bind-mounted anywhere;
- `TaskData` carries hidden check *names* but no gold, no expected values and no reference
  source — it is serialised into run records, so anything in it is effectively public;
- the container gets `network_block=["*"]`, so the candidate cannot fetch the upstream
  project the task was derived from.

Verified by `tests_v1/test_v1_migration.py::test_grading_artifacts_are_not_candidate_readable`,
which greps the candidate's *entire filesystem* from inside the container for
`check_repo_matches_gold_logits`, `def gold_repo` and `build_gold`, checks that no gold
directory exists under any plausible name, and confirms `import trgym.repo.checks` fails
there. Absence, not permissions.

### Boundary 2 — inbound: candidate code cannot reach gold either

This is the boundary that was missing, and it is subtler.

Grading has to **execute** the candidate's code in order to check it. `RepoModules` puts the
workspace on `sys.path[0]` and calls `importlib.import_module("tinygpt")` — and importing a
Python module runs every top-level statement in it. If that happens in the rollout process,
candidate code executes in the one process where `gold_repo()` is importable and gold is on
disk. Three lines at module scope in `tinygpt/config.py` (editable in every task) would have
sufficed:

```python
from trgym.repo.checks import gold_repo
open("/tmp/leak", "w").write((gold_repo("m1_attention_regression") / "tinygpt" / "attention.py").read_text())
```

Boundary 1 does not stop this, because the code is not being run *by the agent inside the
container* — it is being run by the grader, on the host, on the agent's behalf.

The fix is that grading is itself sandboxed. `grade_workspace()` routes through
`trgym.harness.sandbox.run_checks(..., fallback=False)`:

```
--network=none  --cap-drop=ALL  --security-opt=no-new-privileges  --read-only
--memory=2g  --pids-limit=256  non-root uid 10001  tmpfs scratch
```

`fallback=False` matters as much as the container: `fallback=True` silently degrades to an
in-process run when Docker is missing, which would restore the vulnerability on exactly the
machines least equipped to notice.

The required data flow is therefore:

```
candidate bytes  ->  isolated grader process  ->  result
```

and never:

```
candidate bytes  ->  import into the process that owns gold
```

**The one permitted exception**, `allow_in_process=True`, exists for grading trees *this
repo built* — `validate()`'s gold preflight. It must never be used on a tree a policy has
touched. `tests/test_grading_isolation.py` enumerates every opt-in on the public path via
AST and fails if a new one appears, so widening the trusted set requires a deliberate
decision rather than an accident.

### Layer 3 — verdict integrity: a contained process is not an honest one

Boundary 2 answers "can candidate code reach the grading host?" It does not answer "can
candidate code decide its own grade?", and the G5 canaries showed the second answer was yes.

Inside the grading container the candidate and the checks share one process — unavoidably,
because grading has to execute the code it is grading — and `/grader` is mounted and on
`sys.path`. Module-level candidate code could therefore:

```python
import trgym.repo.checks as c
c.ATOL = 1e9                      # every tolerance comparison now passes
c.CHECKS["repo_strict_causality"] = lambda *a, **kw: None
```

and a tree built from the **buggy** template scored a pass through the production path. The
result channel had the same shape of weakness: the worker announced its verdict with a fixed
`<<<TRGYM_RESULT>>>` marker on the stdout it shares with the candidate.

Two independent layers answer this. Neither is called a boundary:

**1. Refuse before executing.** `grading.scan_for_grader_tampering()` reads the candidate's
editable sources as bytes — no import, so nothing it scans can subvert it — and refuses the
grade outright if they reference the grader package, the import cache (`sys.modules`), the
import machinery (`importlib`, `__import__`), builtins, or the result marker. Comments and
docstrings are stripped first (R13). The patterns are narrow by design: `compile` and `eval`
are excluded because `torch.compile` is legitimate ML code, and G6 measured the two closest
existing gates at a 0/32 false-positive rate on benign human-written ML source.

**2. Discard the verdict afterwards.** The container worker snapshots `ATOL`, `RTOL` and the
`CHECKS` registry before importing anything, re-reads them after grading, and reports the run
as `tampered` if they moved — in which case every check is recorded as failed. The result
marker is per-job (`<<<TRGYM_RESULT:{nonce}>>>`); the nonce travels on stdin rather than in
argv or the environment, both of which are readable from inside the grading process; and the
host reads the *last* correctly-nonced block, so a forged one printed at import time is
ignored and recorded as tampering.

**What this is not.** Perfect in-process integrity is not achievable while the grader must
execute the code it grades. An adversary who knows the worker's internals can still walk the
object graph. These layers raise the cost against the threat this project actually has — a
policy optimising against a reward, with no knowledge of the harness — and they are stated as
mitigation, not as a sandbox. A `sys.modules` purge plus `sys.path` removal was considered as
a third layer and rejected: several checks import `trgym.repo.build` and
`trgym.tasks.repo_specs` lazily, so purging would break gold grading to buy something the
other two layers already cover.

### Why boundary 2 was missed

Every isolation test written for G1 probes *outward* — from the container, looking for the
grader. None probed *inward*, from candidate code that the grader executes. The threat model
has two directions and the tests covered one. The general lesson, recorded because it will
recur: **an isolation claim needs a test per boundary, per direction.**

## What else is defended

**Only declared files come back.** `materialize_candidate()` reads back only
`spec.editable`. The graded tree is rebuilt from the buggy template and then overwritten
with the candidate's versions of files it was allowed to touch — so a smuggled `conftest.py`,
`sitecustomize.py` or patched `trgym_visible_checks.py` never enters the graded tree.

**The visible suite carries no ground truth.** `trgym/repo/visible_runtime.py` is planted in
the candidate's workspace and contains the five smoke checks plus their helpers, and nothing
else. It was split out of `trgym/repo/checks.py` (R9) precisely because the planted runner
used to import the module that also defines the hidden oracle.

**Grading trusts bytes, not reports.** The candidate's own test runs are never consulted.
Only the source files read out of the runtime are graded.

**Static anti-exploit gates**, each targeting an observed exploit, each returning the
offending line as evidence: reference import, dynamic execution, shape special-casing,
silent exception swallowing, hardcoded tensors, plus structural gates for public-API
preservation and grader/support-file tampering. They match against tokenized code, not
comments or docstrings (R13) — before that fix they refused 15.6 % of real PyTorch and numpy
functions for what their docstrings said.

## What is explicitly NOT defended

Stated plainly, because a security document that only lists strengths is marketing.

- **Candidate-internal predicates are forgeable (R16).** This is the important limitation of
  the new architecture and it is not hidden. Moving predicates into the trusted process
  stops the candidate *deciding* its verdict; it does not stop the candidate lying about the
  numbers it reports. A submission that fabricates its own gradient norms passes
  `repo_gradients_reach_optimizer`. The full list is `predicates.FORGEABLE` — 19 of the 23
  checks.

  What is **not** forgeable is any predicate whose ground truth is gold:
  `repo_matches_gold_logits`, `repo_supervised_token_count`, `repo_lr_schedule_matches_gold`
  and `repo_contract_public_api`. The candidate cannot know the value it would have to
  claim, because gold is absent from the container it ran in — and `repo_contract_public_api`
  never asks the candidate anything at all, since the trusted side reads the submitted
  sources itself and compares ASTs.

  The honest summary: R16 makes the *oracle* unreachable, which is what the task's integrity
  depends on. It does not make an untrusted process's self-report trustworthy, and nothing
  short of re-deriving every quantity from gold would.
- **This is not a hostile-multi-tenant sandbox.** The Docker configuration raises the cost
  of escape; it is not a claim of resistance to a determined attacker with a kernel exploit.
  The threat model is a reward-seeking policy, not an adversary.
- **No side-channel defence.** Timing, resource-exhaustion and error-message channels are
  not analysed. A policy could in principle infer something about the hidden checks from how
  long grading takes or from which check fails first — failure messages are surfaced to the
  trace by design, because opaque grading is unusable for research.
- **The check *names* are public.** They appear in `TaskData` and in prompts. A policy that
  reasons about `repo_strict_causality` from its name alone is doing legitimate work.
- **Docker-out-of-docker widens the boundary.** Running the v1 runtime from inside a
  container with the host socket mounted (`-v /var/run/docker.sock`) grants effective host
  control to that process. This is a development and CI convenience on Windows, not a
  posture to reproduce for untrusted code.
- **No supply-chain verification.** Dependencies are pinned by version, not by hash beyond
  what the lockfile provides. A compromised upstream wheel is out of scope.
- **The gold oracle is secret, not cryptographically sealed.** Anyone with filesystem
  access to the grading host can read it. The defence is process and container isolation,
  not encryption.
- **In-process verdict integrity is mitigated, not guaranteed.** See layer 3 above. The
  grading process must import the code it grades, so a submission that knows the worker's
  internals can in principle still reach them. Two layers make that expensive; neither makes
  it impossible, and no test in this repo claims otherwise.
- **Not production-hardened.** This is a research environment. Nothing here has been
  penetration-tested, and no claim of production security is made anywhere in this project.

## Reporting

This is a local research artifact with no deployment and no users. If you find a way for a
policy to reach gold, the useful action is to add it to `scripts/fuzz_verifier.py` as a
probe with an expected verdict, so it becomes a permanent regression test rather than a
one-off observation.
