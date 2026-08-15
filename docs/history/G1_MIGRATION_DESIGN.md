# G1 — `verifiers.v1` migration design

Input of record: `artifacts/raw/v1_probe.json` (verifiers 0.3.0, probed inside
`python:3.12-slim`; the original capture stream is `artifacts/raw/v1_probe.raw.txt`).

Status: **design grounded in the real API surface. Not implemented. G1 remains FAIL.**

---

## The finding that changes the plan

I had assumed the migration was a re-housing: my `Budget` + `tools.py` + `session.py`
loop would sit behind a v1 `Harness`, with the sandbox behind a `Runtime`. The real
signature says otherwise:

```
Harness.launch(self, ctx: ModelContext, trace: Trace, runtime: Runtime,
               endpoint: str, secret: str, mcp_urls: dict[str, str],
               data: TaskData) -> ProgramResult
```

`launch` receives an **`endpoint` and a `secret`** and returns a `ProgramResult`
(`exit_code`, `stdout`, `stderr`). That is not a host-side agent loop calling tools into
a sandbox. It is the inverse: v1 starts an agent *program inside the runtime*, hands it
a model endpoint to call back out to, and collects its exit status. The presence of an
`ACP` class with `run(runtime, env, command, prompt, *, mcp_urls, system_prompt,
session_path, session_meta, allow_empty_tool_reply, ...)` confirms the intended shape —
an agent-client-protocol process, with MCP as the tool transport.

My architecture is the opposite polarity. `trgym/harness/session.py` runs the loop on
the host: it owns the turn counter, calls `policy.act(obs)`, and dispatches
`list_files` / `read_file` / `run_command` / `apply_patch` against a workspace path.
The model never runs inside the sandbox; the sandbox only ever executes graded code.

**So G1 is a polarity inversion, not a rename** — which is exactly the class of thing a
cosmetic wrapper would hide, and the reason the user forbade one.

## Two honest options

**Option 1 — custom `Harness` subclass whose `launch` runs the existing loop.**
`Harness` is an ABC; a subclass can implement `launch` to drive `run_episode` directly
and ignore `endpoint`/`secret`. Cheap, and every v1 object would be genuinely
constructed and genuinely on the execution path.

Risk: it satisfies the *type* contract while bypassing the mechanism v1 exists to
provide. If `launch` never dials `endpoint`, no MCP toolset is exercised and the ACP
path is dead code, then this is a wrapper wearing a subclass. The G1 criterion is that
official execution *actually uses* the v1 lifecycle — so this needs a hard test that
distinguishes "v1 objects were instantiated" from "v1 drove the episode", or it fails
on the same grounds a wrapper would.

**Option 2 — expose the four repair tools as an MCP toolset and let v1 drive.**
`Task.toolsets(config) -> list[Toolset]` and the `mcp_urls` plumbing say this is the
intended design. The agent runs in the runtime; `list_files`/`read_file`/`run_command`/
`apply_patch` become MCP tools; grading stays host-side in `Task.score(trace, runtime)`
writing through `Trace.record_reward`.

Cost: the DeepSeek adapter stops being a policy object and becomes an endpoint the
container calls. Budget enforcement moves from my `Budget` dataclass into whatever v1
offers (`runtimes.limiters` — the `fcntl` importer — is likely where). Real work, and
the reason G1 was always the largest gate.

**Preference: Option 2**, because Option 1's failure mode is precisely the one the
acceptance criterion was written to catch, and because discovering that `limiters` is
where budgets live would explain why v1 is POSIX-only rather than treating `fcntl` as
an incidental portability bug.

## What must be settled before writing code

The signatures do not answer these, and guessing them is what the "study official
examples/tests first" instruction forbids:

1. How does a concrete `Taskset.load()` yield tasks — construct `Task(data, config)` per
   item, or return data and let the caller wrap?
2. Does `Task.score` return a value or only mutate `Trace`? (`-> 'None'`, so
   `Trace.record_reward` is the channel — but the `@vf.reward`-decorated-method
   relationship to `score` is unresolved.)
3. What does `@vf.reward` decorate — a `Task` method, or a free function registered
   elsewhere? `weight`/`priority` imply an aggregation the probe does not show.
4. Where do turn and wall-clock budgets live? Confirm `runtimes.limiters`.
5. Is there a built-in `Taskset`/`Task` implementation in the package to read as a
   reference?

**Next action:** extract the installed package's own `v1` examples/tests source out of
the container and read a concrete `Taskset`/`Task` pair end to end. Signature probing
has reached its limit — it produced the polarity finding, but it cannot answer 1–5.

## Non-negotiables carried into implementation

- Development, tests and evaluation run **inside Linux/Docker**. Windows cannot
  `import verifiers.v1` (`fcntl`); confirmed twice, TYPE B.
- The gold oracle stays **unreachable**, not merely protected: built in a process-local
  temp dir, never mounted into the runtime. This constraint survives the migration
  unchanged and constrains where `Task.score` may run.
- Legacy `import verifiers as vf` / `vf.Environment` / `SingleTurnEnv` / `Rubric` move
  to `legacy_research/` and leave the public path.
- One live DeepSeek smoke must prove **in logs** that it traversed v1 — logging
  `module.__file__` for each v1 object actually used, not merely that a v1 symbol was
  importable.
