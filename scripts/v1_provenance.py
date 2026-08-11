"""Record which `verifiers.v1` modules actually execute during a scored rollout.

G1's frozen FAIL condition is "evaluation still runs through a v0 path with a v1 wrapper
that is not actually exercised", and importability alone cannot distinguish the two. So
this drives a real lifecycle -- taskset resolved through the OFFICIAL plugin loader, a
real `DockerRuntime`, `setup`, `validate`, then `score` -- and records, for every object
on that path, the file the code was loaded from.

The load-bearing assertions are:
  * every v1 base class resolves under site-packages (installed, not a local shim)
  * `Task.score` is the INHERITED function object, so reward discovery is v1's
  * the reward/metric functions carry v1's `@reward`/`@metric` markers
  * no v0 *environment* module is ever imported (see V0_EXECUTION_MARKERS below for why
    that is the only v0 claim this can honestly make)

Writes artifacts/raw/v1_provenance.json. Run inside Linux with the docker socket.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, "/work")
sys.path.insert(0, "/work/environments")

OUT = Path("/work/artifacts/raw/v1_provenance.json")
TASK_ID = "m1_attention_regression"

# Only `verifiers.envs.*` is meaningful evidence. `verifiers/__init__.py` lines 29-36
# EAGERLY import `Parser` and `Rubric`, so a bare `import verifiers` -- which any
# `verifiers.v1` import triggers -- always loads 8 v0 modules. Asserting their absence
# would be testing the library's packaging, not our code, and would fail forever.
#
# `verifiers.envs` is different: it is a LAZY import map (`"SingleTurnEnv":
# "verifiers.envs.singleturn_env:SingleTurnEnv"`), so those modules load only if
# something actually reaches for a v0 environment. Their absence is therefore real
# evidence that the v0 execution path was never entered.
V0_EXECUTION_MARKERS = ("verifiers.envs",)
V0_FORCED_BY_LIBRARY = ("verifiers.parsers", "verifiers.rubrics")


def where(obj) -> str:
    try:
        return inspect.getfile(obj)
    except (TypeError, OSError):
        mod = sys.modules.get(getattr(obj, "__module__", "") or "")
        return getattr(mod, "__file__", "<unknown>")


async def main() -> None:
    import verifiers
    from verifiers.v1.runtimes import DockerRuntime
    from verifiers.v1.runtimes.docker import DockerConfig
    from verifiers.v1.task import Task as BaseTask
    from verifiers.v1.taskset import Taskset as BaseTaskset
    from verifiers.v1.trace import Trace
    from verifiers.v1.utils.decorators import discover_decorated
    from verifiers.v1.utils.loaders import load_taskset, taskset_config_type

    rec: dict = {"verifiers_version": verifiers.__version__}

    # Resolve through the official plugin loader, exactly as the CLI does -- not by
    # importing our package directly, which would prove less.
    cfg_type = taskset_config_type("transformer_repair")
    taskset = load_taskset(cfg_type(id="transformer_repair", task_ids=[TASK_ID]))
    tasks = list(taskset)
    task = tasks[0]

    rec["plugin_resolution"] = {
        "taskset_id": "transformer_repair",
        "resolved_class": f"{type(taskset).__module__}.{type(taskset).__name__}",
        "resolved_from": where(type(taskset)),
        "config_type": cfg_type.__name__,
        "n_tasks_enumerated": len(tasks),
    }

    rec["v1_bases"] = {
        name: {"module": cls.__module__, "file": where(cls),
               "under_site_packages": "site-packages" in where(cls)}
        for name, cls in {
            "Taskset": BaseTaskset, "Task": BaseTask, "Trace": Trace,
            "DockerRuntime": DockerRuntime,
        }.items()
    }

    # `score` must be v1's own function object. If our class defined its own, reward
    # discovery/weighting would be ours and the v1 lifecycle would be decorative.
    rec["score_is_inherited"] = {
        "overridden_in_subclass": "score" in vars(type(task)),
        "is_base_function": type(task).score is BaseTask.score,
        "score_defined_in": where(BaseTask.score),
    }

    # The signals v1 will discover, and the markers it discovers them by.
    rewards = discover_decorated(task, "reward")
    metrics = discover_decorated(task, "metric")
    rec["discovered_signals"] = {
        "rewards": [{"name": f.__name__, "weight": getattr(f, "_vf_weight", None),
                     "marked": getattr(f, "reward", False)} for f in rewards],
        "metrics": [{"name": f.__name__, "marked": getattr(f, "metric", False)}
                    for f in metrics],
        "discovered_by": where(discover_decorated),
    }

    # ---- a real scored rollout on a real container -------------------------------
    runtime = DockerRuntime(DockerConfig(image=task.data.image), name="provenance")
    await runtime.start()
    await runtime.prepare_setup()
    await task.setup(runtime)
    gold_ok = await task.validate(runtime)

    from verifiers.v1.configs.agent import AgentConfig
    from verifiers.v1.state import state_cls
    from verifiers.v1.trace import AgentInfo, TraceTask

    trace = Trace(
        task=TraceTask(type=type(task).__name__, data=task.data),
        state=state_cls(type(task))(),
        agent=AgentInfo(config=AgentConfig(), name="provenance", trainable=False),
    )
    await task.score(trace, runtime)

    rec["scored_rollout"] = {
        "runtime_type": runtime.type,
        "runtime_class_file": where(type(runtime)),
        "gold_validate_passed": gold_ok,
        "rewards": {k: {"score": v.score, "weight": v.weight}
                    for k, v in trace.rewards.items()},
        "metrics": dict(trace.metrics),
        "grading_side": trace.info.get("grading_side"),
        "suite": trace.info.get("suite"),
        "hidden_grading": trace.info.get("hidden_grading"),
    }
    await runtime.stop()

    # ---- the negative half: the v0 EXECUTION path was never entered ---------------
    loaded_v0 = sorted(m for m in sys.modules if m.startswith(V0_EXECUTION_MARKERS))
    rec["legacy_v0_env_modules_loaded"] = loaded_v0
    rec["v0_modules_forced_by_library"] = sorted(
        m for m in sys.modules if m.startswith(V0_FORCED_BY_LIBRARY)
    )
    rec["v1_modules_loaded"] = sorted(
        m for m in sys.modules if m.startswith("verifiers.v1")
    )[:60]
    rec["n_v1_modules_loaded"] = sum(
        1 for m in sys.modules if m.startswith("verifiers.v1")
    )

    rec["verdict"] = {
        "all_bases_installed": all(
            b["under_site_packages"] for b in rec["v1_bases"].values()
        ),
        "score_not_overridden": not rec["score_is_inherited"]["overridden_in_subclass"],
        # No v0 ENVIRONMENT was ever constructed. Lazily imported, so absence is real.
        "no_v0_execution_path": not loaded_v0,
        "reward_discovered_by_v1": bool(rec["discovered_signals"]["rewards"]),
        "buggy_scores_zero": all(
            v["score"] == 0.0 for v in rec["scored_rollout"]["rewards"].values()
        ),
    }
    rec["verdict"]["PASS"] = all(rec["verdict"].values())

    OUT.write_text(json.dumps(rec, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(rec["verdict"], indent=2))
    print(f"\nwrote {OUT}")
    print(f"v1 modules loaded: {rec['n_v1_modules_loaded']}, v0 modules: {loaded_v0}")


if __name__ == "__main__":
    asyncio.run(main())
