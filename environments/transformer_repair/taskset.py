"""`TransformerRepairTaskset` -- a native `verifiers.v1` Taskset.

`load()` is the subclass hook and may be a generator; `Taskset.__iter__` is the read
path and layers the config-level system prompt plus any `head`/`shuffle` view on top.
So this file implements `load()` only and never touches `__iter__`.
"""

from __future__ import annotations

from collections.abc import Iterator

from verifiers.v1.task import TaskResources, TaskTimeout
from verifiers.v1.taskset import Taskset

from .configs import TransformerRepairConfig
from .task import (
    TransformerRepairData,
    TransformerRepairTask,
)

SYSTEM_PROMPT = (
    "You are a senior ML engineer debugging a small Transformer training codebase. "
    "You investigate before you edit, and you verify your fix by running the tests "
    "that ship with the repo."
)


def _build_prompt(spec, workdir: str) -> str:
    """Symptom-only prompt. It must not name the root cause or the guilty file.

    That constraint is the whole point of the localization measurement: naming the file
    would turn a repo-scale diagnosis task into a single-file edit.
    """
    editable = "\n".join(f"  - {p}" for p in spec.editable)
    return (
        f"A user reports the following problem with the codebase at `{workdir}`:\n\n"
        f"{spec.symptom}\n\n"
        "Observed evidence:\n"
        + "\n".join(f"  - {e}" for e in spec.evidence)
        + "\n\nYou may read and edit any of these files:\n"
        + editable
        + "\n\nThe repo ships its own quick checks; run them with "
        f"`cd {workdir} && python -m pytest trgym_visible_checks.py -q` "
        "or `python trgym_visible_checks.py`.\n\n"
        "Those visible checks are NOT the grader. They are a smoke test that the "
        "buggy code already passes, so making them green proves nothing on its own -- "
        "diagnose the actual defect. Grading runs a separate hidden suite against your "
        "edited source files. Do not attempt to locate or modify the grader; it does "
        "not exist inside this container."
    )


class TransformerRepairTaskset(Taskset[TransformerRepairTask, TransformerRepairConfig]):
    def load(self) -> Iterator[TransformerRepairTask]:
        from trgym.tasks.repo_specs import _all_tasks as all_repo_tasks

        config = self.config
        allow = set(config.task_ids)
        tiers = set(config.tiers)
        resources = TaskResources(cpu=2, memory=4, disk=8)

        index = 0
        for task_id, spec in sorted(all_repo_tasks().items()):
            if allow and task_id not in allow:
                continue
            if not allow and str(spec.tier) not in tiers and spec.tier not in tiers:
                continue
            yield TransformerRepairTask(
                TransformerRepairData(
                    idx=index,
                    name=task_id,
                    description=f"{spec.family} ({spec.tier})",
                    prompt=_build_prompt(spec, config.task.workdir),
                    system_prompt=SYSTEM_PROMPT,
                    image=config.docker_image,
                    workdir=config.task.workdir,
                    resources=resources,
                    # No egress: the candidate must reason from the code in front of
                    # it, and a network would also make the task unreproducible.
                    network_block=["*"],
                    # `timeout` is a TaskTimeout model, not an int: v1 bounds each
                    # lifecycle phase separately. Setup plants ~10 files through the
                    # runtime, and scoring reads them back out and runs the hidden
                    # suite, so those two get real budgets rather than one shared one.
                    timeout=TaskTimeout(
                        setup=float(config.task.visible_timeout),
                        finalize=60.0,
                        scoring=float(config.task.grade_timeout),
                    ),
                    task_id=task_id,
                    tier=str(spec.tier),
                    family=spec.family,
                    provenance=str(spec.provenance),
                    hidden_checks=tuple(spec.hidden_checks),
                    visible_checks=tuple(spec.visible_checks),
                    editable=tuple(spec.editable),
                    requires_training_run=spec.requires_training_run,
                ),
                config.task,
            )
            index += 1


__all__ = ["TransformerRepairConfig", "TransformerRepairTask", "TransformerRepairTaskset"]
