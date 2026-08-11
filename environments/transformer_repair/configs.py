"""Config layer for the transformer-repair taskset.

`TasksetConfig`/`TaskConfig` are the official v1 config bases (pydantic_config
`BaseConfig` underneath), so these are what `vf-eval`/`validate` bind CLI overrides
onto. Modelled on `verifiers.v1.tasksets.lean.taskset`, which is the shipped
reference for a container-graded taskset.
"""

from __future__ import annotations

from pydantic import Field

from verifiers.v1.configs.task import TaskConfig
from verifiers.v1.configs.taskset import TasksetConfig

# Built by docker/Dockerfile.v1. Carries torch so the candidate can run the visible
# tests itself; carries NO grading code (see grading.py for why that matters).
DEFAULT_DOCKER_IMAGE = "trgym-v1:latest"

REPO_WORKDIR = "/workspace/repo"


class TransformerRepairTaskConfig(TaskConfig):
    workdir: str = REPO_WORKDIR
    """Where the candidate repo is planted inside the runtime."""

    visible_timeout: int = 300
    """Per-invocation bound on the candidate's own visible test runs."""

    grade_timeout: int = 600
    """Bound on host-side hidden grading for one candidate."""

    use_contract_layer: bool = True
    """Grade with the v2 suite (hidden + L1 contract checks) rather than v1.

    Defaults to the hardened suite: G2 froze v2 as the verifier of record, and it was
    proven not to reduce false positives by over-rejecting.
    """


class TransformerRepairConfig(TasksetConfig):
    docker_image: str = DEFAULT_DOCKER_IMAGE

    # `list`, not `tuple`: these are CLI-facing, and pydantic_config cannot build a
    # tuple from a bare `--taskset.task_ids m1_attention_regression`. A config the
    # official CLI cannot express is not really on the v1 path.
    tiers: list[str] = Field(default_factory=lambda: ["M", "H"])
    """Which difficulty tiers to enumerate. Tier E is legacy single-file work."""

    task_ids: list[str] = Field(default_factory=list)
    """Explicit allow-list; empty means every task in `tiers`."""

    task: TransformerRepairTaskConfig = TransformerRepairTaskConfig()
