"""Transformer training-code repair, on the native `verifiers.v1` lifecycle.

The public execution path. Nothing here imports the legacy `verifiers` v0 API
(`vf.Environment`, `SingleTurnEnv`, `Rubric`); that code lives in `legacy_research/`
and a test asserts the separation.

Requires Linux: `verifiers.v1` imports POSIX-only `fcntl`.
"""

from .configs import (
    TransformerRepairConfig,
    TransformerRepairTaskConfig,
)
from .task import (
    TransformerRepairData,
    TransformerRepairTask,
)
from .taskset import TransformerRepairTaskset

__all__ = [
    "TransformerRepairConfig",
    "TransformerRepairData",
    "TransformerRepairTask",
    "TransformerRepairTaskConfig",
    "TransformerRepairTaskset",
]
