"""Fixtures that provision a real v1 DockerRuntime.

Deliberately uses the official `verifiers.v1.runtimes` factory rather than shelling out
to `docker run`, because a hand-rolled container would let G1 pass while the official
runtime path stayed untested -- exactly the cosmetic outcome the gate forbids.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# `verifiers` is the optional `v1` extra. On Windows these tests skip anyway, because
# `verifiers.v1` imports `fcntl` -- so the missing-dependency path was never exercised
# locally and only appeared on Linux CI, where fcntl exists, the skip did not fire, and
# every fixture raised ModuleNotFoundError as a collection ERROR rather than a skip.
#
# A missing OPTIONAL dependency must skip, not fail. Skipping at module scope also keeps
# the omission visible in the pytest summary instead of silently shrinking the suite.
pytest.importorskip(
    "verifiers",
    reason="optional 'v1' extra is not installed; `uv sync --extra v1` (Linux only)",
)

TASK_ID = "m1_attention_regression"


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture
def make_runtime(event_loop):
    """Build a started DockerRuntime with the task's repo already planted."""
    from verifiers.v1.runtimes import DockerRuntime
    from verifiers.v1.runtimes.docker import DockerConfig

    created: list = []

    async def _make(task):
        runtime = DockerRuntime(
            DockerConfig(image=task.data.image),  # type: ignore[call-arg]
            name=f"trgym-{task.data.task_id}-{len(created)}",
        )
        await runtime.start()
        await runtime.prepare_setup()
        await task.setup(runtime)
        created.append(runtime)
        return runtime

    yield _make

    async def _cleanup() -> None:
        for r in created:
            try:
                await r.stop()
            except Exception:  # noqa: BLE001 - teardown must not mask a test failure
                pass

    event_loop.run_until_complete(_cleanup())


@pytest.fixture
def runtime_for_task(make_runtime, event_loop):
    from environments.transformer_repair import (
        TransformerRepairConfig,
        TransformerRepairTaskset,
    )

    cfg = TransformerRepairConfig(task_ids=[TASK_ID])
    task = next(iter(TransformerRepairTaskset(cfg)))
    runtime = event_loop.run_until_complete(make_runtime(task))
    return runtime, task


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers", "slow: needs a real Docker daemon and provisions a v1 runtime"
    )


@pytest.fixture
def make_trace():
    """Build a Trace the way `verifiers.v1.cli.validate` does.

    Trace requires a `TraceTask` and an `AgentInfo`; there is no bare
    `Trace(id=..., task="name")` form. Copied from the official CLI rather than
    guessed, so these tests exercise the same construction real runs use.
    """
    from verifiers.v1.configs.agent import AgentConfig
    from verifiers.v1.state import state_cls
    from verifiers.v1.trace import AgentInfo, Trace, TraceTask

    def _make(task, *, name: str = "test"):
        return Trace(
            task=TraceTask(type=type(task).__name__, data=task.data),
            state=state_cls(type(task))(),
            agent=AgentInfo(config=AgentConfig(), name=name, trainable=False),
        )

    return _make
