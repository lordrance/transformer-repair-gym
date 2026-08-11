"""G1 acceptance: the ten frozen contract checks for the `verifiers.v1` migration.

Runs ONLY inside Linux (`verifiers.v1` imports POSIX-only `fcntl`), which is why this
lives in `tests_v1/` rather than `tests/` -- the host suite must stay green on Windows.

These are written to be *behavioural*. The frozen FAIL condition for G1 is "evaluation
still runs through a v0 path with a v1 wrapper that is not actually exercised", so a
test that merely imports a v1 symbol is worthless here. Each test below asserts
something that would break if the legacy harness were still the real engine:

  * `test_reward_is_recorded_through_trace` fails if grading bypasses `Trace`
  * `test_docker_runtime_actually_executes_candidate_code` fails if the container is
    decorative -- it reads back a file only the container could have produced
  * `test_grading_artifacts_are_not_candidate_readable` fails if the oracle is merely
    permission-protected instead of absent
  * `test_no_cross_rollout_state_leak` fails if two rollouts share a workspace
  * `test_public_path_is_free_of_legacy_verifiers_api` fails on any v0 import
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="verifiers.v1 imports POSIX-only fcntl"
)

TASK_ID = "m1_attention_regression"


# --------------------------------------------------------------------------- #
# 1. fresh install / import of the pinned version
# --------------------------------------------------------------------------- #
def test_pinned_verifiers_version_imports() -> None:
    import verifiers
    from verifiers import v1

    assert verifiers.__version__ == "0.3.0", (
        f"pinned 0.3.0, got {verifiers.__version__}; the migration was written "
        "against 0.3.0's lifecycle and must be re-verified on a bump"
    )
    # Prove it is the installed package, not a local shim shadowing the name.
    assert "site-packages" in v1.__file__, v1.__file__


# --------------------------------------------------------------------------- #
# 2. the public environment imports verifiers.v1 and subclasses its bases
# --------------------------------------------------------------------------- #
def test_public_environment_subclasses_native_v1_bases() -> None:
    from verifiers.v1.task import Task
    from verifiers.v1.taskset import Taskset

    from environments.transformer_repair import (
        TransformerRepairTask,
        TransformerRepairTaskset,
    )

    assert issubclass(TransformerRepairTaskset, Taskset)
    assert issubclass(TransformerRepairTask, Task)
    # Not a wrapper: the base must be the real installed class object.
    assert "site-packages" in inspect.getfile(Taskset)
    assert TransformerRepairTask.NEEDS_CONTAINER is True


# --------------------------------------------------------------------------- #
# 3. the native Taskset loads and enumerates tasks
# --------------------------------------------------------------------------- #
def test_taskset_loads_and_enumerates() -> None:
    from environments.transformer_repair import (
        TransformerRepairConfig,
        TransformerRepairTaskset,
    )

    ts = TransformerRepairTaskset(TransformerRepairConfig())
    tasks = list(ts)
    assert len(tasks) >= 10, f"expected the 10 repo tasks, enumerated {len(tasks)}"

    ids = [t.data.task_id for t in tasks]
    assert len(set(ids)) == len(ids), "duplicate task ids"
    assert TASK_ID in ids

    # Exercise the official view machinery rather than reimplementing slicing.
    assert len(list(ts.head(3))) == 3
    assert {t.data.task_id for t in ts.head(3).shuffle(seed=0)} <= set(ids)

    # The config-layer system prompt is applied by Taskset.__iter__, not by load().
    assert all(t.data.system_prompt for t in tasks)


# --------------------------------------------------------------------------- #
# 4. TaskData carries no gold, and mutating it copies rather than aliases
# --------------------------------------------------------------------------- #
def test_taskdata_is_copy_on_write_and_carries_no_gold() -> None:
    from environments.transformer_repair import (
        TransformerRepairConfig,
        TransformerRepairTaskset,
    )

    task = next(iter(TransformerRepairTaskset(TransformerRepairConfig())))
    clone = task.with_system_prompt("replaced")
    assert clone.data.system_prompt == "replaced"
    assert task.data.system_prompt != "replaced", "with_system_prompt aliased the data"

    # TaskData is serialized into run records, so gold in it would be public.
    blob = task.data.model_dump_json()
    for leak in ("def check_repo_", "gold_repo", "matches_gold_logits_expected"):
        assert leak not in blob, f"TaskData leaks {leak!r}"


# --------------------------------------------------------------------------- #
# 5. the Docker Runtime actually executes the candidate's code
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_docker_runtime_actually_executes_candidate_code(runtime_for_task) -> None:
    runtime, task = runtime_for_task

    async def go() -> None:
        # A value only the container's own interpreter can produce.
        probe = await runtime.run(
            ["python", "-c", "import sys, platform;"
             "print(sys.executable, platform.system(), 6*7)"], {})
        assert probe.exit_code == 0, probe.stderr
        assert "Linux" in probe.stdout and "42" in probe.stdout

        # The planted repo must import and its visible checks must run IN there.
        wd = task.config.workdir
        res = await runtime.run(
            ["bash", "-lc", f"cd {wd} && python trgym_visible_checks.py && echo VISIBLE_OK"],
            {},
        )
        assert "VISIBLE_OK" in res.stdout, (res.stdout, res.stderr)

        # And the candidate's edits must be what grading sees. Write a sentinel
        # through the runtime and prove it comes back out.
        await runtime.write(f"{wd}/tinygpt/config.py",
                           b"# sentinel-edit\nBLOCK_SIZE = 64\n")
        back = await runtime.read(f"{wd}/tinygpt/config.py")
        assert b"sentinel-edit" in back

    asyncio.get_event_loop().run_until_complete(go())


# --------------------------------------------------------------------------- #
# 6. reward reaches Trace, with weights, through the base Task.score
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_reward_is_recorded_through_trace(runtime_for_task, make_trace) -> None:
    from verifiers.v1.trace import Trace

    runtime, task = runtime_for_task
    trace = make_trace(task, name="reward")

    # Never overridden: this must be the base implementation doing discovery +
    # weighting. Overriding `score` is the easiest way to accidentally build the
    # cosmetic wrapper G1 forbids, so assert the inherited function object itself.
    from verifiers.v1.task import Task as _BaseTask

    assert "score" not in vars(type(task)), "Task.score must not be overridden"
    assert type(task).score is _BaseTask.score

    asyncio.get_event_loop().run_until_complete(task.score(trace, runtime))

    assert "semantic_repair" in trace.rewards, f"rewards={dict(trace.rewards)}"
    assert "files_changed" in trace.metrics, f"metrics={dict(trace.metrics)}"
    assert "touched_the_defective_files" in trace.metrics

    # `trace.rewards[k]` is a Reward(score, weight), not a bare float -- so the
    # @reward(weight=1.0) declaration is verifiably what reached the Trace.
    entry = trace.rewards["semantic_repair"]
    assert entry.weight == 1.0, f"weight not propagated: {entry!r}"

    # The buggy repo is planted, so the hidden suite must fail => reward 0.
    assert entry.score == 0.0

    # Grading provenance recorded, and recorded as host-side.
    assert trace.info["grading_side"] == "host"
    assert trace.info["hidden_grading"]["n_total"] >= 1

    # An incomplete trace must still be GRADED, not short-circuited to 0.0. `Trace.ok`
    # defaults False, so a `has_error` early return would score every rollout zero and
    # relabel infra failures as capability failures (R5/R10). Prove grading ran anyway:
    # this trace has `ok is False` yet carries a real grading record.
    assert trace.has_error, "fixture trace is incomplete (ok defaults False)"
    assert trace.info["hidden_grading"]["n_total"] >= 1, "grading was short-circuited"

    # And `infra_error` must be a real discriminator, not a constant. It reads
    # `trace.errors`, so a clean-but-unfinished rollout is 0.0 -- the first version read
    # `has_error` and returned 1.0 for *every* rollout, which is useless (R12).
    # Metrics are plain floats; only rewards are wrapped in Reward(score, weight).
    assert float(trace.metrics["infra_error"]) == 0.0, (
        "a rollout with no recorded errors must not be flagged as an infra failure"
    )
    assert float(trace.metrics["hit_turn_limit"]) == 0.0


@pytest.mark.slow
def test_infra_error_metric_actually_fires(runtime_for_task, make_trace) -> None:
    """The INFRA/CAPABILITY discriminator is worthless if it cannot go to 1.0.

    R12's first version was a constant 1.0; a constant 0.0 would be just as broken and
    the happy-path assertion above would not notice. Record a real error and a real
    turn-limit stop, then re-score.
    """
    from verifiers.v1.trace import Error

    runtime, task = runtime_for_task
    trace = make_trace(task, name="errored")
    trace.errors.append(Error(type="ProviderError", message="simulated 429"))
    trace.stop_condition = "max_turns"

    asyncio.get_event_loop().run_until_complete(task.score(trace, runtime))

    assert float(trace.metrics["infra_error"]) == 1.0, "infra_error never fires"
    assert float(trace.metrics["hit_turn_limit"]) == 1.0, "hit_turn_limit never fires"

    # Crucially, the reward is STILL computed on the workspace rather than zeroed by
    # the presence of an error -- that separation is the whole point of R10.
    assert "semantic_repair" in trace.rewards
    assert trace.info["hidden_grading"]["n_total"] >= 1


@pytest.mark.slow
def test_gold_scores_one_and_buggy_scores_zero(runtime_for_task, make_trace) -> None:
    """The reward must actually separate a correct repair from the planted defect."""
    import tempfile

    from trgym.repo.build import build_gold
    from trgym.tasks.repo_specs import get_repo_task

    runtime, task = runtime_for_task
    spec = get_repo_task(task.data.task_id)

    async def go() -> float:
        gold = Path(tempfile.mkdtemp(prefix="v1_gold_"))
        build_gold(spec, gold)
        wd = task.config.workdir
        for rel in spec.editable:
            src = gold / rel
            if src.exists():
                await runtime.write(f"{wd}/{rel}", src.read_bytes())
        trace = make_trace(task, name="gold")
        await task.score(trace, runtime)
        return float(trace.rewards["semantic_repair"].score)

    assert asyncio.get_event_loop().run_until_complete(go()) == 1.0, (
        "gold sources must score 1.0; if this fails the reward is not measuring repair"
    )


# --------------------------------------------------------------------------- #
# 7. grading artifacts are UNREACHABLE from the candidate, not merely protected
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_grading_artifacts_are_not_candidate_readable(runtime_for_task) -> None:
    runtime, task = runtime_for_task

    async def go() -> None:
        # Search the candidate's whole filesystem for anything grading-shaped.
        hunt = await runtime.run(
            ["bash", "-lc",
             "grep -rl --binary-files=without-match "
             "-e 'check_repo_matches_gold_logits' -e 'def gold_repo' "
             "-e 'build_gold' / 2>/dev/null | head -20; echo HUNT_DONE"], {})
        found = [ln for ln in hunt.stdout.splitlines()
                 if ln and ln != "HUNT_DONE"]
        assert not found, f"grading code reachable from the candidate: {found}"

        # The gold tree must not exist under any obvious name either.
        listing = await runtime.run(
            ["bash", "-lc",
             "ls -d /gold /workspace/gold /opt/gold 2>/dev/null; echo LS_DONE"], {})
        assert listing.stdout.strip() == "LS_DONE", listing.stdout

        # The hidden check names are labels and may appear in the prompt, but the
        # implementations must not: assert no *callable* is importable in there.
        imp = await runtime.run(
            ["python", "-c",
             "import importlib;\n"
             "try:\n"
             "    importlib.import_module('trgym.repo.checks');\n"
             "    print('IMPORTABLE')\n"
             "except Exception as e:\n"
             "    print('ABSENT')"], {})
        assert "ABSENT" in imp.stdout, "the hidden check module is importable in the candidate"

    asyncio.get_event_loop().run_until_complete(go())


# --------------------------------------------------------------------------- #
# 8. setup / finalize / cleanup lifecycle
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_setup_finalize_cleanup_lifecycle(runtime_for_task, make_trace) -> None:
    runtime, task = runtime_for_task
    trace = make_trace(task, name="life")

    async def go() -> None:
        # setup already ran in the fixture; prove it planted a full tree.
        listing = await runtime.run(
            ["bash", "-lc", f"cd {task.config.workdir} && ls tinygpt/ | sort"], {})
        for expected in ("attention.py", "model.py", "train.py", "config.py"):
            assert expected in listing.stdout, listing.stdout

        await task.finalize(trace, runtime)   # must be callable and not raise
        assert await runtime.alive()

    asyncio.get_event_loop().run_until_complete(go())


@pytest.mark.slow
def test_validate_preflights_gold(runtime_for_task) -> None:
    """`validate` must confirm gold passes; it is the guard against a broken task."""
    runtime, task = runtime_for_task
    ok = asyncio.get_event_loop().run_until_complete(task.validate(runtime))
    assert ok is True, "validate() failed: gold does not pass its own hidden suite"


# --------------------------------------------------------------------------- #
# 9. no cross-rollout state leakage
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_no_cross_rollout_state_leak(make_runtime) -> None:
    """An edit in rollout A must be invisible to rollout B.

    Covers the file, env-var and module-global channels in one pass: a second runtime
    must see the pristine planted tree, no sentinel env var, and no leftover
    /tmp scratch from the first.
    """
    from environments.transformer_repair import (
        TransformerRepairConfig,
        TransformerRepairTaskset,
    )

    cfg = TransformerRepairConfig(task_ids=[TASK_ID])
    task = next(iter(TransformerRepairTaskset(cfg)))

    async def go() -> None:
        a = await make_runtime(task)
        b = await make_runtime(task)
        wd = task.config.workdir
        try:
            await a.run(["bash", "-lc",
                         f"echo LEAK_A > {wd}/tinygpt/leaked.py; "
                         "echo LEAK_A > /tmp/leak_marker; "
                         "export TRGYM_LEAK=A"], {})

            listing = await b.run(
                ["bash", "-lc",
                 f"ls {wd}/tinygpt/leaked.py 2>/dev/null; "
                 "cat /tmp/leak_marker 2>/dev/null; "
                 "echo \"env=${TRGYM_LEAK:-unset}\"; echo B_DONE"], {})
            assert "LEAK_A" not in listing.stdout, f"state leaked: {listing.stdout}"
            assert "leaked.py" not in listing.stdout, f"file leaked: {listing.stdout}"
            assert "env=unset" in listing.stdout, f"env leaked: {listing.stdout}"
        finally:
            for r in (a, b):
                await r.stop()

    asyncio.get_event_loop().run_until_complete(go())


# --------------------------------------------------------------------------- #
# 10a. the official v1 CLI can drive the taskset (dry run, no model)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_official_v1_cli_validate_dry_run(tmp_path) -> None:
    """`validate` is the model-free v1 entry point: it resolves the taskset as a
    plugin, provisions a Runtime, runs `setup`, then `validate`, and writes a summary.

    This is the check that the environment is reachable the way a *user* reaches it,
    not only the way our tests import it. It exercised a real packaging bug: the
    taskset id resolves by importing a top-level module named after it, so the package
    had to work under both `environments.transformer_repair` and `transformer_repair`
    (hence relative imports internally), and list-valued config fields had to become
    lists because pydantic_config cannot build a tuple from a CLI scalar.
    """
    import json
    import os
    import subprocess

    out = tmp_path / "valout"
    env = {**os.environ, "PYTHONPATH": f"{REPO_ROOT}:{REPO_ROOT / 'environments'}"}
    proc = subprocess.run(
        ["validate", "transformer_repair", "--only-gold", "-o", str(out),
         "@", "environments/transformer_repair/configs/m1_smoke.toml"],
        cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, timeout=1800,
    )
    summary_path = out / "summary.json"
    assert summary_path.exists(), (
        "the CLI produced no summary.\nstdout:\n"
        + proc.stdout[-3000:]
        + "\nstderr:\n"
        + proc.stderr[-3000:]
    )
    summary = json.loads(summary_path.read_text())
    assert summary["outcomes"]["valid"] == 1, summary
    assert summary["outcomes"]["error"] == 0, summary
    assert summary["valid_rate"] == 1.0, summary

    # And the CLI must have gone through OUR task class, not a fallback.
    rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines() if l]
    assert rows and rows[0]["reason"] == "valid", rows


# --------------------------------------------------------------------------- #
# 10. no legacy v0 API anywhere on the public path
# --------------------------------------------------------------------------- #
PUBLIC_PATH = ("environments",)
LEGACY_MARKERS = ("vf.Environment", "SingleTurnEnv", "MultiTurnEnv", "vf.Rubric", "Rubric(")


def _code_only(text: str) -> str:
    """`text` with comments and string literals removed.

    A raw substring scan flags this module's own docstring, which names the v0 API in
    order to say it is banned. Prose that mentions `vf.Environment` is documentation;
    only executable references are violations.
    """
    import io
    import tokenize

    kept: list[str] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            kept.append(tok.string)
    except tokenize.TokenError:
        return text  # a file we cannot tokenize is scanned conservatively
    return " ".join(kept)


def test_public_path_is_free_of_legacy_verifiers_api() -> None:
    offenders: list[str] = []
    for pkg in PUBLIC_PATH:
        for py in sorted((REPO_ROOT / pkg).rglob("*.py")):
            text = py.read_text(encoding="utf-8")
            rel = py.relative_to(REPO_ROOT).as_posix()
            code = _code_only(text)
            for marker in LEGACY_MARKERS:
                if marker in code:
                    offenders.append(f"{rel}: {marker}")
            # `import verifiers as vf` is the v0 entry point; v1 must be imported
            # as `verifiers.v1...`. Detect via AST so a comment cannot trip it.
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "verifiers":
                            offenders.append(f"{rel}: import verifiers as {alias.asname}")
                if isinstance(node, ast.ImportFrom) and node.module == "verifiers":
                    names = ", ".join(a.name for a in node.names)
                    if not all(a.name.startswith("v1") for a in node.names):
                        offenders.append(f"{rel}: from verifiers import {names}")
    assert not offenders, "legacy v0 API on the public path: " + "; ".join(offenders)


def test_the_legacy_detector_actually_detects() -> None:
    """A ban that cannot fire is decoration. The moved v0 file is the positive control.

    `legacy_research/transformer_repair_v0.py` is genuine `SingleTurnEnv`/`Rubric` code,
    so the detector must flag it. If this passes while the public-path test also passes,
    the separation is real rather than an artifact of a broken scan.
    """
    legacy = REPO_ROOT / "legacy_research" / "transformer_repair_v0.py"
    assert legacy.exists(), "the v0 environment should be preserved, not deleted"

    code = _code_only(legacy.read_text(encoding="utf-8"))
    hits = [m for m in LEGACY_MARKERS if m in code]
    assert hits, "detector failed to flag known-legacy code; the scan is broken"

    # And it must not fire on this file's own prose about the ban.
    prose = _code_only('"""We forbid vf.Environment and SingleTurnEnv."""\nx = 1\n')
    assert not any(m in prose for m in LEGACY_MARKERS), (
        "detector flags documentation; it must inspect code only"
    )


def test_public_path_imports_v1_submodules_explicitly() -> None:
    """The environment must reach into `verifiers.v1.*`, which is the v1 contract."""
    seen = set()
    for py in sorted((REPO_ROOT / "environments").rglob("*.py")):
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                "verifiers.v1"
            ):
                seen.add(node.module)
    for required in (
        "verifiers.v1.task",
        "verifiers.v1.taskset",
        "verifiers.v1.trace",
        "verifiers.v1.runtimes",
        "verifiers.v1.utils.decorators",
        "verifiers.v1.configs.task",
        "verifiers.v1.configs.taskset",
    ):
        assert required in seen, f"public path never imports {required}; saw {sorted(seen)}"
