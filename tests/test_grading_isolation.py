"""R14 regression: candidate code must never execute in the process that owns gold.

The defect these tests exist to prevent: `grade_workspace` used to call
`run_repo_checks` directly, which puts the candidate's workspace on `sys.path[0]` and
`importlib.import_module("tinygpt")`. Importing runs every module-level statement the
candidate wrote — inside the one process where `trgym.repo.checks.gold_repo()` is
importable and the gold tree is on disk. Three lines at module scope in an editable file
could have read gold, or set `ATOL = 1e9` on the already-imported grader module.

G1's existing isolation tests all probe *outward* (from the container, looking for the
grader) and so none of them could see this. The threat model has two directions and the
tests covered one. These cover the other.

Deliberately split into two kinds:

  * **static** (`test_*_static`) — AST checks over the source. They need no Docker, no
    container and no candidate execution, so they run everywhere including CI on Windows,
    and they cannot be defeated by a mock.
  * **behavioural** (marked `needs_docker`) — actually grade a workspace and assert the
    work happened in a container.

See PROTOCOL_CHANGELOG R14.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GRADING_PY = REPO_ROOT / "environments" / "transformer_repair" / "grading.py"
TASK_PY = REPO_ROOT / "environments" / "transformer_repair" / "task.py"
SANDBOX_PY = REPO_ROOT / "trgym" / "harness" / "sandbox.py"


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _load_grading():
    """Import `grading.py` without importing its package.

    `environments/transformer_repair/__init__.py` pulls in `task.py` -> `verifiers.v1` ->
    POSIX-only `fcntl`, so the ordinary import dies on Windows. `grading.py` itself has no
    v1 dependency, and these tests exercise one pure function in it, so loading the file
    directly keeps them running on every platform -- which is the point of this file.
    """
    import importlib.util
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location("_trgym_grading_under_test", GRADING_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _function(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


# --------------------------------------------------------------------------- #
# The default must be sandboxed
# --------------------------------------------------------------------------- #
def test_grade_workspace_defaults_to_sandboxed_static() -> None:
    """`allow_in_process` must exist and must default to False.

    A default of True would silently restore the vulnerability for every caller that
    does not think about it, which is exactly how the original defect shipped.
    """
    fn = _function(_module_ast(GRADING_PY), "grade_workspace")
    kwonly = {a.arg: d for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults)}
    assert "allow_in_process" in kwonly, (
        "grade_workspace must take an explicit keyword-only trusted-tree opt-in"
    )
    default = kwonly["allow_in_process"]
    assert isinstance(default, ast.Constant) and default.value is False, (
        f"allow_in_process must default to False, got {ast.unparse(default)}"
    )


def test_untrusted_path_uses_the_sandbox_with_no_fallback_static() -> None:
    """The sandboxed branch must call `run_checks(..., fallback=False)`.

    `fallback=True` silently degrades to an in-process run when Docker is missing, which
    would reintroduce the defect on exactly the machines least able to notice.
    """
    src = GRADING_PY.read_text(encoding="utf-8")
    tree = _module_ast(GRADING_PY)
    fn = _function(tree, "grade_workspace")

    calls = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", getattr(n.func, "attr", "")) == "run_checks"
    ]
    assert calls, "grade_workspace must call trgym.harness.sandbox.run_checks"
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert "fallback" in kw, "run_checks must be called with an explicit fallback"
        assert isinstance(kw["fallback"], ast.Constant) and kw["fallback"].value is False, (
            "fallback must be False: a silent in-process fallback is the vulnerability"
        )
    assert "run_repo_checks" not in ast.unparse(fn), (
        "grade_workspace must not call run_repo_checks directly; that is the in-process path"
    )


def test_in_process_grading_is_reachable_only_behind_the_flag_static() -> None:
    """`_grade_in_process` must be private and called only from the guarded branch."""
    tree = _module_ast(GRADING_PY)
    assert any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == "_grade_in_process"
        for n in ast.walk(tree)
    ), "the in-process path should be a separate, underscore-private function"

    # Nothing except grade_workspace may call it.
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name in ("grade_workspace", "_grade_in_process"):
            continue
        assert "_grade_in_process" not in ast.unparse(node), (
            f"{node.name} bypasses the sandbox by calling _grade_in_process"
        )


# --------------------------------------------------------------------------- #
# The reward path must never opt out
# --------------------------------------------------------------------------- #
def test_reward_path_never_grades_in_process_static() -> None:
    """`semantic_repair` grades policy-touched bytes and must never set the flag.

    This is the assertion that actually protects the security property: `validate`'s gold
    preflight is a legitimate in-process caller, so the flag cannot simply be banned
    repo-wide. It has to be banned *on the candidate path specifically*.
    """
    tree = _module_ast(TASK_PY)
    reward_fn = _function(tree, "semantic_repair")
    assert "allow_in_process" not in ast.unparse(reward_fn), (
        "semantic_repair must grade candidate bytes in the sandbox, never in-process"
    )


def test_only_gold_preflight_opts_into_in_process_static() -> None:
    """Enumerate every `allow_in_process=True` on the public path and pin the callers.

    A new opt-in should fail this test and force a deliberate decision, rather than
    quietly widening the trusted set.
    """
    permitted = {"validate"}
    offenders: list[str] = []

    for py in sorted((REPO_ROOT / "environments").rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for call in ast.walk(fn):
                if not isinstance(call, ast.Call):
                    continue
                for kw in call.keywords:
                    if kw.arg != "allow_in_process":
                        continue
                    if isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        if fn.name not in permitted:
                            offenders.append(
                                f"{py.relative_to(REPO_ROOT).as_posix()}::{fn.name}"
                            )
    assert not offenders, (
        "allow_in_process=True outside the permitted gold-preflight callers: "
        + ", ".join(sorted(set(offenders)))
    )


def test_the_static_detector_actually_detects() -> None:
    """A guard that cannot fail is decoration. Prove each detector fires.

    Same discipline as the rmtree and legacy-API guards: construct the defect and require
    the check to see it.
    """
    # A grade_workspace whose default is True.
    bad_default = (
        "def grade_workspace(ws, tid, checks, *, allow_in_process=True):\n"
        "    return None\n"
    )
    fn = _function(ast.parse(bad_default), "grade_workspace")
    kwonly = {a.arg: d for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults)}
    assert kwonly["allow_in_process"].value is True, "detector cannot read the default"

    # A reward that opts out of the sandbox.
    bad_reward = (
        "async def semantic_repair(self, trace, runtime):\n"
        "    return grade_workspace(w, t, c, allow_in_process=True)\n"
    )
    reward_fn = _function(ast.parse(bad_reward), "semantic_repair")
    assert "allow_in_process" in ast.unparse(reward_fn), (
        "detector would not notice a reward grading in-process"
    )

    # A fallback=True sandbox call.
    bad_fallback = "def grade_workspace(a):\n    return run_checks(a, b, c, fallback=True)\n"
    gw = _function(ast.parse(bad_fallback), "grade_workspace")
    call = next(
        n for n in ast.walk(gw)
        if isinstance(n, ast.Call)
        and getattr(n.func, "id", getattr(n.func, "attr", "")) == "run_checks"
    )
    kw = {k.arg: k.value for k in call.keywords}
    assert kw["fallback"].value is True, "detector cannot read fallback"


# --------------------------------------------------------------------------- #
# R16 -- the oracle is not in the room the candidate runs in
# --------------------------------------------------------------------------- #
# R14 sandboxed grading; R15 added integrity layers because candidate code still shared a
# process with the checks. The G5 canaries then showed the remaining hole was simpler than
# either: the whole repository was bind-mounted at /grader, so candidate code could read
# `trgym/repo/checks.py` outright, or walk the object graph to `gold_repo()`. No pattern
# gate can fix that. These pin the structural answer -- the candidate container contains
# no oracle at all, and every verdict is decided outside it.
PROBE_PY = SANDBOX_PY.parents[1] / "repo" / "candidate_probe.py"
PREDICATES_PY = SANDBOX_PY.parents[1] / "repo" / "predicates.py"
OBS_PROTOCOL_PY = SANDBOX_PY.parents[1] / "repo" / "obs_protocol.py"


def test_candidate_container_mounts_no_repository_static() -> None:
    """The container that runs candidate code must not mount the repo, at any path."""
    src = SANDBOX_PY.read_text(encoding="utf-8")
    fn = _function(_module_ast(SANDBOX_PY), "run_checks_containerized")
    body = ast.unparse(fn)

    assert "/grader" not in body, (
        "the candidate container must not mount the repository: /grader is what let "
        "candidate code read the hidden checks and reach gold (R16)"
    )
    assert "REPO_ROOT" not in body, (
        "no repository path may reach the candidate container's argv"
    )
    assert '"-v", f"{staging}:/probe:ro"' in src, (
        "the probe must be staged alone and mounted read-only"
    )
    assert '"-v", f"{workspace}:/workspace:rw"' in src, (
        "the workspace remains the only writable mount"
    )
    for flag in ("--network=none", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                 "--read-only"):
        assert flag in body, f"{flag} must survive the R16 refactor"


def test_probe_never_imports_the_grader_static() -> None:
    """The candidate-side probe must be self-contained.

    If it imported `trgym` it would need the repository mounted, which is exactly the
    thing R16 removed. The duplication of `RepoModules` inside the probe is the price.
    """
    tree = ast.parse(PROBE_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("trgym"), (
                    f"the probe imports {alias.name}; it must not depend on the repo"
                )
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("trgym"), (
                f"the probe imports from {node.module}; it must not depend on the repo"
            )


def test_probe_returns_observations_and_never_a_verdict_static() -> None:
    """Nothing the probe emits may be a pass/fail decision."""
    src = PROBE_PY.read_text(encoding="utf-8")
    assert '"observations"' in src and '"errors"' in src
    assert '"passed"' not in src, (
        "the probe must not emit a verdict field; predicates live in the trusted process"
    )
    assert "CheckFailure" not in src, (
        "the probe must not decide check outcomes"
    )


def test_trusted_comparator_never_imports_candidate_code_static() -> None:
    """The comparator may open gold. It must never import the candidate's package.

    `RepoModules` executes every top-level statement in the tree it is pointed at, so
    pointing it at the workspace would put candidate code in the one process that holds
    the oracle -- the original R14 defect, reintroduced from the other side.
    """
    src = PREDICATES_PY.read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            if name != "RepoModules":
                continue
            arg = ast.unparse(node.args[0]) if node.args else ""
            assert "gold_repo" in arg, (
                f"RepoModules({arg}) in the trusted comparator: only gold may be imported "
                "here, never the candidate's workspace"
            )


def test_tampered_verdict_is_discarded_static() -> None:
    """A run flagged `tampered` must produce no passing check."""
    fn = _function(_module_ast(GRADING_PY), "grade_workspace")
    body = ast.unparse(fn)
    assert "result.tampered" in body, (
        "grade_workspace must consult SandboxResult.tampered; an unchecked flag is decoration"
    )


def test_observation_marker_is_per_job_and_read_from_the_end_static() -> None:
    """The observation channel must be nonced, and the last block must win.

    Candidate code shares stdout with the probe. A fixed marker read with `find` lets a
    submission print its own block *before* the real one and be believed.
    """
    src = SANDBOX_PY.read_text(encoding="utf-8")
    protocol = OBS_PROTOCOL_PY.read_text(encoding="utf-8")
    assert 'MARKER = "<<<TRGYM_OBS"' in src, "the marker must be a prefix completed per job"
    assert "secrets.token_hex" in src, "the per-job marker must be unguessable"
    assert "text.rfind(marker)" in protocol, (
        "the decoder must take the LAST nonced block; `find` believes a candidate's forgery"
    )
    assert '"nonce": nonce' in src, (
        "the nonce must travel on stdin: argv and the environment are readable from inside "
        "the candidate container"
    )


def test_untrusted_payload_is_never_unpickled_static() -> None:
    """Candidate bytes must not reach an unpickler.

    `torch.load` is an unpickler underneath. Decoding tensors with `frombuffer` after an
    explicit dtype/shape/length check is the difference between parsing data and running
    whatever the candidate serialised.
    """
    protocol = OBS_PROTOCOL_PY.read_text(encoding="utf-8")
    tree = ast.parse(protocol)
    banned = {"load", "loads"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in banned:
                owner = ast.unparse(func.value)
                assert owner != "torch", "torch.load must never see candidate output"
                assert owner != "pickle", "pickle must never see candidate output"
    assert "frombuffer" in protocol, "tensors must be decoded as raw buffers"


def test_the_pattern_denylist_is_gone(tmp_path: Path) -> None:
    """The static tamper gate must not come back.

    It was deleted in R16, for two reasons worth keeping written down. It stopped being
    load-bearing once the candidate container held no oracle to reach for. And while it
    stood it *hid* the boundary from measurement: every canary aimed at the oracle was
    refused before it executed, so the suite scored "contained" for probes that never
    ran. Re-adding a denylist would restore that blind spot, so this fails if it returns.
    """
    grading = _load_grading()
    assert not hasattr(grading, "scan_for_grader_tampering"), (
        "the pattern denylist is gone; absence of the oracle is the defence now"
    )
    assert not hasattr(grading, "_TAMPER_PATTERNS")

    src = GRADING_PY.read_text(encoding="utf-8")
    fn = _function(_module_ast(GRADING_PY), "grade_workspace")
    assert "refused to grade" not in ast.unparse(fn), (
        "grade_workspace must not refuse trees on a source pattern"
    )
    assert "re.compile" not in src, "no regex gate over candidate sources"


# --------------------------------------------------------------------------- #
# Behavioural: the grade really happens in a container
# --------------------------------------------------------------------------- #
@pytest.mark.needs_docker
def test_untrusted_grade_runs_in_a_container() -> None:
    """Grade a template-built tree and assert the backend was the sandbox, not this process.

    `SandboxResult.backend` records which path ran, so this distinguishes "a container was
    available" from "a container was used".
    """
    import tempfile

    from trgym.harness.sandbox import docker_available, image_exists, run_checks
    from trgym.repo.build import build_gold
    from trgym.repo.verifier_v2 import v2_checks
    from trgym.tasks.repo_specs import get_repo_task

    if not (docker_available() and image_exists()):
        pytest.skip("docker sandbox image unavailable; build with scripts/build_sandbox.py")

    spec = get_repo_task("m1_attention_regression")
    ws = Path(tempfile.mkdtemp(prefix="r14_backend_"))
    build_gold(spec, ws)
    result = run_checks(ws, spec.task_id, v2_checks(spec), fallback=False)
    assert result.backend != "in_process", (
        f"grading ran in-process (backend={result.backend!r})"
    )
    assert all(ok for _, ok, _ in result.results), "gold must pass in the sandbox too"


@pytest.mark.needs_docker
def test_candidate_module_level_code_cannot_reach_gold() -> None:
    """The end-to-end property, stated as a test rather than as prose.

    A candidate that executes `import trgym.repo.checks` at module scope must not be able
    to obtain the gold sources. Under the sandbox the module is not present at all, so the
    import fails inside the container and the check simply fails — the candidate learns
    nothing about gold.
    """
    import tempfile

    from trgym.harness.sandbox import docker_available, image_exists, run_checks
    from trgym.repo.build import build_repo
    from trgym.repo.verifier_v2 import v2_checks
    from trgym.tasks.repo_specs import get_repo_task

    if not (docker_available() and image_exists()):
        pytest.skip("docker sandbox image unavailable")

    spec = get_repo_task("m1_attention_regression")
    ws = Path(tempfile.mkdtemp(prefix="r14_reach_"))
    build_repo(spec, ws)

    probe = ws / "tinygpt" / "config.py"
    probe.write_text(
        probe.read_text(encoding="utf-8")
        + "\n\n# R14 reachability probe: must fail inside the sandbox.\n"
        "import pathlib as _p\n"
        "try:\n"
        "    from trgym.repo.checks import gold_repo as _g\n"
        "    _p.Path('/tmp/R14_GOLD_REACHED').write_text(str(_g('m1_attention_regression')))\n"
        "except Exception:\n"
        "    pass\n",
        encoding="utf-8",
    )

    run_checks(ws, spec.task_id, v2_checks(spec), fallback=False)

    # The marker is written inside the container's tmpfs, which is destroyed with it. The
    # load-bearing assertion is that nothing appeared on the HOST.
    assert not Path("/tmp/R14_GOLD_REACHED").exists()
    assert not (Path(tempfile.gettempdir()) / "R14_GOLD_REACHED").exists(), (
        "candidate module-level code reached the host filesystem during grading"
    )
