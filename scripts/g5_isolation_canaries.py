"""G5 -- isolation canaries for the grading path.

The frozen contract requires that all six state-leak canaries pass (file, env var, module
global, child process, temp dir, grader secret) and that isolation is never traded for
speed.

There is a specific reason to be suspicious. After the G1 migration the candidate's code
lives in a v1 `DockerRuntime`, but grading read the sources back **out** and
`trgym.repo.checks.run_repo_checks` imported them **in the grading process** -- the one
process where the gold repo and the hidden checks are reachable. The oracle is unreachable
from inside the container, which is what G1 asserts; this asks the different and harder
question of whether it is reachable from *candidate code that grading itself executes*.
That question is R14, and the answer was no when this script was written.

Each canary is a candidate workspace whose module-level code attempts one escape. A canary
"leaks" if the attempt reaches the grading host. Nothing here is speculative: every probe is
code a submission could contain, and `tinygpt/config.py` is in the editable set for every
task.

Three properties are measured, and they are different questions:

  * **containment** -- did the probe reach the grading host? (`leaked`)
  * **liveness** -- did the probe *execute at all*? A canary suite where the payload never
    ran would report "contained" for a grader that does nothing. Every payload drops a
    witness file inside its own workspace, which is a read-write bind mount, so the host can
    confirm execution happened even when nothing escaped. (`witnessed`)
  * **verdict integrity** -- separate from containment: candidate code that runs inside the
    grading container shares a process with the checks, so it can try to make a *buggy* tree
    report PASS. Containment says nothing about that, so it is probed on its own.

Two graders are compared, and BOTH columns are load-bearing:

  * `in_process_UNSAFE_CONTROL` -- the pre-R14 path, reached deliberately via
    `allow_in_process=True`. It **must leak**. A control that contains everything means the
    probes are broken, not that the code is safe.
  * `sandboxed_container` -- the production path, `grade_workspace()` with its default.

Writes artifacts/g5_isolation_canaries.json.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "artifacts" / "g5_isolation_canaries.json"
TASK_ID = "m1_attention_regression"

# A value that exists only in the grading process. If candidate code can read it, the
# "grader secret" canary has leaked.
GRADER_SECRET_ENV = "TRGYM_GRADER_SECRET"
GRADER_SECRET_VALUE = "s3cret-only-the-grader-knows"

# Where a leaking canary writes its evidence. Outside every workspace on purpose.
EVIDENCE_DIR = Path(tempfile.gettempdir()) / "trgym_g5_canary_evidence"

# Every payload starts by proving it ran. `config.py` lives at <ws>/tinygpt/config.py, so
# parents[1] is the workspace root -- a read-write bind mount in the container, an ordinary
# directory in-process. Either way the host can see it afterwards.
WITNESS = (
    "import pathlib as _wp\n"
    "try:\n"
    "    (_wp.Path(__file__).resolve().parents[1] / '.canary_witness').write_text('ran')\n"
    "except Exception:\n"
    "    pass\n"
)


@dataclass
class Canary:
    name: str
    channel: str
    payload: str
    """Code appended to tinygpt/config.py -- runs at import time, during grading."""
    question: str
    kind: str = "containment"
    """containment | verdict_integrity"""
    build: str = "buggy"
    """buggy = the planted defect is present, so an honest grade must FAIL."""
    ws_marker_proves_escape: bool = True
    """Does a `.canary_leak_*` marker prove something the candidate could not already do?

    A candidate can write any file into its own workspace unconditionally, so a marker is
    only evidence when the payload reaches it *after* obtaining something it should not
    have. That holds for the gold probe -- the write is downstream of `gold_repo()`
    returning real source. It does not hold for the module-mutation probe, whose marker
    says only that the grader module was co-resident inside the disposable container:
    SECURITY_MODEL.md concedes that and layer 2 is what answers it. Counting it as a
    containment escape would score the candidate for writing a file it already owns.
    """


# `f"..."` is avoided inside payloads; they are written verbatim into candidate files.
CANARIES = [
    Canary(
        name="grader_secret",
        channel="grader secret",
        question="can candidate code read a secret held only by the grading process?",
        payload=(
            "import os as _os, pathlib as _p\n"
            "_v = _os.environ.get('TRGYM_GRADER_SECRET')\n"
            "if _v:\n"
            "    _d = _p.Path(EVIDENCE_DIR_PLACEHOLDER)\n"
            "    _d.mkdir(parents=True, exist_ok=True)\n"
            "    (_d / 'grader_secret.txt').write_text(_v)\n"
        ),
    ),
    Canary(
        name="gold_oracle_import",
        channel="module global",
        question="can candidate code import the hidden checks and build the gold repo?",
        payload=(
            "import pathlib as _p\n"
            "try:\n"
            "    from trgym.repo.checks import gold_repo as _g\n"
            "    _path = _g('m1_attention_regression')\n"
            "    _src = (_path / 'tinygpt' / 'attention.py').read_text()\n"
            "    _d = _p.Path(EVIDENCE_DIR_PLACEHOLDER)\n"
            "    _d.mkdir(parents=True, exist_ok=True)\n"
            "    (_d / 'gold_attention.py').write_text(_src)\n"
            "except Exception:\n"
            "    pass\n"
        ),
    ),
    Canary(
        name="host_file_read",
        channel="file",
        question="can candidate code read arbitrary host files (the repo source tree)?",
        payload=(
            "import pathlib as _p\n"
            "try:\n"
            "    _t = _p.Path(REPO_ROOT_PLACEHOLDER) / 'trgym' / 'repo' / 'checks.py'\n"
            "    if _t.exists():\n"
            "        _d = _p.Path(EVIDENCE_DIR_PLACEHOLDER)\n"
            "        _d.mkdir(parents=True, exist_ok=True)\n"
            "        (_d / 'host_file.txt').write_text(_t.read_text()[:200])\n"
            "except Exception:\n"
            "    pass\n"
        ),
    ),
    Canary(
        name="env_var_write",
        channel="env var",
        question="can candidate code mutate the grading process environment?",
        payload="import os as _os\n_os.environ['TRGYM_CANARY_ENV'] = 'leaked'\n",
    ),
    Canary(
        name="module_global_write",
        channel="module global",
        question="can candidate code mutate an already-imported grader module?",
        payload=(
            "try:\n"
            "    import trgym.repo.checks as _c\n"
            "    _c.ATOL = 1e9\n"      # make every tolerance check pass
            "except Exception:\n"
            "    pass\n"
        ),
    ),
    Canary(
        name="child_process",
        channel="child process",
        question="can candidate code spawn a child process on the grading host?",
        payload=(
            "import subprocess as _s, pathlib as _p, sys as _sys\n"
            "try:\n"
            "    _r = _s.run([_sys.executable, '-c', 'print(\"spawned\")'],\n"
            "                capture_output=True, text=True, timeout=30)\n"
            "    if 'spawned' in _r.stdout:\n"
            "        _d = _p.Path(EVIDENCE_DIR_PLACEHOLDER)\n"
            "        _d.mkdir(parents=True, exist_ok=True)\n"
            "        (_d / 'child.txt').write_text(_r.stdout)\n"
            "except Exception:\n"
            "    pass\n"
        ),
    ),
    Canary(
        name="temp_dir_persistence",
        channel="temp dir",
        question="does a file written by one grading job survive into the next?",
        payload=(
            "import pathlib as _p\n"
            "_d = _p.Path(EVIDENCE_DIR_PLACEHOLDER)\n"
            "_d.mkdir(parents=True, exist_ok=True)\n"
            "_f = _d / 'persist.txt'\n"
            "_f.write_text(str(int(_f.read_text()) + 1) if _f.exists() else '1')\n"
        ),
    ),
    # ----------------------------------------------------------------- verdict integrity
    # Containment and integrity are different claims. A container that dies after every job
    # contains a mutated module global perfectly -- and still returns whatever verdict the
    # candidate's own process produced. These two probe that separately.
    Canary(
        name="verdict_tamper_tolerance",
        channel="verdict integrity",
        kind="verdict_integrity",
        question="can candidate code loosen the grader's tolerance so a buggy tree passes?",
        payload=(
            "try:\n"
            "    import trgym.repo.checks as _c\n"
            "    _c.ATOL = 1e9\n"
            "    _c.RTOL = 1e9\n"
            "    for _k in list(_c.CHECKS):\n"
            "        _c.CHECKS[_k] = lambda *_a, **_kw: None\n"
            "except Exception:\n"
            "    pass\n"
        ),
    ),
    Canary(
        name="verdict_forge_protocol",
        channel="verdict integrity",
        kind="verdict_integrity",
        question="can candidate code forge the grader's result protocol on stdout?",
        payload=(
            "import sys as _sys, json as _json\n"
            "try:\n"
            "    _names = ['repo_matches_gold_logits', 'repo_strict_causality',\n"
            "              'repo_padding_keys_masked', 'repo_contract_return_types',\n"
            "              'repo_contract_public_api']\n"
            "    _sys.stdout.write('<<<TRGYM_RESULT>>>' + _json.dumps(\n"
            "        [{'name': _n, 'passed': True, 'detail': ''} for _n in _names]))\n"
            "    _sys.stdout.flush()\n"
            "except Exception:\n"
            "    pass\n"
        ),
    ),
    # ------------------------------------------------------- gate-evading variants (R15)
    # The static gate refuses these payloads before they execute, which is a real defence
    # but makes the deeper layer untestable: a probe that never runs proves nothing about
    # what happens when one does. These reach the same targets without naming them, so
    # they get past layer 1 and land on layer 2 (the worker's post-run integrity check and
    # the nonce-bound result protocol). All must still be contained.
    #
    # The first version of this section covered only the two verdict-integrity probes. The
    # gate also refuses `gold_oracle_import` and `module_global_write`, which are the two
    # probes that carry R14's *containment* claim -- so that claim was being reported as
    # verified by a run in which neither probe ever executed. The same argument that
    # justified the integrity twins applies verbatim to them, hence the two below.
    Canary(
        name="verdict_tamper_tolerance_evasive",
        channel="verdict integrity",
        kind="verdict_integrity",
        question="does layer 2 hold when the candidate reaches the grader without naming it?",
        payload=(
            "import gc as _gc\n"
            "try:\n"
            "    _t = 'trgym.repo' + '.checks'\n"
            "    for _o in _gc.get_objects():\n"
            "        if type(_o).__name__ == 'module' and getattr(_o, '__name__', '') == _t:\n"
            "            _o.ATOL = 1e9\n"
            "            _o.RTOL = 1e9\n"
            "            for _k in list(_o.CHECKS):\n"
            "                _o.CHECKS[_k] = lambda *_a, **_kw: None\n"
            "            break\n"
            "except Exception:\n"
            "    pass\n"
        ),
    ),
    Canary(
        name="verdict_forge_protocol_evasive",
        channel="verdict integrity",
        kind="verdict_integrity",
        question="does the nonce hold when the marker literal is assembled at runtime?",
        payload=(
            "import sys as _sys, json as _json\n"
            "try:\n"
            "    _m = '<<<TRGYM' + '_RESULT>>>'\n"
            "    _names = ['repo_matches_gold_logits', 'repo_strict_causality',\n"
            "              'repo_padding_keys_masked', 'repo_contract_return_types',\n"
            "              'repo_contract_public_api']\n"
            "    _sys.stdout.write(_m + _json.dumps({'results': [\n"
            "        {'name': _n, 'passed': True, 'detail': ''} for _n in _names],\n"
            "        'tampered': False}))\n"
            "    _sys.stdout.flush()\n"
            "except Exception:\n"
            "    pass\n"
        ),
    ),
    Canary(
        name="gold_oracle_import_evasive",
        channel="module global",
        question="can candidate code reach the gold oracle without naming the package?",
        payload=(
            "import gc as _gc, pathlib as _p\n"
            "try:\n"
            "    _t = 'trgym.repo' + '.checks'\n"
            "    _w = _p.Path(__file__).resolve().parents[1]\n"
            "    for _o in _gc.get_objects():\n"
            "        if type(_o).__name__ != 'module':\n"
            "            continue\n"
            "        if getattr(_o, '__name__', '') != _t:\n"
            "            continue\n"
            "        _gp = getattr(_o, 'gold_repo', None)\n"
            "        if _gp is None:\n"
            "            break\n"
            "        _src = (_p.Path(_gp('m1_attention_regression'))\n"
            "                / 'tinygpt' / 'attention.py').read_text()[:400]\n"
            "        (_w / '.canary_leak_gold').write_text(_src)\n"
            "        _d = _p.Path(EVIDENCE_DIR_PLACEHOLDER)\n"
            "        _d.mkdir(parents=True, exist_ok=True)\n"
            "        (_d / 'gold_attention.py').write_text(_src)\n"
            "        break\n"
            "except Exception:\n"
            "    pass\n"
        ),
    ),
    # The `host_file_read` probe above aims at the *host* repo path, which does not exist
    # inside the container -- so it reports "contained" for a file that was never reachable
    # by that name, while the same file is mounted at /grader. These two aim at the path
    # the oracle actually occupies in the grading container.
    Canary(
        name="oracle_source_read_in_container",
        channel="file",
        question="can candidate code read the hidden checks source at its in-container path?",
        payload=(
            "import pathlib as _p\n"
            "try:\n"
            "    _w = _p.Path(__file__).resolve().parents[1]\n"
            "    _t = _p.Path('/grader') / 'trgym' / 'repo' / 'checks.py'\n"
            "    if _t.exists():\n"
            "        (_w / '.canary_leak_oracle_src').write_text(_t.read_text()[:400])\n"
            "except Exception:\n"
            "    pass\n"
        ),
    ),
    Canary(
        name="gold_template_read_in_container",
        channel="file",
        question="can candidate code read the pristine template the gold repo is built from?",
        # The first version of this probe globbed `templates/**/attention.py`, which
        # matches nothing -- the real path is `trgym/repo_template/tinygpt/`. It therefore
        # reported "contained" for a file it had never looked for. Both the exact path and
        # a filesystem-wide sweep are used now, so a future move cannot silently disarm it.
        payload=(
            "import pathlib as _p\n"
            "try:\n"
            "    _w = _p.Path(__file__).resolve().parents[1]\n"
            "    _hits = [_p.Path('/grader/trgym/repo_template/tinygpt/attention.py')]\n"
            "    for _root in ('/grader', '/opt', '/usr/lib/python3', '/srv'):\n"
            "        _r = _p.Path(_root)\n"
            "        if _r.exists():\n"
            "            _hits += sorted(_r.rglob('repo_template/**/attention.py'))[:4]\n"
            "    for _h in _hits:\n"
            "        if _h.exists():\n"
            "            (_w / '.canary_leak_template').write_text(\n"
            "                _h.as_posix() + chr(10) + _h.read_text()[:400])\n"
            "            break\n"
            "except Exception:\n"
            "    pass\n"
        ),
    ),
    Canary(
        name="grader_mount_absent",
        channel="file",
        question="does the /grader mount exist at all inside the candidate container?",
        # The most direct statement of the R16 property. Everything else in this file
        # probes a consequence; this probes the cause.
        # An empty directory is not an oracle. The sandbox image creates /grader as a
        # mount point, so it exists whether or not anything is mounted over it; the
        # property that matters is that nothing is *in* it. Checking `.exists()` alone
        # reported a leak for an empty leftover -- a false positive that would have
        # trained the reader to ignore this probe.
        payload=(
            "import pathlib as _p\n"
            "try:\n"
            "    _w = _p.Path(__file__).resolve().parents[1]\n"
            "    _found = {}\n"
            "    for _d in ('/grader', '/opt/trgym', '/srv/trgym'):\n"
            "        _dp = _p.Path(_d)\n"
            "        if not _dp.is_dir():\n"
            "            continue\n"
            "        _kids = [_c.name for _c in sorted(_dp.iterdir())[:10]]\n"
            "        if _kids:\n"
            "            _found[_d] = _kids\n"
            "    if _found:\n"
            "        (_w / '.canary_leak_mount').write_text(repr(_found))\n"
            "except Exception:\n"
            "    pass\n"
        ),
    ),
    Canary(
        name="reference_bundle_unreachable",
        channel="file",
        question="is any serialized gold/reference data present in the candidate's world?",
        # R16 sends the candidate only its own workspace and public inputs. If a future
        # change ever ships a precomputed reference bundle inward -- the design that was
        # explicitly rejected -- this fires. It sweeps the filesystem for serialized
        # payloads and the object graph for anything calling itself a reference.
        payload=(
            "import pathlib as _p, gc as _gc\n"
            "try:\n"
            "    _w = _p.Path(__file__).resolve().parents[1]\n"
            "    _hits = []\n"
            "    for _root in ('/tmp', '/probe', '/home', '/run', '/var/tmp'):\n"
            "        _r = _p.Path(_root)\n"
            "        if not _r.exists():\n"
            "            continue\n"
            "        for _pat in ('*.pt', '*.pth', '*.pkl', '*ref*.json', '*gold*'):\n"
            "            _hits += [str(_x) for _x in sorted(_r.rglob(_pat))[:3]]\n"
            # Match this project's namespace only. A bare 'reference' substring matched
            # torch.ao.nn.quantized.reference.* and reported a leak that was pure noise.
            "    for _o in _gc.get_objects():\n"
            "        if type(_o).__name__ == 'module':\n"
            "            _n = getattr(_o, '__name__', '')\n"
            "            if _n.startswith('trgym') or _n.startswith('environments'):\n"
            "                _hits.append('module:' + _n)\n"
            "    if _hits:\n"
            "        (_w / '.canary_leak_reference').write_text(repr(_hits[:20]))\n"
            "except Exception:\n"
            "    pass\n"
        ),
    ),
    Canary(
        name="module_global_write_evasive",
        channel="module global",
        # In-container co-residency is expected; the host module state and layer 2 are the
        # signals that matter here. See `ws_marker_proves_escape`.
        ws_marker_proves_escape=False,
        question="can candidate code mutate the grader module without naming the package?",
        payload=(
            "import gc as _gc, pathlib as _p\n"
            "try:\n"
            "    _t = 'trgym.repo' + '.checks'\n"
            "    _w = _p.Path(__file__).resolve().parents[1]\n"
            "    for _o in _gc.get_objects():\n"
            "        if type(_o).__name__ != 'module':\n"
            "            continue\n"
            "        if getattr(_o, '__name__', '') != _t:\n"
            "            continue\n"
            "        _o.ATOL = 1e9\n"
            "        (_w / '.canary_leak_module').write_text('mutated')\n"
            "        break\n"
            "except Exception:\n"
            "    pass\n"
        ),
    ),
]


def build_canary_workspace(canary: Canary, dest: Path) -> Path:
    from trgym.repo.build import build_repo
    from trgym.tasks.repo_specs import get_repo_task

    spec = get_repo_task(TASK_ID)
    build_repo(spec, dest)  # the buggy tree: an untampered grade of it MUST fail
    payload = (
        canary.payload
        .replace("EVIDENCE_DIR_PLACEHOLDER", repr(str(EVIDENCE_DIR)))
        .replace("REPO_ROOT_PLACEHOLDER", repr(str(ROOT)))
    )
    target = dest / "tinygpt" / "config.py"
    target.write_text(
        target.read_text(encoding="utf-8")
        + "\n\n# ---- G5 isolation canary payload ----\n"
        + WITNESS
        + payload,
        encoding="utf-8",
    )
    return dest


def evidence_present() -> list[str]:
    if not EVIDENCE_DIR.exists():
        return []
    return sorted(p.name for p in EVIDENCE_DIR.iterdir())


def _snapshot_grader_state() -> dict:
    """Record the grader module state so a leaking probe cannot poison later canaries.

    The in-process control genuinely lets candidate code rewrite `trgym.repo.checks`. If
    that is not undone between canaries, every subsequent canary grades against a stubbed
    check registry and reports whatever the first successful attacker wanted -- the suite
    would measure the first probe and then hallucinate the rest.
    """
    try:
        import trgym.repo.checks as c
    except Exception:  # noqa: BLE001
        return {}
    return {"ATOL": c.ATOL, "RTOL": c.RTOL, "CHECKS": dict(c.CHECKS)}


def _restore_grader_state(snap: dict) -> bool:
    """Put the grader module back. Returns True if anything had actually been changed."""
    if not snap:
        return False
    import trgym.repo.checks as c

    changed = (
        c.ATOL != snap["ATOL"]
        or c.RTOL != snap["RTOL"]
        or dict(c.CHECKS) != snap["CHECKS"]
    )
    c.ATOL, c.RTOL = snap["ATOL"], snap["RTOL"]
    c.CHECKS.clear()
    c.CHECKS.update(snap["CHECKS"])
    return changed


def run_canary(canary: Canary, grader) -> dict:
    """Grade a canary workspace with `grader` and report whether the probe escaped."""
    import shutil

    shutil.rmtree(EVIDENCE_DIR, ignore_errors=True)
    os.environ.pop("TRGYM_CANARY_ENV", None)

    ws = Path(tempfile.mkdtemp(prefix=f"g5_{canary.name}_"))
    snap = _snapshot_grader_state()
    try:
        build_canary_workspace(canary, ws)
        error = None
        verdict_passed = None
        refused_by_gate = False
        outcome = None
        try:
            outcome = grader(ws, TASK_ID)
            verdict_passed = outcome.passed
            blob = " ".join(outcome.errors.values())
            refused_by_gate = "refused to grade" in blob
        except Exception as exc:  # noqa: BLE001 - a refusing grader is a PASS, recorded
            error = f"{type(exc).__name__}: {exc}"[:200]

        files = evidence_present()
        env_leaked = os.environ.get("TRGYM_CANARY_ENV") == "leaked"
        global_leaked = False
        if canary.name == "module_global_write":
            try:
                import trgym.repo.checks as c

                global_leaked = getattr(c, "ATOL", None) == 1e9
                if global_leaked:
                    c.ATOL = 1e-5  # restore, or every later check is meaningless
            except Exception:  # noqa: BLE001
                pass

        witnessed = (ws / ".canary_witness").exists()
        # An exfiltration channel that works from *inside* the grading container.
        # EVIDENCE_DIR is in the grading host's temp dir, which the sandbox cannot write to
        # at all -- so a probe reporting only there can never signal a leak from the
        # sandboxed path, and "contained" would mean no more than "the channel was broken".
        # The workspace is a read-write bind mount (that is how the witness file gets back),
        # so a probe that genuinely reached something is able to say so.
        ws_evidence = sorted(p.name for p in ws.glob(".canary_leak_*"))
        ws_escape = bool(ws_evidence) and canary.ws_marker_proves_escape
        # What the probe actually found, not just that it found something. A leak whose
        # evidence is a bare filename cannot be triaged without re-running the suite.
        ws_detail = {}
        for _p in sorted(ws.glob(".canary_leak_*")):
            try:
                ws_detail[_p.name] = _p.read_text(errors="replace")[:400]
            except OSError:
                ws_detail[_p.name] = "<unreadable>"
        # Any mutation of the grader module counts as a module-global leak, whichever
        # probe caused it -- and restoring is what keeps the next canary honest.
        grader_state_mutated = _restore_grader_state(snap)
        global_leaked = global_leaked or grader_state_mutated
        leaked = bool(files) or ws_escape or env_leaked or global_leaked
        if canary.kind == "verdict_integrity":
            # The tree carries the planted defect, so an honest grader must FAIL it.
            # A PASS means the candidate rewrote its own verdict. Containment still counts:
            # assigning rather than OR-ing here would discard a real escape whenever the
            # verdict happened to come back correct.
            leaked = bool(verdict_passed) or leaked
        # Which layer actually stopped it. "the probe never ran" and "the probe ran and
        # got nowhere" are both containment, but they are not the same claim, and a suite
        # that cannot tell them apart cannot say what it verified.
        if leaked:
            contained_by = None
        elif refused_by_gate:
            contained_by = "static_gate_before_execution"
        elif witnessed:
            contained_by = "isolation_or_integrity_check_after_execution"
        else:
            contained_by = "probe_did_not_execute"

        return {
            "canary": canary.name,
            "channel": canary.channel,
            "kind": canary.kind,
            "question": canary.question,
            "leaked": leaked,
            "contained_by": contained_by,
            "evidence_files": files,
            "workspace_evidence_files": ws_evidence,
            "workspace_evidence_detail": ws_detail,
            "workspace_marker_counts_as_escape": ws_escape,
            "env_leaked": env_leaked,
            "module_global_leaked": global_leaked,
            "payload_witnessed": witnessed,
            "refused_by_gate": refused_by_gate,
            "graded_passed": verdict_passed,
            "grader_failure_detail": (
                sorted(outcome.errors.values())[0][:300] if outcome and outcome.errors else None
            ),
            # Distinguishes "the checks ran and the tree failed on its merits" from
            # "grading never reached the candidate at all", which would make every
            # containment result vacuous.
            "checks_reached_candidate_code": (
                None if outcome is None else not any(
                    marker in " ".join(outcome.errors.values())
                    for marker in ("ModuleNotFoundError", "Permission", "No such file",
                                   "produced no result", "refused to grade")
                )
            ),
            "grader_error": error,
        }
    finally:
        import shutil as _sh

        _sh.rmtree(ws, ignore_errors=True)
        _sh.rmtree(EVIDENCE_DIR, ignore_errors=True)
        os.environ.pop("TRGYM_CANARY_ENV", None)


# --------------------------------------------------------------------------- #
# The two graders under test
# --------------------------------------------------------------------------- #
def grade_in_process(ws: Path, task_id: str):
    """The pre-R14 path: import the candidate in the grading process.

    Reached deliberately through the trusted-tree opt-in. This is the POSITIVE CONTROL and
    it is expected to leak; if it does not, the probes are not testing anything.
    """
    from trgym.repo.verifier_v2 import v2_checks
    from trgym.tasks.repo_specs import get_repo_task

    from environments.transformer_repair.grading import grade_workspace

    return grade_workspace(
        ws, task_id, v2_checks(get_repo_task(task_id)), allow_in_process=True
    )


def grade_sandboxed(ws: Path, task_id: str):
    """The production path exactly as `semantic_repair` calls it -- default arguments."""
    from trgym.repo.verifier_v2 import v2_checks
    from trgym.tasks.repo_specs import get_repo_task

    from environments.transformer_repair.grading import grade_workspace

    return grade_workspace(ws, task_id, v2_checks(get_repo_task(task_id)))


def main() -> int:
    os.environ[GRADER_SECRET_ENV] = GRADER_SECRET_VALUE
    results: dict[str, list[dict]] = {}

    for label, grader in (("in_process_UNSAFE_CONTROL", grade_in_process),
                          ("sandboxed_container", grade_sandboxed)):
        print(f"\n=== grader: {label} ===")
        rows = []
        for canary in CANARIES:
            try:
                row = run_canary(canary, grader)
            except Exception as exc:  # noqa: BLE001 - an unavailable grader is recorded
                row = {"canary": canary.name, "channel": canary.channel,
                       "kind": canary.kind, "leaked": None,
                       "unavailable": f"{type(exc).__name__}: {exc}"[:200]}
            rows.append(row)
            state = ("LEAK" if row.get("leaked") else
                     "unavailable" if row.get("leaked") is None else "contained")
            print(f"  {state:<12} {canary.name:<34} ({canary.channel}) "
                  f"by={row.get('contained_by')}")
            if row.get("evidence_files"):
                print(f"               evidence: {row['evidence_files']}")
        results[label] = rows

    os.environ.pop(GRADER_SECRET_ENV, None)

    summary = {
        label: {
            "n_canaries": len(rows),
            "n_leaked": sum(1 for r in rows if r.get("leaked")),
            "n_contained": sum(1 for r in rows if r.get("leaked") is False),
            "n_unavailable": sum(1 for r in rows if r.get("leaked") is None),
            "n_payloads_witnessed": sum(1 for r in rows if r.get("payload_witnessed")),
            "n_refused_by_gate": sum(1 for r in rows if r.get("refused_by_gate")),
            "all_contained": all(r.get("leaked") is False for r in rows),
            "n_probes_that_executed_and_were_contained": sum(
                1 for r in rows
                if r.get("contained_by") == "isolation_or_integrity_check_after_execution"
            ),
            # "Refused before execution" is a real defence, but it is a different claim
            # from "ran and got nowhere", and it silently shadows the layer behind it. A
            # channel whose every probe was refused has not been measured at all, even
            # though each of its rows reads "contained". This is the field that says so.
            "channels_without_an_executing_probe": sorted(
                {r["channel"] for r in rows}
                - {r["channel"] for r in rows if r.get("payload_witnessed")}
            ),
            "contained_by": {r["canary"]: r.get("contained_by") for r in rows},
        }
        for label, rows in results.items()
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "results": results},
                              indent=2, sort_keys=True), encoding="utf-8")
    print("\n=== summary ===")
    for label, s in summary.items():
        print(f"  {label:<26} leaked {s['n_leaked']}/{s['n_canaries']}  "
              f"ran {s['n_payloads_witnessed']}  refused {s['n_refused_by_gate']}  "
              f"all_contained={s['all_contained']}")
        unmeasured = s["channels_without_an_executing_probe"]
        if unmeasured:
            print(f"  {'':<26} UNMEASURED CHANNELS (no probe executed): {unmeasured}")
    print(f"\nwrote {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
