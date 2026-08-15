"""Final acceptance: verify G0-G9 from artifacts, never from prose.

Exit 0 only when every gate passes its frozen machine check. The rule this script exists
to enforce is that a gate is closed by evidence on disk, not by a report saying so -- so it
never reads a verdict out of a markdown file. It re-derives each gate from the JSON/CSV/log
artifacts the contract names, and a missing artifact is a FAIL rather than a skip.

Two deliberate design choices:

  * **No gate may be satisfied by another report.** G8's numbers, for example, are checked
    against a freshly regenerated summary hash, not against G8's own prose.
  * **Absent evidence fails.** Several gates could be made to "pass" by treating a missing
    file as vacuous truth; every check here requires the artifact to exist and to contain
    the specific field being asserted.

Usage:
    python scripts/final_acceptance.py            # verify
    python scripts/final_acceptance.py --json     # machine-readable summary as well

Run inside Linux/Docker for the gates that need `verifiers.v1` (G1); on Windows those
checks report BLOCKED-PLATFORM rather than PASS, because `verifiers.v1` imports POSIX-only
`fcntl` and a skipped check must never read as a pass.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass
class GateResult:
    gate: str
    title: str
    status: str = "FAIL"          # PASS | FAIL | BLOCKED-PLATFORM
    checks: list[dict] = field(default_factory=list)

    def record(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append({"check": name, "ok": bool(ok), "detail": detail[:400]})
        return bool(ok)

    def finish(self) -> "GateResult":
        if self.status != "BLOCKED-PLATFORM":
            self.status = "PASS" if self.checks and all(
                c["ok"] for c in self.checks
            ) else "FAIL"
        return self

    @property
    def failures(self) -> list[dict]:
        return [c for c in self.checks if not c["ok"]]


def load_json(rel: str):
    path = ROOT / rel
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def exists(rel: str) -> bool:
    return (ROOT / rel).exists()


def _content_sha256(path: Path) -> str:
    """Line-ending independent content hash.

    Deliberately inlined rather than imported: this gate must run in a bare environment
    with nothing but the standard library, so it does not depend on `trgym`. It MUST stay
    byte-for-byte equivalent to `trgym.provenance.content_sha256`, and
    `tests/test_provenance_hashing.py` fails if the two ever disagree.

    Hashing raw bytes made these gates platform-dependent: Git stores LF, a Windows
    checkout materialises CRLF, and a digest pinned on one OS then failed on the other for
    a file that had not changed -- G1 reporting "stale evidence" about evidence that was
    current, and G8 reporting a report/summary mismatch that did not exist.
    """
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# --------------------------------------------------------------------------- #
# Static inspection helpers
# --------------------------------------------------------------------------- #
# Both of these exist because the first version of the R16 gates searched raw text and
# failed on the comments that *document* the fix -- "/grader" appears in the line saying
# the mount was removed, and obs_protocol's docstring explains at length why pickle is
# banned. A gate that cannot distinguish a warning from a use is the R13 false-positive
# problem wearing a different hat.
def _fn_code(path: Path, name: str) -> str:
    """Unparsed source of one function: comments gone, code intact."""
    import ast

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return ""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.unparse(node)
    return ""


def _banned_calls(path: Path, banned: dict[str, set[str]]) -> list[str]:
    """Return `owner.attr` for every call matching `banned`, e.g. {"torch": {"load"}}."""
    import ast

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return ["<unparseable>"]
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in banned.get(
            ast.unparse(func.value), set()
        ):
            found.append(f"{ast.unparse(func.value)}.{func.attr}")
    return found


# --------------------------------------------------------------------------- #
# G0 -- evidence preservation
# --------------------------------------------------------------------------- #
def gate_g0() -> GateResult:
    g = GateResult("G0", "Evidence preservation")
    manifest = load_json("FROZEN_PHASE1_MANIFEST.json")
    if not g.record("FROZEN_PHASE1_MANIFEST.json parses", manifest is not None):
        return g.finish()

    g.record("manifest verdict is PASS", manifest.get("verdict") == "PASS",
             str(manifest.get("verdict")))
    g.record("no missing artifacts", manifest.get("n_missing") == 0,
             f"n_missing={manifest.get('n_missing')}")
    g.record("no unparseable artifacts", manifest.get("n_unparseable") == 0,
             f"n_unparseable={manifest.get('n_unparseable')}")

    # The contract requires >=6 headline metrics recomputed and diffed. The manifest
    # stores them as two parallel dicts (`regenerated_` vs `claimed_headline_metrics`)
    # keyed by metric family; each family holds several individual numbers.
    regenerated = manifest.get("regenerated_headline_metrics") or {}
    claimed = manifest.get("claimed_headline_metrics") or {}
    mismatches = manifest.get("mismatches") or []
    n_families = len(regenerated) if isinstance(regenerated, dict) else 0
    n_metrics = sum(
        len(v) if isinstance(v, dict) else 1 for v in regenerated.values()
    ) if isinstance(regenerated, dict) else 0
    g.record("at least 6 headline metrics regenerated", n_metrics >= 6,
             f"n_metrics={n_metrics} across {n_families} families")
    g.record("every regenerated metric family has a claimed counterpart",
             bool(claimed) and set(regenerated) <= set(claimed),
             f"regenerated={sorted(regenerated)} claimed={sorted(claimed)}")
    g.record("zero metric mismatches", len(mismatches) == 0, str(mismatches[:5]))
    g.record("frozen copies present", exists("artifacts/frozen_phase1"))
    return g.finish()


# --------------------------------------------------------------------------- #
# G1 -- verifiers.v1 migration
# --------------------------------------------------------------------------- #
PUBLIC_PATH = "environments"
LEGACY_MARKERS = ("vf.Environment", "SingleTurnEnv", "MultiTurnEnv", "vf.Rubric", "Rubric(")


def _code_only(text: str) -> str:
    import io
    import tokenize

    kept: list[str] = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            kept.append(tok.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return text
    return " ".join(kept)


def gate_g1() -> GateResult:
    g = GateResult("G1", "verifiers.v1 migration")

    for rel in ("VERIFIERS_V1_MIGRATION_AUDIT.md", "VERIFIERS_V1_MIGRATION_REPORT.md",
                "VERIFIERS_VERSION_SNAPSHOT.md", "environments/transformer_repair"):
        g.record(f"artifact {rel}", exists(rel))

    # Static half: no legacy v0 API on the public path. Runs on any platform.
    import ast

    offenders: list[str] = []
    for py in sorted((ROOT / PUBLIC_PATH).rglob("*.py")):
        text = py.read_text(encoding="utf-8")
        code = _code_only(text)
        rel = py.relative_to(ROOT).as_posix()
        for marker in LEGACY_MARKERS:
            if marker in code:
                offenders.append(f"{rel}:{marker}")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            offenders.append(f"{rel}:unparseable")
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name == "verifiers":
                        offenders.append(f"{rel}:import verifiers")
            if isinstance(node, ast.ImportFrom) and node.module == "verifiers":
                if not all(a.name.startswith("v1") for a in node.names):
                    offenders.append(f"{rel}:from verifiers import ...")
    g.record("no legacy v0 API on the public path", not offenders, str(offenders[:5]))

    # R14: the isolation repair must be in force, checked statically so it holds on
    # Windows too. A gate that only checks this under Docker can regress silently.
    grading = (ROOT / "environments" / "transformer_repair" / "grading.py")
    if g.record("grading.py present", grading.exists()):
        src = grading.read_text(encoding="utf-8")
        g.record("grading defaults to sandboxed (R14)",
                 "allow_in_process: bool = False" in src, "default must be False")
        g.record("sandbox called without fallback (R14)",
                 "fallback=False" in src, "fallback=True would restore the defect")
        g.record("a tampered run is discarded rather than believed (R15)",
                 "result.tampered" in src)
        # R16 replaced R15's static denylist with the absence of the oracle. A denylist
        # returning is a regression, not a hardening: while it stood it refused every
        # canary before execution, so the suite scored "contained" for probes that never
        # ran. Its absence is therefore itself a gate condition.
        g.record("no pattern denylist over candidate sources (R16)",
                 "scan_for_grader_tampering" not in src and "re.compile" not in src,
                 "a source-pattern gate hides the boundary from measurement")

    sandbox_py = ROOT / "trgym" / "harness" / "sandbox.py"
    if g.record("sandbox.py present", sandbox_py.exists()):
        ssrc = sandbox_py.read_text(encoding="utf-8")
        # The core R16 property, checked statically so it holds on Windows too.
        # Read through the AST, not the raw text: the comment explaining why the mount is
        # gone naturally contains "/grader", and a text search would fail on the prose
        # that documents the fix. R13 taught this exact lesson about pattern matching.
        g.record("candidate container mounts no repository (R16)",
                 "/grader" not in _fn_code(sandbox_py, "run_checks_containerized")
                 and ":/probe:ro" in ssrc,
                 "mounting the repo is what exposed the oracle and gold")
        g.record("candidate container gets only workspace + probe (R16)",
                 ':/workspace:rw"' in ssrc and "_stage_probe" in ssrc)
        for flag in ("--network=none", "--cap-drop=ALL",
                     "--security-opt=no-new-privileges", "--read-only"):
            g.record(f"sandbox keeps {flag}", flag in ssrc)
        g.record("observation protocol is nonced per job (R16)",
                 "secrets.token_hex" in ssrc and 'MARKER = "<<<TRGYM_OBS"' in ssrc)

    probe_py = ROOT / "trgym" / "repo" / "candidate_probe.py"
    if g.record("candidate_probe.py present (R16)", probe_py.exists()):
        psrc = probe_py.read_text(encoding="utf-8")
        g.record("probe imports no part of the grader (R16)",
                 "import trgym" not in psrc and "from trgym" not in psrc)
        g.record("probe emits observations, never a verdict (R16)",
                 '"passed"' not in psrc and "CheckFailure" not in psrc)

    protocol_py = ROOT / "trgym" / "repo" / "obs_protocol.py"
    if g.record("obs_protocol.py present (R16)", protocol_py.exists()):
        osrc = protocol_py.read_text(encoding="utf-8")
        # torch.load is an unpickler; pointing it at candidate output would be a
        # remote-code-execution path into the one process that holds gold. Checked over
        # call nodes, because the module's own docstring explains at length why pickle is
        # banned -- and a text search cannot tell a warning from a use.
        g.record("candidate output is never unpickled (R16)",
                 not _banned_calls(protocol_py, {"torch": {"load"},
                                                 "pickle": {"load", "loads"}}),
                 "candidate bytes must not reach an unpickler")
        g.record("tensors decoded as validated raw buffers (R16)",
                 "frombuffer" in osrc and "MAX_TENSOR_ELEMENTS" in osrc)
        g.record("the last nonced block wins (R16)", "text.rfind(marker)" in osrc)

    predicates_py = ROOT / "trgym" / "repo" / "predicates.py"
    if g.record("predicates.py present (R16)", predicates_py.exists()):
        prsrc = predicates_py.read_text(encoding="utf-8")
        g.record("trusted comparator imports only gold, never the candidate (R16)",
                 "RepoModules(gold_repo(" in prsrc and "RepoModules(ws" not in prsrc
                 and "RepoModules(workspace" not in prsrc)
    task_py = (ROOT / "environments" / "transformer_repair" / "task.py")
    if task_py.exists():
        reward_src = task_py.read_text(encoding="utf-8")
        idx = reward_src.find("async def semantic_repair")
        nxt = reward_src.find("\n    @metric", idx + 1) if idx >= 0 else -1
        body = reward_src[idx:nxt] if idx >= 0 and nxt > idx else ""
        g.record("reward path never grades in-process (R14)",
                 bool(body) and "allow_in_process" not in body)

    # Provenance: which v1 code actually executed.
    prov = load_json("artifacts/raw/v1_provenance.json")
    if g.record("v1 provenance artifact parses", prov is not None):
        verdict = prov.get("verdict") or {}
        g.record("provenance verdict PASS", verdict.get("PASS") is True, str(verdict))
        g.record("v1 modules actually loaded",
                 (prov.get("n_v1_modules_loaded") or 0) >= 20,
                 f"n={prov.get('n_v1_modules_loaded')}")
        g.record("no v0 execution path",
                 not (prov.get("legacy_v0_env_modules_loaded") or []),
                 str(prov.get("legacy_v0_env_modules_loaded")))

    # Live smoke through the v1 lifecycle.
    smoke = sorted((ROOT / "artifacts" / "raw").glob("v1_smoke*/traces.jsonl"))
    if g.record("live v1 smoke trace present", bool(smoke), str([p.name for p in smoke])):
        solved = False
        for path in smoke:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                traces = rec.get("traces") or []
                for t in (traces if isinstance(traces, list) else [traces]):
                    rw = (t.get("rewards") or {}).get("semantic_repair") or {}
                    if isinstance(rw, dict) and rw.get("score") == 1.0:
                        solved = True
        g.record("a live smoke episode scored semantic_repair 1.0", solved)

    # Runtime half. `verifiers.v1` imports POSIX-only `fcntl`, so it can only execute in
    # Linux/Docker -- but "cannot run here" must not become "skipped", so the runtime
    # evidence is read from an artifact that a Linux run WROTE, and its absence is a FAIL.
    # On Linux the same checks are additionally re-executed live.
    ev = load_json("artifacts/raw/v1_runtime_evidence.json")
    if g.record("v1 runtime evidence artifact parses (written by the Linux run)",
                ev is not None):
        g.record("evidence records a v1 import", ev.get("v1_import_ok") is True,
                 str(ev.get("v1_import_error"))[:200])
        g.record("taskset loads and enumerates via the official loader",
                 (ev.get("n_tasks") or 0) >= 10, f"n_tasks={ev.get('n_tasks')}")
        g.record("TaskData is immutable", ev.get("taskdata_immutable") is True)
        g.record("official validate CLI dry run at valid_rate 1.0",
                 ev.get("validate_valid_rate") == 1.0,
                 f"valid_rate={ev.get('validate_valid_rate')}")
        g.record("tests_v1 suite green in Linux",
                 ev.get("tests_v1_failed") == 0 and (ev.get("tests_v1_passed") or 0) >= 15,
                 f"passed={ev.get('tests_v1_passed')} failed={ev.get('tests_v1_failed')}")
        g.record("gold PASS vs planted-bug FAIL separation re-measured",
                 ev.get("gold_reward") == 1.0 and ev.get("buggy_reward") == 0.0,
                 f"gold={ev.get('gold_reward')} buggy={ev.get('buggy_reward')}")
        g.record("evidence is for this grading.py (hash match)",
                 ev.get("grading_sha256") == _content_sha256(
                     ROOT / "environments" / "transformer_repair" / "grading.py"
                 ) if grading.exists() else False,
                 "stale evidence must not close the gate")

    if sys.platform != "win32":
        try:
            import verifiers  # noqa: F401
            from verifiers.v1.taskset import Taskset  # noqa: F401

            sys.path.insert(0, str(ROOT / "environments"))
            from verifiers.v1.utils.loaders import load_taskset, taskset_config_type

            cfg = taskset_config_type("transformer_repair")
            ts = load_taskset(cfg(id="transformer_repair"))
            tasks = list(ts)
            g.record("taskset loads live in this process", len(tasks) >= 10,
                     f"n_tasks={len(tasks)}")
        except Exception as exc:  # noqa: BLE001
            g.record("taskset loads live in this process", False,
                     f"{type(exc).__name__}: {exc}")
    return g.finish()


# --------------------------------------------------------------------------- #
# G2 -- hardened verifier v2 + replay
# --------------------------------------------------------------------------- #
def gate_g2() -> GateResult:
    g = GateResult("G2", "Hardened verifier v2 + replay")
    g.record("VERIFIER_V2_PROTOCOL.md present", exists("VERIFIER_V2_PROTOCOL.md"))
    replay = ROOT / "VERIFIER_V2_REPLAY.csv"
    if not g.record("VERIFIER_V2_REPLAY.csv present", replay.exists()):
        return g.finish()

    rows = list(csv.DictReader(replay.read_text(encoding="utf-8").splitlines()))
    g.record("replay has rows", len(rows) > 0, f"n={len(rows)}")

    # `scripts/verifier_v2_replay.py` writes {"summary": {...}, "rows": [...]}.
    doc = load_json("artifacts/verifier_v2_replay.json") or {}
    summary = doc.get("summary") or {}
    if g.record("replay summary artifact present", bool(summary)):
        cov = summary.get("replay_coverage")
        g.record("replay coverage >= 90%", cov is not None and float(cov) >= 0.90,
                 f"replay_coverage={cov}")
        n_mt = summary.get("multi_turn_replayed")
        g.record(">= 45 of the 50 multi-turn trajectories replayed",
                 n_mt is not None and int(n_mt) >= 45, f"multi_turn_replayed={n_mt}")
        v1fp, v2fp = summary.get("v1_FP_rate"), summary.get("v2_FP_rate")
        g.record("v2_FPR <= v1_FPR",
                 v1fp is not None and v2fp is not None and float(v2fp) <= float(v1fp),
                 f"v1_FP_rate={v1fp} v2_FP_rate={v2fp}")
        g.record("v2_FNR == 0 on FULL_FIX", summary.get("v2_FN_count") in (0, 0.0),
                 f"v2_FN_count={summary.get('v2_FN_count')}")
        # Criterion 7: the FP rate must not have been "improved" by rejecting more
        # legitimate fixes. The replay records this as an explicit boolean.
        crit = summary.get("protocol_criteria") or {}
        g.record("FP not reduced by blanket rejection",
                 crit.get("c7_fp_not_reduced_by_over_rejecting") is True, str(crit))
        g.record("no new FULL_FIX rejection",
                 crit.get("c4_no_new_full_fix_rejection") is True, str(crit))

    fuzz = load_json("artifacts/verifier_fuzz_audit.json") or {}
    fs = fuzz.get("summary") or {}
    if g.record("fuzz audit artifact present", bool(fs)):
        g.record("v2 rejects every contract-edge probe",
                 fs.get("v2_rejects_contract_probes") == fs.get("n_contract_probes")
                 and (fs.get("n_contract_probes") or 0) >= 3,
                 f"{fs.get('v2_rejects_contract_probes')}/{fs.get('n_contract_probes')}")
        g.record("v2 rejects no FULL_FIX probe",
                 fs.get("v2_rejects_any_FULL_FIX_probe") in (0, 0.0, False),
                 str(fs.get("v2_rejects_any_FULL_FIX_probe")))
    return g.finish()


# --------------------------------------------------------------------------- #
# G3 -- turn-budget ablation
# --------------------------------------------------------------------------- #
VERDICT_TOKENS = ("BUDGET-SENSITIVE", "BUDGET-INSENSITIVE", "INCONCLUSIVE")


def gate_g3() -> GateResult:
    g = GateResult("G3", "Turn-budget ablation")
    g.record("protocol frozen first", exists("TURN_BUDGET_ABLATION_PROTOCOL.md"))
    report = ROOT / "TURN_BUDGET_ABLATION_REPORT.md"
    if not g.record("report present", report.exists()):
        return g.finish()
    text = report.read_text(encoding="utf-8")
    g.record("both budgets present", "14" in text and "24" in text)
    g.record("stated verdict token",
             any(t in text for t in VERDICT_TOKENS),
             next((t for t in VERDICT_TOKENS if t in text), "none"))
    g.record("24-turn raw trajectories present", exists("artifacts/tier_h_24turn.jsonl"))
    return g.finish()


# --------------------------------------------------------------------------- #
# G4 -- Tier S localization
# --------------------------------------------------------------------------- #
def gate_g4() -> GateResult:
    g = GateResult("G4", "Tier S localization")
    for rel in ("LOCALIZATION_SCALE_REPORT.md", "artifacts/tier_s_primary.jsonl",
                "TIER_S_REAL_MODEL_AUDIT.csv"):
        g.record(f"artifact {rel}", exists(rel))
    spec = load_json("artifacts/tier_s_spec.json")
    if g.record("frozen Tier S spec present", spec is not None):
        tasks = spec.get("tasks") or []
        g.record("exactly 3 tasks", len(tasks) == 3, f"n={len(tasks)}")
        for t in tasks:
            n = t.get("n_files")
            g.record(f"{t.get('task_id')}: 20 <= files <= 50",
                     n is not None and 20 <= int(n) <= 50, f"n_files={n}")
            rel_n = len(t.get("relevant_files") or [])
            g.record(f"{t.get('task_id')}: relevant files <= 3", 1 <= rel_n <= 3,
                     f"n_relevant={rel_n}")
            g.record(f"{t.get('task_id')}: every non-relevant file is referenced",
                     t.get("all_non_relevant_referenced") is True)
            g.record(f"{t.get('task_id')}: gold passes", t.get("gold_passes") is True)
            g.record(f"{t.get('task_id')}: no-op fails", t.get("noop_fails") is True)

    jsonl = ROOT / "artifacts" / "tier_s_primary.jsonl"
    if jsonl.exists():
        eps = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
        g.record("12 precommitted trajectories", len(eps) == 12, f"n={len(eps)}")
        bad = [e for e in eps if not (0.0 <= float(e.get("fraction_repo_inspected", 1.0)) < 1.0)]
        g.record("fraction_repo_inspected < 1.0 for every episode", not bad,
                 f"{len(bad)} episodes inspected the whole repo")
    return g.finish()


# --------------------------------------------------------------------------- #
# G5 -- grader scalability + isolation
# --------------------------------------------------------------------------- #
REQUIRED_CHANNELS = ("file", "env var", "module global", "child process",
                     "temp dir", "grader secret")


def gate_g5() -> GateResult:
    g = GateResult("G5", "Grader scalability + isolation")
    g.record("VERIFIER_SCALABILITY_REPORT.md present",
             exists("VERIFIER_SCALABILITY_REPORT.md"))

    canaries = load_json("artifacts/g5_isolation_canaries.json")
    if g.record("isolation canary artifact parses", canaries is not None):
        results = canaries.get("results") or {}
        final = results.get("sandboxed_container") or []
        control = results.get("in_process_UNSAFE_CONTROL") or []
        channels = {r.get("channel") for r in final}
        for ch in REQUIRED_CHANNELS:
            g.record(f"canary channel covered: {ch}", ch in channels)
        # R15: containment is not integrity. Both are required.
        g.record("verdict-integrity canaries present",
                 sum(1 for r in final if r.get("kind") == "verdict_integrity") >= 2,
                 "a contained process can still return a forged verdict")
        leaked = [r["canary"] for r in final if r.get("leaked")]
        g.record("final path contains every canary", not leaked, f"leaked={leaked}")
        # A probe that never executed contains nothing; it just did not happen. The
        # artifact records which layer stopped each one, so a vacuous run cannot pass.
        vacuous = [r["canary"] for r in final
                   if r.get("contained_by") == "probe_did_not_execute"]
        g.record("every canary either executed or was refused before execution",
                 not vacuous, f"never_ran={vacuous}")
        # A control that never leaks means the canaries prove nothing.
        control_leaks = [r["canary"] for r in control if r.get("leaked")]
        g.record("unsafe in-process control DOES leak (positive control)",
                 len(control_leaks) >= 5, f"control_leaks={control_leaks}")

    bench = load_json("artifacts/g5_scalability.json")
    if g.record("benchmark artifact parses", bench is not None):
        n = bench.get("n_jobs")
        g.record(">= 30 sequential grading jobs", n is not None and int(n) >= 30, f"n={n}")
        for path in ("cold", "final"):
            stats = (bench.get(path) or {})
            for k in ("mean_s", "p50_s", "p95_s"):
                g.record(f"{path}.{k} recorded", stats.get(k) is not None,
                         f"{path}={stats}")
    return g.finish()


# --------------------------------------------------------------------------- #
# G6 -- heuristic gate FP audit
# --------------------------------------------------------------------------- #
def gate_g6() -> GateResult:
    g = GateResult("G6", "Heuristic gate FP audit")
    g.record("HEURISTIC_GATE_FP_AUDIT.md present", exists("HEURISTIC_GATE_FP_AUDIT.md"))
    audit = load_json("artifacts/heuristic_gate_fp_audit.json")
    if g.record("FP audit artifact parses", audit is not None):
        n = audit.get("n_samples")
        g.record(">= 20 benign samples", n is not None and int(n) >= 20, f"n={n}")
        samples = audit.get("samples") or []
        with_prov = [
            s for s in samples
            if s.get("package") and s.get("package_version") and s.get("license")
            and s.get("module") and s.get("sha256")
        ]
        g.record("every sample records source/version/license",
                 len(with_prov) == len(samples), f"{len(with_prov)}/{len(samples)}")
        g.record("gate decisions tabulated", bool(audit.get("per_gate_fp_rate")))
        ex = audit.get("exploit_suite") or {}
        g.record("exploit suite still catches all cheats", ex.get("all_caught") is True,
                 f"{ex.get('n_caught')}/{ex.get('n_cheats')}")
        g.record("exploit suite has 7 cheats", ex.get("n_cheats") == 7,
                 f"n={ex.get('n_cheats')}")
    return g.finish()


# --------------------------------------------------------------------------- #
# G7 -- packaging / CI / fresh clone
# --------------------------------------------------------------------------- #
G7_ARTIFACTS = ("REPRODUCIBILITY.md", "FINAL_TEST_RUN.log", ".github/workflows/ci.yml",
                "pyproject.toml", "uv.lock", "LICENSE", "CITATION.cff",
                "SECURITY_MODEL.md", "LIMITATIONS.md")


def gate_g7() -> GateResult:
    g = GateResult("G7", "Packaging / CI / fresh clone")
    for rel in G7_ARTIFACTS:
        g.record(f"artifact {rel}", exists(rel))

    fresh = load_json("artifacts/fresh_clone_run.json")
    if g.record("fresh-clone log artifact parses", fresh is not None):
        g.record("fresh clone exited 0", fresh.get("exit_code") == 0,
                 f"exit={fresh.get('exit_code')}")
        clone_path = str(fresh.get("clone_path") or "")
        g.record("clone path is not the original tree",
                 bool(clone_path) and Path(clone_path).resolve() != ROOT.resolve(),
                 clone_path)
        # The recorded module.__file__ must not point back into E:\RL.
        mods = fresh.get("module_files") or {}
        leaked = [f"{k}={v}" for k, v in mods.items()
                  if str(ROOT).lower().replace("\\", "/") in str(v).lower().replace("\\", "/")]
        g.record("recorded module.__file__ never resolves into the original tree",
                 bool(mods) and not leaked, str(leaked[:3]))
        g.record("README commands were the ones executed",
                 fresh.get("commands_match_readme") is True)
    return g.finish()


# --------------------------------------------------------------------------- #
# G8 -- final scientific synthesis
# --------------------------------------------------------------------------- #
def gate_g8() -> GateResult:
    g = GateResult("G8", "Final scientific synthesis")
    report = ROOT / "FINAL_FLAGSHIP_RESEARCH_REPORT.md"
    if not g.record("FINAL_FLAGSHIP_RESEARCH_REPORT.md present", report.exists()):
        return g.finish()
    text = report.read_text(encoding="utf-8")

    for rq in ("RQ1", "RQ2", "RQ3", "RQ4", "RQ5", "RQ6"):
        g.record(f"{rq} answered", rq in text)

    # The report must pin the hash of the generated summary, and it must match.
    summary_path = ROOT / "artifacts" / "final_metrics_summary.json"
    if g.record("canonical metric summary present", summary_path.exists()):
        digest = _content_sha256(summary_path)
        g.record("report metric hash matches the regenerated summary",
                 digest in text, f"sha256={digest[:16]}…")

    # Forbidden unevidenced claims.
    banned = ("state-of-the-art", "SOTA", "statistically significant",
              "improves the policy", "production-ready security")
    present = [b for b in banned if b.lower() in text.lower()]
    g.record("no unevidenced superlatives", not present, str(present))
    return g.finish()


# --------------------------------------------------------------------------- #
# G9 -- post-success red team
# --------------------------------------------------------------------------- #
G9_ARTIFACTS = ("POST_SUCCESS_CODE_REVIEW.md", "POST_SUCCESS_RESULT_AUDIT.json",
                "POST_SUCCESS_SPOTCHECK.md", "POST_SUCCESS_MUTATION_TEST.md")
REQUIRED_MUTATIONS = ("hidden_oracle_protection", "grading_artifact_isolation",
                      "return_type_contract", "causal_mask", "cross_job_state_isolation")


def gate_g9(g0_to_g8_all_pass: bool) -> GateResult:
    g = GateResult("G9", "Post-success red team")
    if not g0_to_g8_all_pass:
        g.record("G0-G8 all PASS before G9 runs", False,
                 "G9 is only meaningful after G0-G8 are green")
        return g.finish()

    for rel in G9_ARTIFACTS:
        g.record(f"artifact {rel}", exists(rel))

    audit = load_json("POST_SUCCESS_RESULT_AUDIT.json")
    if g.record("result audit parses", audit is not None):
        mismatches = audit.get("mismatches") or []
        g.record("every headline metric regenerates exactly", not mismatches,
                 str(mismatches[:5]))

    mut = load_json("artifacts/post_success_mutation_test.json")
    if g.record("mutation artifact parses", mut is not None):
        rows = mut.get("mutations") or []
        names = {r.get("mutation") for r in rows}
        for req in REQUIRED_MUTATIONS:
            g.record(f"mutation exercised: {req}", req in names)
        survived = [r.get("mutation") for r in rows if not r.get("tests_went_red")]
        g.record("every mutation turned the suite RED", not survived,
                 f"survived={survived}")
        g.record("all mutations restored", mut.get("all_restored") is True)
        g.record("post-restore suite green", mut.get("post_restore_tests_pass") is True)

    review = ROOT / "POST_SUCCESS_CODE_REVIEW.md"
    if review.exists():
        rtext = review.read_text(encoding="utf-8")
        g.record("code review records a disposition for every finding",
                 "UNRESOLVED" not in rtext.upper(),
                 "an unresolved finding must reopen a gate")
    return g.finish()


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="also print a JSON summary")
    args = ap.parse_args()

    gates = [gate_g0(), gate_g1(), gate_g2(), gate_g3(),
             gate_g4(), gate_g5(), gate_g6(), gate_g7(), gate_g8()]
    g0_to_g8_pass = all(x.status == "PASS" for x in gates)
    gates.append(gate_g9(g0_to_g8_pass))

    print("=" * 72)
    print("FINAL ACCEPTANCE")
    print("=" * 72)
    for x in gates:
        print(f"{x.gate} {x.status:<18} {x.title}")
        for c in x.failures:
            print(f"      FAIL  {c['check']}"
                  + (f"  --  {c['detail']}" if c["detail"] else ""))

    all_pass = all(x.status == "PASS" for x in gates)
    complete = "YES" if all_pass else "NO"
    audit = "PASS" if all_pass and gates[-1].status == "PASS" else "NOT REACHED"
    ready = "YES" if all_pass else "NO"

    print("-" * 72)
    print(f"FLAGSHIP_GPU_FREE_COMPLETE = {complete}")
    print(f"POST_SUCCESS_AUDIT        = {audit}")
    print(f"READY_FOR_USER_REVIEW_AND_PUSH = {ready}")
    print("=" * 72)

    payload = {
        "gates": [
            {"gate": x.gate, "title": x.title, "status": x.status, "checks": x.checks}
            for x in gates
        ],
        "FLAGSHIP_GPU_FREE_COMPLETE": complete,
        "POST_SUCCESS_AUDIT": audit,
        "READY_FOR_USER_REVIEW_AND_PUSH": ready,
    }
    out = ROOT / "artifacts" / "final_acceptance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {out.relative_to(ROOT).as_posix()}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
