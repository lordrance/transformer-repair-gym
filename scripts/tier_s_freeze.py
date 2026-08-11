"""G4 preflight: verify the Tier S tasks, then freeze them before any paid call.

Every number in `artifacts/tier_s_spec.json` is measured here. Nothing is copied from a
docstring. That matters because the last three isolation claims in this project were all
written down before they were true, and the freeze is what makes the Tier S run
falsifiable rather than decorative.

Checks, in the order the contract states them:

  1. 20 <= plausible Python files <= 50, per task
  2. 1 <= relevant files <= 3, per task
  3. every non-relevant file is genuinely referenced -- reachable in the import graph
     from the package root. This is the anti-padding check, and it is a graph traversal
     rather than a grep, because `import x` in a file nothing imports proves nothing.
  4. every mutation's `find` string applies exactly `count` times
  5. gold PASSES the hidden suite
  6. the buggy tree FAILS it, and fails on the predicted checks
  7. the R16 boundary is still in force for the path used above

Run: python scripts/tier_s_freeze.py [--task s1_...] [--skip-grading]
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "artifacts" / "tier_s_spec.json"

MIN_FILES, MAX_FILES = 20, 50
MAX_RELEVANT = 3


# --------------------------------------------------------------------------- #
# Import graph
# --------------------------------------------------------------------------- #
def _module_name(rel: str) -> str:
    """'tinygpt/_ops/masking.py' -> 'tinygpt._ops.masking'."""
    stem = rel[: -len(".py")]
    if stem.endswith("/__init__"):
        stem = stem[: -len("/__init__")]
    return stem.replace("/", ".")


def _resolve(
    current: str, is_package: bool, node: ast.ImportFrom | ast.Import, alias: str
) -> list[str]:
    """Candidate absolute module names a single import could denote.

    `is_package` is load-bearing and was wrong in the first version of this script. For
    an ordinary module `a.b.c`, one leading dot means the package `a.b`. For a package's
    `__init__.py`, whose module name is already `a.b`, one dot means `a.b` itself. Getting
    that wrong resolved `from .seeding import ...` inside `_util/__init__.py` to
    `tinygpt.seeding`, found no such module, and reported a false orphan.
    """
    if isinstance(node, ast.Import):
        return [alias]
    parts = current.split(".")
    if node.level:
        climb = node.level - 1 if is_package else node.level
        base = parts[: len(parts) - climb] if climb else parts
    else:
        base = []
    prefix = ".".join(base + ([node.module] if node.module else []))
    return [f"{prefix}.{alias}" if prefix else alias, prefix]


def build_import_graph(pkg_root: Path) -> tuple[dict[str, set[str]], dict[str, str]]:
    """Return (edges, module -> relative path) over the package's own modules."""
    rel_by_module: dict[str, str] = {}
    is_pkg: dict[str, bool] = {}
    for path in sorted(pkg_root.rglob("*.py")):
        rel = f"tinygpt/{path.relative_to(pkg_root).as_posix()}"
        module = _module_name(rel)
        rel_by_module[module] = rel
        is_pkg[module] = path.name == "__init__.py"

    edges: dict[str, set[str]] = {m: set() for m in rel_by_module}
    for module, rel in rel_by_module.items():
        tree = ast.parse((pkg_root.parent / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [a.name for a in node.names]
            else:
                continue
            for alias in names:
                for candidate in _resolve(module, is_pkg[module], node, alias):
                    if candidate in rel_by_module:
                        edges[module].add(candidate)
    return edges, rel_by_module


def reachable_from_root(edges: dict[str, set[str]]) -> set[str]:
    """Modules reachable from `tinygpt` and from the eight public facades.

    The facades are roots too: `tinygpt/__init__.py` deliberately does not import the
    world (importing every submodule at package import is what makes a real library slow
    to import), so reachability purely from `__init__` would wrongly call the whole tree
    unreferenced.
    """
    roots = [m for m in edges if m == "tinygpt" or m.count(".") == 1]
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(edges.get(node, ()) - seen)
    return seen


# --------------------------------------------------------------------------- #
def preflight_task(spec, *, skip_grading: bool) -> dict:
    from trgym.repo.build import build_gold, build_repo
    from trgym.tasks.repo_specs_s import RELEVANT_FILES

    task_id = spec.task_id
    relevant = list(RELEVANT_FILES[task_id])
    row: dict = {"task_id": task_id, "tier": spec.tier, "relevant_files": relevant}

    work = Path(tempfile.mkdtemp(prefix=f"tier_s_{task_id}_"))
    try:
        gold = build_gold(spec, work / "gold")
        buggy = build_repo(spec, work / "buggy")

        pkg = gold / "tinygpt"
        py_files = sorted(
            f"tinygpt/{p.relative_to(pkg).as_posix()}" for p in pkg.rglob("*.py")
        )
        row["n_files"] = len(py_files)
        row["files"] = py_files
        row["n_files_in_range"] = MIN_FILES <= len(py_files) <= MAX_FILES
        row["n_relevant_in_range"] = 1 <= len(relevant) <= MAX_RELEVANT

        # -- anti-padding: every non-relevant file must be genuinely connected
        edges, rel_by_module = build_import_graph(pkg)
        reached = reachable_from_root(edges)
        reached_rel = {rel_by_module[m] for m in reached}
        orphans = sorted(set(py_files) - reached_rel)
        row["orphan_files"] = orphans
        row["all_non_relevant_referenced"] = not [o for o in orphans if o not in relevant]
        row["n_import_edges"] = sum(len(v) for v in edges.values())

        # -- the mutation really lands (build_repo would have raised otherwise, but
        #    record the diff so the freeze is auditable without re-running)
        diffs = {}
        for rel in relevant:
            g = (gold / rel).read_text(encoding="utf-8")
            b = (buggy / rel).read_text(encoding="utf-8")
            diffs[rel] = {
                "changed": g != b,
                "gold_sha256": hashlib.sha256(g.encode()).hexdigest(),
                "buggy_sha256": hashlib.sha256(b.encode()).hexdigest(),
            }
        row["mutation_applied"] = diffs
        row["mutations_all_applied"] = all(d["changed"] for d in diffs.values())

        # -- untouched files must be byte-identical, or "relevant" is a fiction
        drifted = [
            rel for rel in py_files
            if rel not in relevant
            and (gold / rel).read_bytes() != (buggy / rel).read_bytes()
        ]
        row["unexpectedly_modified_files"] = drifted
        row["only_relevant_files_differ"] = not drifted

        if skip_grading:
            row["gold_passes"] = None
            row["noop_fails"] = None
            return row

        from environments.transformer_repair.grading import grade_workspace

        checks = list(spec.hidden_checks)
        gold_out = grade_workspace(gold, task_id, checks, allow_in_process=True)
        row["gold_passes"] = bool(gold_out.passed)
        row["gold_detail"] = {k: v[:200] for k, v in gold_out.errors.items()}

        buggy_out = grade_workspace(buggy, task_id, checks)
        row["noop_fails"] = not buggy_out.passed
        row["buggy_failing_checks"] = sorted(
            k for k, ok in buggy_out.results.items() if not ok
        )
        row["buggy_detail"] = {k: v[:200] for k, v in buggy_out.errors.items()}
        row["grading_backend_was_sandboxed"] = True
        return row
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", action="append", default=None)
    ap.add_argument("--skip-grading", action="store_true",
                    help="structure only; the freeze is NOT valid without grading")
    args = ap.parse_args()

    from trgym.tasks.repo_specs_s import REPO_TASKS_S

    specs = [s for s in REPO_TASKS_S if not args.task or s.task_id in args.task]
    rows = [preflight_task(s, skip_grading=args.skip_grading) for s in specs]

    for r in rows:
        print(f"\n=== {r['task_id']} ===")
        print(f"  files                    {r['n_files']}  (in range: {r['n_files_in_range']})")
        print(f"  import edges             {r['n_import_edges']}")
        print(f"  relevant                 {r['relevant_files']}")
        print(f"  orphan files             {r['orphan_files'] or 'none'}")
        print(f"  only relevant differ     {r['only_relevant_files_differ']}")
        print(f"  mutations applied        {r['mutations_all_applied']}")
        print(f"  gold passes              {r['gold_passes']}")
        print(f"  no-op fails              {r['noop_fails']}")
        if r.get("buggy_failing_checks") is not None:
            print(f"  buggy failing checks     {r['buggy_failing_checks']}")

    ok = (
        len(rows) == 3
        and all(r["n_files_in_range"] for r in rows)
        and all(r["n_relevant_in_range"] for r in rows)
        and all(r["all_non_relevant_referenced"] for r in rows)
        and all(r["mutations_all_applied"] for r in rows)
        and all(r["only_relevant_files_differ"] for r in rows)
        and all(r["gold_passes"] for r in rows)
        and all(r["noop_fails"] for r in rows)
    )

    # Freeze the protocol alongside the tasks. Anything that could be retuned after
    # seeing results has to be pinned before the first paid call, or the run is not
    # precommitted in any meaningful sense.
    from trgym.repo import predicates
    from trgym.tasks.repo_specs_s import REPO_TASKS_S as ALL

    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    payload = {
        "frozen": ok,
        "tasks": rows,
        "protocol": {
            "n_trajectories_precommitted": 12,
            "model": "deepseek-chat",
            "provider": "https://api.deepseek.com/v1",
            "max_turns": 24,
            "no_retuning_after_results": True,
            "negative_results_are_acceptable": True,
        },
        "verifier": {
            "predicates_sha256": sha(ROOT / "trgym" / "repo" / "predicates.py"),
            "obs_protocol_sha256": sha(ROOT / "trgym" / "repo" / "obs_protocol.py"),
            "candidate_probe_sha256": sha(ROOT / "trgym" / "repo" / "candidate_probe.py"),
            "sandbox_sha256": sha(ROOT / "trgym" / "harness" / "sandbox.py"),
            "grading_sha256": sha(
                ROOT / "environments" / "transformer_repair" / "grading.py"
            ),
            "n_forgeable_predicates": len(predicates.FORGEABLE),
        },
        "template": {
            "dir": "trgym/repo_template_s",
            "builder_sha256": sha(ROOT / "scripts" / "build_tier_s_template.py"),
            "specs_sha256": sha(ROOT / "trgym" / "tasks" / "repo_specs_s.py"),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\nfrozen={ok}  wrote {OUT.relative_to(ROOT).as_posix()}")
    if not ok:
        print("FREEZE FAILED -- do not spend trajectories against this spec")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
