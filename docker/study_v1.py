"""Dump the installed `verifiers.v1` source facts needed to answer G1's open questions.

Runs INSIDE a Linux container (v1 imports POSIX-only `fcntl`). Writes files into a
mounted /out rather than printing to stdout: the v1_probe.json corruption (R8) came from
redirecting a container's stdout through PowerShell, which folds stderr into the payload.

Answers, from source rather than signatures:
  Q1 how a concrete Taskset.load() yields tasks
  Q2 whether Task.score returns or mutates Trace
  Q3 what @vf.reward decorates and how weight/priority aggregate
  Q4 where turn/wall-clock budgets live
  Q5 whether a built-in Taskset/Task implementation exists to copy
"""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
from pathlib import Path

OUT = Path("/out")
OUT.mkdir(parents=True, exist_ok=True)

import verifiers  # noqa: E402
from verifiers import v1  # noqa: E402

V1_DIR = Path(inspect.getfile(v1)).parent

report: dict = {"verifiers_version": getattr(verifiers, "__version__", "?")}

# ---------------------------------------------------------------- file inventory
tree = sorted(p.relative_to(V1_DIR).as_posix() for p in V1_DIR.rglob("*.py"))
report["v1_files"] = tree
(OUT / "v1_tree.txt").write_text("\n".join(tree), encoding="utf-8")

# Copy the whole v1 package source out so it can be read on the host without Docker.
src_root = OUT / "v1_src"
for rel in tree:
    dst = src_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text((V1_DIR / rel).read_text(encoding="utf-8"), encoding="utf-8")

# Anything shipped that looks like an example or a test, anywhere in site-packages.
extras = []
for base in {Path(inspect.getfile(verifiers)).parent.parent}:
    for pat in ("**/example*/**/*.py", "**/tests/**/*.py", "**/test_*.py"):
        for p in base.glob(pat):
            if "verifiers" in p.as_posix():
                extras.append(p.as_posix())
report["shipped_examples_and_tests"] = sorted(set(extras))

# --------------------------------------------------- Q5: concrete implementations
# Import every v1 submodule and record real subclasses, so "is there a built-in to
# copy" is answered by the class graph rather than by guessing from names.
bases = {"Taskset": v1.Taskset, "Task": v1.Task, "Harness": v1.Harness,
         "Runtime": v1.Runtime}
concrete: dict[str, list[dict]] = {k: [] for k in bases}
import_errors: dict[str, str] = {}

for mod in pkgutil.walk_packages([str(V1_DIR)], prefix="verifiers.v1."):
    try:
        m = importlib.import_module(mod.name)
    except Exception as exc:  # noqa: BLE001 - an unimportable submodule is data
        import_errors[mod.name] = f"{type(exc).__name__}: {exc}"
        continue
    for name, obj in vars(m).items():
        if not inspect.isclass(obj) or obj.__module__ != mod.name:
            continue
        for label, base in bases.items():
            if issubclass(obj, base) and obj is not base:
                concrete[label].append({
                    "class": name, "module": mod.name,
                    "abstract": bool(getattr(obj, "__abstractmethods__", None)),
                    "abstract_methods": sorted(getattr(obj, "__abstractmethods__", [])),
                    "mro": [c.__name__ for c in obj.__mro__[1:4]],
                })
report["concrete_implementations"] = concrete
report["submodule_import_errors"] = import_errors

# ------------------------------------------ Q1/Q2/Q3/Q4: read the defining source
def grab(obj, label: str) -> None:
    try:
        text = inspect.getsource(obj)
    except (OSError, TypeError) as exc:
        report.setdefault("source_unavailable", {})[label] = str(exc)
        return
    (OUT / f"src_{label}.py").write_text(text, encoding="utf-8")

for label, base in bases.items():
    grab(base, label.lower())
grab(v1.Trace, "trace")
grab(v1.TaskData, "taskdata")
grab(v1.reward, "reward_decorator")
grab(v1.metric, "metric_decorator")
if hasattr(v1, "ACP"):
    grab(v1.ACP, "acp")

# Q4: budgets. `limiters` is the fcntl importer, so it is the prime suspect.
try:
    from verifiers.v1.runtimes import limiters
    (OUT / "src_limiters.py").write_text(
        inspect.getsource(limiters), encoding="utf-8")
    report["limiters_public"] = [n for n in dir(limiters) if not n.startswith("_")]
except Exception as exc:  # noqa: BLE001
    report["limiters_error"] = f"{type(exc).__name__}: {exc}"

# Q3: how rewards are discovered and aggregated. Find whoever reads the marker that
# @vf.reward sets, since that is the aggregation site.
marker_names: list[str] = []
try:
    def _probe() -> float:
        return 1.0
    decorated = v1.reward(_probe, weight=2.0, priority=3) if callable(v1.reward) else None
    if decorated is not None:
        marker_names = [a for a in dir(decorated) if not a.startswith("__")]
        report["reward_marker_attrs"] = {
            a: repr(getattr(decorated, a)) for a in marker_names
        }
except Exception as exc:  # noqa: BLE001
    report["reward_probe_error"] = f"{type(exc).__name__}: {exc}"

# Grep the package for the marker attribute to locate the aggregator.
hits: dict[str, list[str]] = {}
needles = [n for n in marker_names if "reward" in n.lower() or "weight" in n.lower()]
needles += ["__vf_reward__", "record_reward", "toolsets", "def load", "INFINITE"]
for rel in tree:
    text = (V1_DIR / rel).read_text(encoding="utf-8")
    for needle in set(needles):
        if needle in text:
            hits.setdefault(needle, []).append(rel)
report["needle_hits"] = {k: sorted(v) for k, v in hits.items()}

# Trace fields: what a rollout record actually carries.
try:
    report["trace_fields"] = sorted(v1.Trace.model_fields)
    report["taskdata_fields"] = sorted(v1.TaskData.model_fields)
except Exception as exc:  # noqa: BLE001
    report["pydantic_fields_error"] = f"{type(exc).__name__}: {exc}"

(OUT / "v1_study.json").write_text(
    json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
print("WROTE", sorted(p.name for p in OUT.iterdir()))
