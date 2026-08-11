"""Probe the real verifiers.v1 surface from inside Linux.

`verifiers.v1` imports `fcntl` transitively and therefore cannot load on Windows
at all. Everything we need to know about the v1 lifecycle has to be discovered
from a Linux container. This script is the discovery tool; its output drives the
G1 migration design.
"""

from __future__ import annotations

import inspect
import json
import pkgutil

out: dict = {}

import verifiers  # noqa: E402

out["verifiers_version"] = getattr(verifiers, "__version__", "?")

try:
    import verifiers.v1 as v1
except Exception as exc:  # noqa: BLE001
    out["v1_import"] = f"FAILED {type(exc).__name__}: {exc}"
    print(json.dumps(out, indent=2))
    raise SystemExit(1)

out["v1_import"] = "OK"
out["v1_exports"] = sorted(n for n in dir(v1) if not n.startswith("_"))
out["v1_submodules"] = sorted(m.name for m in pkgutil.iter_modules(v1.__path__))


def sig(obj, name: str) -> str:
    try:
        return f"{name}{inspect.signature(obj)}"
    except Exception:  # noqa: BLE001
        return f"{name}(?)"


def describe(cls_name: str) -> dict | None:
    cls = getattr(v1, cls_name, None)
    if cls is None:
        return None
    info: dict = {"kind": type(cls).__name__}
    if inspect.isclass(cls):
        info["init"] = sig(cls.__init__, "__init__")
        info["public"] = sorted(n for n in dir(cls) if not n.startswith("_"))
        methods = {}
        for n in info["public"]:
            m = getattr(cls, n, None)
            if callable(m):
                methods[n] = sig(m, n)
        info["methods"] = methods
        if hasattr(cls, "model_fields"):
            info["model_fields"] = list(cls.model_fields)
        try:
            import dataclasses

            if dataclasses.is_dataclass(cls):
                info["dataclass_fields"] = [f.name for f in dataclasses.fields(cls)]
        except Exception:  # noqa: BLE001
            pass
    elif callable(cls):
        info["signature"] = sig(cls, cls_name)
    return info


for name in (
    "Task", "TaskData", "Taskset", "Harness", "HarnessSession", "Runtime",
    "Trace", "Rubric", "Environment", "reward", "metric", "setup", "teardown",
    "ProgramResult", "ACP",
):
    d = describe(name)
    if d is not None:
        out.setdefault("api", {})[name] = d

# Which runtimes ship?
try:
    import verifiers.v1.runtimes as rt

    out["runtimes"] = sorted(n for n in dir(rt) if not n.startswith("_"))
    out["runtime_submodules"] = sorted(m.name for m in pkgutil.iter_modules(rt.__path__))
except Exception as exc:  # noqa: BLE001
    out["runtimes"] = f"FAILED {type(exc).__name__}: {exc}"

# CLI entry points
try:
    import importlib.metadata as md

    eps = md.entry_points()
    out["console_scripts"] = sorted(
        e.name for e in eps.select(group="console_scripts") if "vf" in e.name or "verifiers" in e.value
    )
except Exception as exc:  # noqa: BLE001
    out["console_scripts"] = f"FAILED {exc}"

print(json.dumps(out, indent=2, default=str))
