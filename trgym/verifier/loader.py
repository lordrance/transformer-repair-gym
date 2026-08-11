"""Load candidate modules from a workspace directory.

Each load gets a unique module name so repeated grading in one process cannot
pick up a stale `sys.modules` entry.
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path
from types import ModuleType

_COUNTER = itertools.count()


class CandidateLoadError(RuntimeError):
    """Raised when the candidate file cannot even be imported."""


def load_module(path: Path, *, extra_sys_path: Path | None = None) -> ModuleType:
    path = Path(path)
    if not path.exists():
        raise CandidateLoadError(f"{path.name} is missing from the workspace")

    name = f"_trgym_candidate_{next(_COUNTER)}_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise CandidateLoadError(f"cannot build import spec for {path}")

    module = importlib.util.module_from_spec(spec)
    added = False
    if extra_sys_path is not None and str(extra_sys_path) not in sys.path:
        sys.path.insert(0, str(extra_sys_path))
        added = True
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - candidate code is arbitrary
        sys.modules.pop(name, None)
        raise CandidateLoadError(f"{path.name} raised on import: {type(exc).__name__}: {exc}")
    finally:
        if added:
            sys.path.remove(str(extra_sys_path))
    return module


def require(module: ModuleType, *names: str) -> None:
    """Assert the candidate still exposes the public API."""
    missing = [n for n in names if not hasattr(module, n)]
    if missing:
        raise CandidateLoadError(
            f"{module.__name__.split('_')[-1]}: public API removed: {', '.join(missing)}"
        )
