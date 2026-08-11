"""G7: prove the project reproduces from a clean clone, not from this working tree.

The failure this exists to catch is subtle and common: a "fresh clone" that passes because
the interpreter is still resolving imports out of the original checkout. An editable
install, a stray `PYTHONPATH`, a `.pth` file or a cached `__pycache__` will all do it, and
the run looks perfectly green while proving nothing.

So the check is not "did the tests pass in the clone". It is:

  1. clone to a directory that is not this tree, via a real `git clone`;
  2. build an environment there from the committed lockfile;
  3. run exactly the commands the README documents -- parsed from README.md between the
     REPRO markers, so the two cannot drift apart silently;
  4. ask the clone's own interpreter where each module actually came from, and fail if any
     `__file__` resolves back into the original tree.

Nothing is pushed anywhere. The clone's origin is a local path.

Run: python scripts/fresh_clone_repro.py [--keep]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "fresh_clone_run.json"

# Modules whose provenance is interrogated inside the clone.
PROBE_MODULES = ("trgym", "trgym.repo.predicates", "trgym.repo.obs_protocol",
                 "trgym.harness.sandbox", "environments.transformer_repair.grading")

# `environments.transformer_repair` pulls in `verifiers.v1`, which imports `fcntl` and
# therefore cannot load on Windows at all. Its import failure in a Windows clone is the
# documented platform limitation, not a packaging defect, so it is recorded rather than
# treated as a leak -- and it is still probed, because if it ever DID import here it
# would mean something had gone wrong with the isolation.
PLATFORM_LIMITED = {"environments.transformer_repair.grading"}

REPRO_RE = re.compile(
    r"<!--\s*REPRO-BEGIN.*?-->\s*```(?:bash|powershell)?\n(.*?)```", re.S
)


def readme_commands() -> list[str]:
    """The commands the README tells a reader to run."""
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    match = REPRO_RE.search(text)
    if not match:
        raise SystemExit("README.md has no REPRO-BEGIN block; nothing to verify against")
    return [
        line.strip()
        for line in match.group(1).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def run(cmd: list[str], cwd: Path, env: dict, timeout: int = 3600) -> dict:
    started = time.perf_counter()
    proc = subprocess.run(
        cmd, cwd=str(cwd), env=env, capture_output=True, text=True, timeout=timeout
    )
    return {
        "cmd": " ".join(cmd),
        "exit_code": proc.returncode,
        "seconds": round(time.perf_counter() - started, 2),
        "stdout_tail": proc.stdout[-3000:],
        "stderr_tail": proc.stderr[-2000:],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="do not delete the clone")
    args = ap.parse_args()

    commands = readme_commands()
    clone_parent = Path(tempfile.mkdtemp(prefix="trgym_freshclone_"))
    clone = clone_parent / "transformer-repair-gym"

    # A clean environment. Inheriting PYTHONPATH is precisely how a fresh-clone check
    # accidentally re-imports the original tree and reports success.
    env = {
        k: v for k, v in os.environ.items()
        if k.upper() not in {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV"}
    }
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    steps: list[dict] = []
    try:
        steps.append(run(["git", "clone", "--quiet", str(ROOT), str(clone)],
                         cwd=clone_parent, env=env))
        if steps[-1]["exit_code"] != 0:
            raise RuntimeError("git clone failed")

        # `final_acceptance.py` is run in the clone but is NOT fatal to reproduction.
        # It reports the project's gate state, and a truthful "6/10" is a correct
        # reproduction, not a broken one. Requiring it to pass here would also be
        # circular: G7 needs this artifact to be clean, and the gate needs G7. Whether
        # acceptance PASSES in a clean room is G9 stage E, recorded separately below.
        advisory = {"final_acceptance"}
        for line in commands:
            step = run(line.split(), cwd=clone, env=env)
            step["advisory"] = any(token in line for token in advisory)
            steps.append(step)

        # Where did the clone's interpreter actually load these from?
        probe_src = (
            "import json, importlib, sys\n"
            "out = {}\n"
            f"for name in {list(PROBE_MODULES)!r}:\n"
            "    try:\n"
            "        out[name] = getattr(importlib.import_module(name), '__file__', None)\n"
            "    except Exception as exc:\n"
            "        out[name] = f'IMPORT FAILED: {type(exc).__name__}: {exc}'\n"
            "out['sys.path'] = [p for p in sys.path if p]\n"
            "print('<<<PROBE>>>' + json.dumps(out))\n"
        )
        probe = run(["uv", "run", "python", "-c", probe_src], cwd=clone, env=env)
        steps.append(probe)

        module_files: dict = {}
        marker = "<<<PROBE>>>"
        if marker in probe["stdout_tail"]:
            payload = probe["stdout_tail"].rsplit(marker, 1)[1].strip().splitlines()[0]
            module_files = json.loads(payload)

        def leaks_into_original(value: str) -> bool:
            return str(ROOT).lower().replace("\\", "/") in str(value).lower().replace("\\", "/")

        leaked = {
            k: v for k, v in module_files.items()
            if k != "sys.path" and v and leaks_into_original(v)
        }
        path_leaks = [p for p in module_files.get("sys.path", []) if leaks_into_original(p)]

        failed = [s for s in steps if s["exit_code"] != 0 and not s.get("advisory")]
        acceptance = next(
            (s for s in steps if s.get("advisory") and "final_acceptance" in s["cmd"]), None
        )
        record = {
            # G9 stage E: acceptance must PASS in a clean room with no reuse of the
            # original venv, PYTHONPATH or bytecode cache.
            "acceptance_in_clone_exit_code": acceptance["exit_code"] if acceptance else None,
            "clean_room_acceptance_passed": bool(acceptance and acceptance["exit_code"] == 0),
            "clone_path": str(clone),
            "origin": str(ROOT),
            "commit": run(["git", "rev-parse", "HEAD"], cwd=clone, env=env)["stdout_tail"].strip(),
            "commands_from_readme": commands,
            "commands_match_readme": [s["cmd"] for s in steps[1 : 1 + len(commands)]]
            == commands,
            "steps": steps,
            "module_files": {k: v for k, v in module_files.items() if k != "sys.path"},
            "platform_limited_modules": sorted(PLATFORM_LIMITED),
            "modules_importable_in_clone": sorted(
                k for k, v in module_files.items()
                if k != "sys.path" and v and not str(v).startswith("IMPORT FAILED")
            ),
            "sys_path_entries_into_original_tree": path_leaks,
            "modules_resolving_into_original_tree": leaked,
            "pushed_anywhere": False,
            "exit_code": 0 if (not failed and not leaked and not path_leaks) else 1,
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")

        print(f"clone            {clone}")
        print(f"commit           {record['commit']}")
        print(f"commands match   {record['commands_match_readme']}")
        for s in steps:
            print(f"  [{s['exit_code']}] {s['seconds']:>7.2f}s  {s['cmd'][:88]}")
        print(f"module __file__  {json.dumps(record['module_files'], indent=2)[:1200]}")
        if leaked:
            print(f"LEAKED into the original tree: {leaked}")
        if path_leaks:
            print(f"sys.path points into the original tree: {path_leaks}")
        print(f"\nexit_code {record['exit_code']}  wrote {OUT.relative_to(ROOT).as_posix()}")
        return record["exit_code"]
    finally:
        if not args.keep:
            shutil.rmtree(clone_parent, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
