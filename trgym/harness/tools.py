"""Tools a candidate can call during a multi-turn repair session.

Every tool is a pure function of (workspace, args) -> ToolResult, with no hidden
state, so a session is replayable from its transcript. Budgets are enforced here
rather than trusted to the model: turns, wall clock, output size and total CPU
time are all capped, and exceeding a cap ends the episode instead of silently
truncating.

Nothing here is a security boundary. Path containment stops accidents and honest
mistakes; it does not stop a determined escape. Real isolation is the container
in `trgym/harness/sandbox.py`. See SANDBOX_DESIGN.md.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

MAX_OUTPUT_CHARS = 6000
MAX_READ_CHARS = 20000
DEFAULT_COMMAND_TIMEOUT_S = 180


@dataclass
class ToolResult:
    ok: bool
    output: str
    truncated: bool = False
    duration_s: float = 0.0

    def render(self) -> str:
        body = self.output
        if self.truncated:
            body += f"\n\n[output truncated to {MAX_OUTPUT_CHARS} characters]"
        return body


@dataclass
class Budget:
    max_turns: int = 14
    max_wall_s: float = 900.0
    max_command_s: float = DEFAULT_COMMAND_TIMEOUT_S
    max_commands: int = 24

    turns_used: int = 0
    commands_used: int = 0
    started_at: float = field(default_factory=time.perf_counter)

    def exhausted(self) -> str | None:
        if self.turns_used >= self.max_turns:
            return f"turn budget exhausted ({self.max_turns})"
        if self.commands_used >= self.max_commands:
            return f"command budget exhausted ({self.max_commands})"
        elapsed = time.perf_counter() - self.started_at
        if elapsed >= self.max_wall_s:
            return f"wall-clock budget exhausted ({self.max_wall_s:.0f}s)"
        return None


def _clip(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text, False
    return text[:MAX_OUTPUT_CHARS], True


def _resolve(workspace: Path, rel: str) -> Path:
    """Resolve `rel` inside the workspace, refusing anything that escapes it."""
    root = Path(workspace).resolve()
    target = (root / rel).resolve()
    if root not in target.parents and target != root:
        raise ValueError(f"path {rel!r} is outside the workspace")
    return target


# --------------------------------------------------------------------------- #
# Read-only tools
# --------------------------------------------------------------------------- #
def list_files(workspace: Path, pattern: str = "**/*.py") -> ToolResult:
    root = Path(workspace).resolve()
    paths = sorted(
        p.relative_to(root).as_posix()
        for p in root.glob(pattern)
        if p.is_file() and "__pycache__" not in p.parts
    )
    lines = [f"{p}  ({(root / p).stat().st_size} bytes)" for p in paths]
    body, trunc = _clip("\n".join(lines) or "(no files matched)")
    return ToolResult(ok=True, output=body, truncated=trunc)


def read_file(workspace: Path, path: str, start: int = 1, end: int | None = None) -> ToolResult:
    try:
        target = _resolve(workspace, path)
    except ValueError as exc:
        return ToolResult(ok=False, output=str(exc))
    if not target.is_file():
        return ToolResult(ok=False, output=f"{path} does not exist")

    lines = target.read_text(encoding="utf-8").splitlines()
    end = len(lines) if end is None else min(end, len(lines))
    start = max(1, start)
    numbered = [f"{i:4d} | {lines[i - 1]}" for i in range(start, end + 1)]
    body, trunc = _clip("\n".join(numbered)[:MAX_READ_CHARS])
    return ToolResult(ok=True, output=body, truncated=trunc)


# --------------------------------------------------------------------------- #
# Execution tools
# --------------------------------------------------------------------------- #
ALLOWED_COMMANDS = {
    "run_visible_tests": [sys.executable, "-m", "pytest", "tests/test_visible.py", "-q",
                          "-p", "no:cacheprovider"],
    "run_training": [sys.executable, "-m", "tinygpt.train", "--steps", "40"],
    "run_training_short": [sys.executable, "-m", "tinygpt.train", "--steps", "8"],
    "run_training_json": [sys.executable, "-m", "tinygpt.train", "--steps", "40", "--json"],
}


def run_command(
    workspace: Path, name: str, budget: Budget, repo_root: Path | None = None
) -> ToolResult:
    """Run one of a fixed set of commands. Arbitrary shell is deliberately absent.

    A fixed allowlist keeps the action space small enough to reason about and
    removes a whole class of prompt-injection style escapes. If a task needs a
    new capability, it gets a named tool, not a shell.
    """
    if name not in ALLOWED_COMMANDS:
        return ToolResult(
            ok=False,
            output=f"unknown command {name!r}; available: {sorted(ALLOWED_COMMANDS)}",
        )

    import os

    env = dict(os.environ)
    # The visible-test runner imports trgym; nothing else needs the repo root.
    env["PYTHONPATH"] = str(repo_root) if repo_root else ""
    env.pop("DEEPSEEK_API_KEY", None)
    env.pop("TINKER_API_KEY", None)

    started = time.perf_counter()
    budget.commands_used += 1
    try:
        proc = subprocess.run(
            ALLOWED_COMMANDS[name],
            cwd=str(Path(workspace).resolve()),
            capture_output=True,
            text=True,
            timeout=budget.max_command_s,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            ok=False,
            output=f"command timed out after {budget.max_command_s:.0f}s",
            duration_s=budget.max_command_s,
        )
    duration = time.perf_counter() - started
    combined = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
    body, trunc = _clip(combined.strip() or "(no output)")
    return ToolResult(
        ok=proc.returncode == 0, output=body, truncated=trunc, duration_s=duration
    )


# --------------------------------------------------------------------------- #
# Editing
# --------------------------------------------------------------------------- #
def apply_patch(workspace: Path, path: str, diff_text: str) -> ToolResult:
    """Apply a unified diff to one file inside the workspace."""
    from trgym.patching import PatchError, apply_unified_diff, last_fuzz

    try:
        target = _resolve(workspace, path)
    except ValueError as exc:
        return ToolResult(ok=False, output=str(exc))
    if not target.is_file():
        return ToolResult(ok=False, output=f"{path} does not exist")
    if not path.startswith("tinygpt/"):
        return ToolResult(
            ok=False,
            output=f"{path} is not editable; only files under tinygpt/ can be changed",
        )

    original = target.read_text(encoding="utf-8")
    try:
        patched = apply_unified_diff(original, diff_text)
    except PatchError as exc:
        return ToolResult(ok=False, output=f"patch did not apply: {exc}")

    target.write_text(patched, encoding="utf-8")
    fuzz = last_fuzz()
    note = f" (context fuzz: {fuzz} line(s))" if fuzz else ""
    return ToolResult(ok=True, output=f"applied to {path}{note}")


TOOL_SPECS = [
    {
        "name": "list_files",
        "description": "List Python files in the repository with their sizes.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "glob, default **/*.py"}
            },
        },
    },
    {
        "name": "read_file",
        "description": "Read a file with line numbers. Use start/end to page through it.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start": {"type": "integer"},
                "end": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "run_command",
        "description": (
            "Run one of: run_visible_tests, run_training (40 steps), "
            "run_training_short (8 steps), run_training_json (history as JSON)."
        ),
        "parameters": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "apply_patch",
        "description": (
            "Apply a unified diff to one file under tinygpt/. Include at least one "
            "line of context per hunk."
        ),
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "diff": {"type": "string"}},
            "required": ["path", "diff"],
        },
    },
    {
        "name": "submit",
        "description": "Finish the episode and grade the current state of the repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "what you changed and why"}
            },
        },
    },
]
