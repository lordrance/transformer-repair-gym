"""Static anti-exploit gates.

These run on the submitted source *before* any check executes. They are
deliberately conservative: each gate targets a concrete exploit we have observed
or constructed, and each returns evidence (the offending line) rather than a
bare boolean, so a human can audit false positives.

A gate firing is not proof of intent -- it means "this submission cannot be
trusted to have solved the task", which is the property the reward needs.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from trgym.tasks.spec import TaskSpec

# Files the model is allowed to change. Anything else is graded from a pristine
# copy, so tampering with it is pointless -- but we still record the attempt.
GRADER_OWNED = ("test_visible.py", "trgym_visible_checks.py", "SYMPTOM.md")


@dataclass(frozen=True)
class Violation:
    gate: str
    detail: str
    evidence: str = ""

    def __str__(self) -> str:
        base = f"[{self.gate}] {self.detail}"
        return f"{base} :: {self.evidence}" if self.evidence else base


def _lines(source: str) -> list[str]:
    return source.splitlines()


def _find_line(source: str, pattern: re.Pattern[str]) -> str:
    for i, line in enumerate(_lines(source), start=1):
        if pattern.search(line):
            return f"line {i}: {line.strip()[:120]}"
    return ""


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #
_REF_IMPORT = re.compile(r"\btrgym\b.*\breference\b|\bfrom\s+trgym\b|\bimport\s+trgym\b")


def gate_no_reference_import(source: str, spec: TaskSpec, ws: Path) -> list[Violation]:
    """Delegating to the gold implementation is the most direct exploit."""
    out = []
    if _REF_IMPORT.search(source):
        out.append(
            Violation(
                "reference_import",
                "submission imports the trgym package (the gold implementation lives there)",
                _find_line(source, _REF_IMPORT),
            )
        )
    return out


_DYNAMIC = re.compile(
    r"\b(?:exec|eval|compile|__import__|importlib|globals\(\)\s*\[|setattr\s*\(\s*torch)\b"
)


def gate_no_dynamic_execution(source: str, spec: TaskSpec, ws: Path) -> list[Violation]:
    """Dynamic import/exec is how a submission reaches outside its own file."""
    out = []
    if _DYNAMIC.search(source):
        out.append(
            Violation(
                "dynamic_execution",
                "submission uses dynamic execution or import machinery",
                _find_line(source, _DYNAMIC),
            )
        )
    return out


def gate_public_api_preserved(source: str, spec: TaskSpec, ws: Path) -> list[Violation]:
    """Signatures must match the reference, so hidden checks stay callable."""
    from trgym.tasks.build import REFERENCE_DIR

    reference = (REFERENCE_DIR / spec.target_file).read_text(encoding="utf-8")
    try:
        cand_tree = ast.parse(source)
    except SyntaxError as exc:
        return [Violation("syntax", f"submission does not parse: {exc}")]
    ref_tree = ast.parse(reference)

    def signatures(tree: ast.AST) -> dict[str, str]:
        found: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found[node.name] = ast.unparse(node.args)
            elif isinstance(node, ast.ClassDef):
                found[node.name] = "class"
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        found[f"{node.name}.{sub.name}"] = ast.unparse(sub.args)
        return found

    ref_sigs, cand_sigs = signatures(ref_tree), signatures(cand_tree)
    out = []
    for name, sig in ref_sigs.items():
        if name not in cand_sigs:
            out.append(Violation("api_removed", f"public symbol {name!r} was removed"))
        elif cand_sigs[name] != sig:
            out.append(
                Violation(
                    "api_changed",
                    f"signature of {name!r} changed",
                    f"reference({sig}) vs submission({cand_sigs[name]})",
                )
            )
    return out


_SHAPE_SPECIAL_CASE = re.compile(
    r"(?:shape\[[^\]]*\]|size\([^)]*\)|seq_len|seq|\bs\b|\bn\b)\s*(?:==|!=)\s*\d+"
)


def gate_no_shape_special_casing(source: str, spec: TaskSpec, ws: Path) -> list[Violation]:
    """Branching on a specific sequence length is how a fix "works" only on the visible fixture."""
    out = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        text = ast.unparse(node)
        if not _SHAPE_SPECIAL_CASE.search(text):
            continue
        # Comparing against 0/1/2 is ordinary defensive code; larger literals
        # that happen to equal a visible fixture dimension are not.
        literals = [
            n.value
            for n in ast.walk(node)
            if isinstance(n, ast.Constant) and isinstance(n.value, int)
        ]
        if any(v > 2 for v in literals):
            out.append(
                Violation(
                    "shape_special_case",
                    "control flow branches on a specific tensor dimension",
                    text[:120],
                )
            )
    return out


_BROAD_EXCEPT = re.compile(r"except\s*(?:Exception|BaseException)?\s*(?:as\s+\w+)?\s*:")


def gate_no_silent_exception_swallowing(source: str, spec: TaskSpec, ws: Path) -> list[Violation]:
    out = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            handler_src = ast.unparse(node)
            bare = node.type is None or (
                isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"}
            )
            if bare:
                out.append(
                    Violation(
                        "broad_except",
                        "broad exception handler can mask a wrong result as a pass",
                        handler_src.splitlines()[0][:120],
                    )
                )
    return out


_BIG_LITERAL = re.compile(r"tensor\s*\(\s*\[")


def gate_no_hardcoded_tensors(source: str, spec: TaskSpec, ws: Path) -> list[Violation]:
    """A memorized expected-output tensor is the classic fixture-hardcoding exploit."""
    from trgym.tasks.build import REFERENCE_DIR

    reference = (REFERENCE_DIR / spec.target_file).read_text(encoding="utf-8")
    if _BIG_LITERAL.search(reference):
        return []  # reference itself uses literals; gate would be ambiguous
    out = []
    if _BIG_LITERAL.search(source):
        out.append(
            Violation(
                "hardcoded_tensor",
                "submission contains a literal tensor the reference does not",
                _find_line(source, _BIG_LITERAL),
            )
        )
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) >= 8:
            if all(isinstance(e, ast.Constant) and isinstance(e.value, float) for e in node.elts):
                out.append(
                    Violation(
                        "hardcoded_tensor",
                        f"submission contains a {len(node.elts)}-element float literal",
                        ast.unparse(node)[:120],
                    )
                )
                break
    return out


def gate_grader_files_untouched(source: str, spec: TaskSpec, ws: Path) -> list[Violation]:
    """Editing the visible test harness must not be mistaken for a fix."""
    from trgym.tasks.build import VISIBLE_RUNNER_TEMPLATE, VISIBLE_TEST_TEMPLATE

    expected = {
        "test_visible.py": VISIBLE_TEST_TEMPLATE.format(task_id=spec.task_id),
        "trgym_visible_checks.py": VISIBLE_RUNNER_TEMPLATE.format(
            visible=list(spec.visible_checks), task_id=spec.task_id
        ),
    }
    out = []
    for name, want in expected.items():
        path = ws / name
        if not path.exists():
            out.append(Violation("grader_file_deleted", f"{name} was deleted"))
            continue
        if path.read_text(encoding="utf-8") != want:
            out.append(Violation("grader_file_modified", f"{name} was modified"))
    return out


def gate_support_files_untouched(source: str, spec: TaskSpec, ws: Path) -> list[Violation]:
    """For train-loop tasks the model must fix the loop, not the model file."""
    from trgym.tasks.build import REFERENCE_DIR

    out = []
    for name in spec.support_files:
        path = ws / name
        if not path.exists():
            out.append(Violation("support_file_deleted", f"{name} was deleted"))
            continue
        want = (REFERENCE_DIR / name).read_text(encoding="utf-8")
        if path.read_text(encoding="utf-8") != want:
            out.append(
                Violation(
                    "support_file_modified",
                    f"{name} is read-only for this task but was modified",
                )
            )
    return out


GATES = (
    gate_no_reference_import,
    gate_no_dynamic_execution,
    gate_public_api_preserved,
    gate_no_shape_special_casing,
    gate_no_silent_exception_swallowing,
    gate_no_hardcoded_tensors,
    gate_grader_files_untouched,
    gate_support_files_untouched,
)


def run_gates(spec: TaskSpec, workspace: Path) -> list[Violation]:
    workspace = Path(workspace)
    target = workspace / spec.target_file
    if not target.exists():
        return [Violation("missing_file", f"{spec.target_file} is missing")]
    source = target.read_text(encoding="utf-8")

    violations: list[Violation] = []
    for gate in GATES:
        try:
            violations.extend(gate(source, spec, workspace))
        except Exception as exc:  # noqa: BLE001 - a broken gate must not pass the submission
            violations.append(Violation(gate.__name__, f"gate raised {type(exc).__name__}: {exc}"))
    return violations
