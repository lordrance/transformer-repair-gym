"""The static gates must judge executable code, not prose about it.

G6 measured a 15.6 % submission-level false-positive rate against 32 real functions from
PyTorch and numpy, and every one of the five misfires was a docstring:

  * `MultiheadAttention` / `TransformerEncoderLayer` -> `dynamic_execution`, because their
    docstrings say "training is disabled (using ``.eval()``)";
  * `Embedding` / `CrossEntropyLoss` / `default_collate` -> `hardcoded_tensor`, from
    `tensor([...])` in docstring *examples*.

`code_only()` blanks comments and string literals before pattern matching, which took that
rate to 0.0 %. See PROTOCOL_CHANGELOG R13.

The danger of that fix is obvious: a gate that ignores too much stops protecting anything.
So these tests come in pairs -- prose must NOT fire, real code MUST still fire. The
exploit suite (`scripts/audit_cheats.py`, 7/7) is the other half of that guarantee.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from trgym.verifier.anticheat import (
    _BIG_LITERAL,
    _DYNAMIC,
    _REF_IMPORT,
    code_only,
    gate_no_dynamic_execution,
    gate_no_reference_import,
    gate_no_shape_special_casing,
    gate_no_silent_exception_swallowing,
)

# The four content-only gates never read spec/ws; the audit relies on that.
CONTENT_GATES = (
    gate_no_reference_import,
    gate_no_dynamic_execution,
    gate_no_shape_special_casing,
    gate_no_silent_exception_swallowing,
)


def test_content_gates_ignore_spec_and_workspace() -> None:
    """`scripts/heuristic_gate_fp_audit.py` passes None for both; pin that."""
    src = "def f(x):\n    return x + 1\n"
    for gate in CONTENT_GATES:
        assert gate(src, None, None) == []  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# code_only: blanks prose, preserves code and line numbers
# --------------------------------------------------------------------------- #
def test_code_only_blanks_docstrings_and_comments() -> None:
    src = (
        'def f(x):\n'
        '    """Disable training with .eval() and pass tensor([1.0, 2.0]).\n'
        '\n'
        '    Example: exec("whatever")\n'
        '    """\n'
        '    # eval() in a comment too\n'
        '    return x\n'
    )
    stripped = code_only(src)
    assert not _DYNAMIC.search(stripped), "docstring/comment prose still matches"
    assert not _BIG_LITERAL.search(stripped)
    assert "return x" in stripped, "executable code was destroyed"
    # Line numbers must survive, or evidence reporting points at the wrong line.
    assert len(stripped.splitlines()) == len(src.splitlines())


def test_code_only_keeps_real_violations_visible() -> None:
    src = (
        'def f(x):\n'
        '    """A perfectly innocent docstring."""\n'
        '    exec("import trgym.reference as r")\n'
        '    return x\n'
    )
    stripped = code_only(src)
    assert _DYNAMIC.search(stripped), "a real exec() call must survive stripping"


def test_code_only_survives_unparseable_source() -> None:
    """Broken source is scanned as-is: fail loud, not silent."""
    broken = "def f(:\n    this is not python\n"
    assert code_only(broken) == broken


# --------------------------------------------------------------------------- #
# The gates themselves: paired negative / positive
# --------------------------------------------------------------------------- #
DOCSTRING_ONLY = (
    'def scaled_dot_product(q, k, v):\n'
    '    """Attention.\n'
    '\n'
    '    Note: dropout is skipped when training is disabled (using ``.eval()``).\n'
    '\n'
    '    Example::\n'
    '        >>> torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])\n'
    '    """\n'
    '    return q @ k.transpose(-2, -1) @ v\n'
)


def test_no_gate_fires_on_prose_only(tmp_path: Path) -> None:
    for gate in CONTENT_GATES:
        assert gate(DOCSTRING_ONLY, None, None) == [], (  # type: ignore[arg-type]
            f"{gate.__name__} fired on documentation"
        )


@pytest.mark.parametrize(
    "gate, source",
    [
        (gate_no_reference_import, "from trgym.reference import gold\n"),
        (gate_no_dynamic_execution, "y = eval('1+1')\n"),
        (
            gate_no_shape_special_casing,
            "def f(x):\n"
            "    if x.shape[1] == 8:\n"
            "        return correct(x)\n"
            "    return wrong(x)\n",
        ),
        (
            gate_no_silent_exception_swallowing,
            "def f(x):\n    try:\n        return g(x)\n    except Exception:\n        pass\n",
        ),
    ],
)
def test_each_gate_still_fires_on_a_real_violation(gate, source) -> None:
    """Every gate must remain able to fire. A silent gate is worse than no gate."""
    violations = gate(source, None, None)
    assert violations, f"{gate.__name__} no longer detects its own exploit"


def test_evidence_reports_the_original_line_text() -> None:
    """Blanking must not leak into the human-readable evidence string."""
    src = (
        'def f(x):\n'
        '    """doc."""\n'
        '    return eval("2")   # the real one\n'
    )
    violations = gate_no_dynamic_execution(src, None, None)  # type: ignore[arg-type]
    assert violations
    evidence = violations[0].evidence
    assert "line 3" in evidence, evidence
    assert 'eval("2")' in evidence, f"evidence shows blanked text: {evidence!r}"
