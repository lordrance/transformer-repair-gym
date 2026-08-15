"""Pinned digests must mean the same thing on every platform.

Several gates re-check a SHA-256 that was pinned elsewhere: `final_acceptance.py` G8
compares the digest printed in the research report against the summary on disk, and G1
rejects `v1_runtime_evidence.json` whose `grading_sha256` no longer matches `grading.py`.
Those checks are supposed to catch *stale evidence*.

Hashing raw bytes made them catch something else. Git stores text with LF and a Windows
checkout materialises CRLF, so the same unchanged file hashed one way on the machine that
pinned the digest and another on a Linux CI runner. The gates then reported stale evidence
that was perfectly current -- the worst kind of failure, because it teaches the reader to
ignore the alarm.

`final_acceptance.py` cannot import `trgym` (it must run in a bare environment), so it
carries its own copy of the normalisation. These tests exist because two copies of a rule
are two chances to drift.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

from trgym.provenance import content_sha256, content_sha256_of, normalize

REPO_ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE_PY = REPO_ROOT / "scripts" / "final_acceptance.py"


@pytest.fixture(scope="module")
def acceptance():
    """Load the gate script by path, without running it.

    The module must be registered in `sys.modules` *before* `exec_module`: the script uses
    `from __future__ import annotations`, so `@dataclass` resolves its string annotations
    by looking its own module up there, and a module that is not registered yet resolves
    to None.
    """
    import sys

    name = "_acceptance_under_test"
    spec = importlib.util.spec_from_file_location(name, ACCEPTANCE_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


# --------------------------------------------------------------------------- #
# The property that matters
# --------------------------------------------------------------------------- #
def test_crlf_and_lf_hash_identically(tmp_path: Path) -> None:
    lf = tmp_path / "lf.txt"
    crlf = tmp_path / "crlf.txt"
    lf.write_bytes(b"alpha\nbeta\ngamma\n")
    crlf.write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")

    assert content_sha256(lf) == content_sha256(crlf), (
        "the same content with different checkout line endings must hash identically, "
        "or a digest pinned on Windows fails on Linux for a file that never changed"
    )


def test_different_content_still_differs(tmp_path: Path) -> None:
    """Normalisation must not make the hash blind to real changes."""
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_bytes(b"alpha\nbeta\n")
    b.write_bytes(b"alpha\nbetaX\n")
    assert content_sha256(a) != content_sha256(b)


def test_a_trailing_newline_change_is_still_detected(tmp_path: Path) -> None:
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_bytes(b"alpha\n")
    b.write_bytes(b"alpha")
    assert content_sha256(a) != content_sha256(b)


def test_lone_cr_is_preserved(tmp_path: Path) -> None:
    """Only CRLF collapses. A bare CR is content, and nothing here uses it as an ending."""
    assert normalize(b"a\rb") == b"a\rb"
    assert normalize(b"a\r\nb") == b"a\nb"


def test_matches_plain_sha256_for_lf_content(tmp_path: Path) -> None:
    """On an LF file the digest is the ordinary one, so nothing surprising is introduced."""
    f = tmp_path / "lf.txt"
    payload = b"only\nlf\nhere\n"
    f.write_bytes(payload)
    assert content_sha256(f) == hashlib.sha256(payload).hexdigest()
    assert content_sha256_of(payload) == hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------- #
# The two implementations must not drift
# --------------------------------------------------------------------------- #
def test_acceptance_gate_has_its_own_copy(acceptance) -> None:
    assert hasattr(acceptance, "_content_sha256"), (
        "final_acceptance.py must carry its own normalising hash; it cannot import trgym"
    )


@pytest.mark.parametrize(
    "payload",
    [b"", b"\n", b"a\r\nb\r\n", b"a\nb\n", b"mixed\r\nand\nplain\r\n", b"no-trailing"],
)
def test_both_implementations_agree(acceptance, tmp_path: Path, payload: bytes) -> None:
    f = tmp_path / "sample.bin"
    f.write_bytes(payload)
    assert acceptance._content_sha256(f) == content_sha256(f), (
        "final_acceptance.py's inlined hash has drifted from trgym.provenance"
    )


def test_agreement_on_real_repository_files(acceptance) -> None:
    """Exercise the two implementations on the files the gates actually pin."""
    for rel in (
        "environments/transformer_repair/grading.py",
        "artifacts/final_metrics_summary.json",
        "trgym/repo/predicates.py",
    ):
        path = REPO_ROOT / rel
        if path.exists():
            assert acceptance._content_sha256(path) == content_sha256(path), rel
