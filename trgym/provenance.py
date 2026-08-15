"""Content hashing that means the same thing on every platform.

Several gates pin a SHA-256 and later re-check it: `final_acceptance.py` G8 compares the
digest printed in the research report against the summary on disk, and G1 rejects
`v1_runtime_evidence.json` whose `grading_sha256` no longer matches `grading.py`. Those
checks exist to catch stale evidence, and they are worth keeping strict.

They were hashing raw bytes, which made them **platform-dependent**. Git stores text with
LF; a Windows checkout materialises CRLF. So the same file, byte-identical in the
repository, hashes one way on the machine that pinned the digest and another way on a
Linux CI runner. The gate then reports "stale evidence must not close the gate" for
evidence that is not stale at all -- a false alarm that trains the reader to ignore a real
signal.

Normalising CRLF to LF before hashing fixes it without touching how Git stores anything,
and without renormalising the working tree -- which would be the dangerous option here,
since G0 verifies frozen artifacts by content and a mass line-ending rewrite would move
every one of those hashes at once.

This is deliberately *not* a general-purpose checksum: it is a content identity for text
files under version control. Binary artifacts should be hashed with `hashlib` directly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def normalize(data: bytes) -> bytes:
    """CRLF -> LF. Lone CR is left alone; no text file in this project uses it."""
    return data.replace(b"\r\n", b"\n")


def content_sha256(path: str | Path) -> str:
    """SHA-256 of a text file's content, independent of checkout line endings."""
    return hashlib.sha256(normalize(Path(path).read_bytes())).hexdigest()


def content_sha256_of(data: bytes) -> str:
    """Same, for content already in memory."""
    return hashlib.sha256(normalize(data)).hexdigest()
