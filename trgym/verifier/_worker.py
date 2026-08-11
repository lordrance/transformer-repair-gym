"""Subprocess entry point for running checks against untrusted candidate code.

Reads a JSON request on stdin, writes a JSON response on stdout. Keeping this in
a separate process is what gives us a hard timeout and stops a candidate that
monkey-patches torch from contaminating the parent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MARKER = "<<<TRGYM_RESULT>>>"


def main() -> int:
    request = json.loads(sys.stdin.read())
    sys.path.insert(0, request["repo_root"])

    from trgym.verifier.hidden import run_checks

    results = run_checks(Path(request["workspace"]), request["task_id"], request["checks"])
    payload = {"results": [{"name": n, "passed": ok, "detail": d} for n, ok, d in results]}
    sys.stdout.write(MARKER + json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
