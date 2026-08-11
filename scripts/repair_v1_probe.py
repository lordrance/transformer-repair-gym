"""Recover the JSON payload from artifacts/raw/v1_probe.json.

The probe was run as `docker run ... > v1_probe.json` from PowerShell, which folds
the container's stderr (a pip root-user warning) into the same stream. The file is
therefore a valid probe result with a prose prefix, not a corrupt result -- but it
does not parse, and G0 never noticed because G0 checks existence and hash, not
parseability. See PROTOCOL_CHANGELOG R8.

The original byte stream is preserved as v1_probe.raw.txt; the parsed payload is
written alongside it as v1_probe.json so downstream code can load it.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "artifacts" / "raw" / "v1_probe.json"
RAW = ROOT / "artifacts" / "raw" / "v1_probe.raw.txt"


def _balanced_span(text: str, start: int) -> str | None:
    """The brace-balanced substring beginning at `start`, or None if unterminated."""
    depth, in_str, esc = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract(text: str) -> dict:
    """Return the first top-level JSON object in `text` that actually parses.

    The prose prefix here is PowerShell's NativeCommandError wrapper, which echoes
    the failing command -- and that echo contains braces. So the first '{' is not
    the payload; every candidate has to be tried.
    """
    pos = text.find("{")
    while pos >= 0:
        span = _balanced_span(text, pos)
        if span is not None:
            try:
                return json.loads(span)
            except json.JSONDecodeError:
                pass
        pos = text.find("{", pos + 1)
    raise SystemExit("no parseable top-level JSON object found in the probe output")


def main() -> None:
    text = SRC.read_text(encoding="utf-8", errors="replace")
    try:
        json.loads(text)
    except json.JSONDecodeError:
        pass
    else:
        print("v1_probe.json already parses; nothing to do")
        return

    payload = extract(text)

    # Preserve the original bytes before replacing the file (G0 discipline: the
    # polluted stream is the artifact of record for how the probe was actually run).
    if not RAW.exists():
        RAW.write_text(text, encoding="utf-8")
        print(f"preserved original stream -> {RAW.relative_to(ROOT).as_posix()}")

    SRC.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"recovered JSON payload -> {SRC.relative_to(ROOT).as_posix()}")
    print(f"top-level keys: {sorted(payload)}")


if __name__ == "__main__":
    main()
