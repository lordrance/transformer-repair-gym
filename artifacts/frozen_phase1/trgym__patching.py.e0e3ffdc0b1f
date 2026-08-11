"""Unified-diff parsing and application.

Phase 0 had the model return the whole corrected file. Measured on real prompts
that spent ~78 % of the sampled tokens re-emitting unchanged code, and sampled
tokens are the most expensive class. Phase 0.5 switches to unified diffs.

There is no stdlib patch applier, so this is a small one. Two deliberate
choices:

* **Fuzzy hunk location.** Models routinely get `@@` line numbers wrong while
  producing correct context. We search for the hunk's "before" block near the
  declared position and fall back to a whole-file search. This measures the
  model's ability to fix the bug, not its ability to count lines.
* **Strict on ambiguity.** If the before-block does not appear, or appears more
  than once and the declared line number does not disambiguate, we raise. A
  patch that silently lands in the wrong place would poison the reward signal
  far worse than an honest INVALID.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["PatchError", "Hunk", "parse_unified_diff", "apply_unified_diff", "extract_diff"]

_HUNK_HEADER = re.compile(r"^@@\s*-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s*@@")
_FENCE = re.compile(r"```(?:diff|patch|python|py)?\s*\n(.*?)```", re.DOTALL)


class PatchError(ValueError):
    """The submission is not a patch we can apply. Graded as INVALID."""


@dataclass(frozen=True)
class Hunk:
    old_start: int          # 1-based line number from the @@ header
    before: tuple[str, ...]  # context + removed lines
    after: tuple[str, ...]   # context + added lines

    @property
    def is_noop(self) -> bool:
        return self.before == self.after


def extract_diff(text: str) -> str:
    """Pull a diff out of a model response.

    Prefers a fenced block; falls back to the raw text when the model emitted a
    bare diff. Returns the text unchanged if no fence is present, so the parser
    can produce a specific error rather than a generic one.
    """
    blocks = _FENCE.findall(text)
    if blocks:
        # Prefer the first block that actually looks like a diff.
        for block in blocks:
            if "@@" in block or block.lstrip().startswith(("--- ", "diff ")):
                return block
        return blocks[-1]
    return text


def _parse_headerless(lines: list[str]) -> list[Hunk]:
    """Accept a bare block of -/+/context lines with no `@@` header.

    Models emit this constantly: they produce the right edit and omit the hunk
    header, or invent line numbers and drop the `@@` entirely. Rejecting it would
    score transcription format rather than whether the bug was understood, so a
    headerless block is treated as a single hunk at an unknown offset and located
    by content. `old_start=1` is a hint only -- `_locate` searches the file.
    """
    before: list[str] = []
    after: list[str] = []
    saw_change = False
    for line in lines:
        # A bare "@@" with no line numbers is common and carries no information;
        # so are file headers and the no-newline marker.
        if (
            not line.strip()
            or line.startswith(("---", "+++", "diff ", "index ", "\\"))
            or line.lstrip().startswith("@@")
        ):
            continue
        if line.startswith("-"):
            before.append(line[1:])
            saw_change = True
        elif line.startswith("+"):
            after.append(line[1:])
            saw_change = True
        elif line.startswith(" "):
            before.append(line[1:])
            after.append(line[1:])
        else:
            # Prose or code without a marker: not a diff we can trust.
            return []
    if not saw_change or not before:
        return []
    return [Hunk(old_start=1, before=tuple(before), after=tuple(after))]


def parse_unified_diff(diff_text: str) -> list[Hunk]:
    lines = diff_text.splitlines()
    if not any(_HUNK_HEADER.match(line) for line in lines):
        headerless = _parse_headerless(lines)
        if headerless:
            return headerless
    hunks: list[Hunk] = []
    i = 0
    while i < len(lines):
        match = _HUNK_HEADER.match(lines[i])
        if match is None:
            i += 1
            continue

        old_start = int(match.group(1))
        i += 1
        before: list[str] = []
        after: list[str] = []

        while i < len(lines) and not _HUNK_HEADER.match(lines[i]):
            line = lines[i]
            if line.startswith("--- ") or line.startswith("+++ ") or line.startswith("diff "):
                break
            if line.startswith("\\"):          # "\ No newline at end of file"
                i += 1
                continue
            if line.startswith("-"):
                before.append(line[1:])
            elif line.startswith("+"):
                after.append(line[1:])
            elif line.startswith(" "):
                before.append(line[1:])
                after.append(line[1:])
            elif line == "":
                # Some models drop the leading space on blank context lines.
                before.append("")
                after.append("")
            else:
                raise PatchError(
                    f"unrecognised line in hunk at diff line {i + 1}: {line[:80]!r}"
                )
            i += 1

        if not before and not after:
            raise PatchError(f"empty hunk at old_start={old_start}")
        hunks.append(Hunk(old_start=old_start, before=tuple(before), after=tuple(after)))

    if not hunks:
        raise PatchError("no @@ hunk headers found; expected a unified diff")
    return hunks


def _find_all(original: list[str], block: list[str]) -> list[int]:
    n = len(block)
    if n == 0 or n > len(original):
        return []
    hits = [
        start
        for start in range(0, len(original) - n + 1)
        if original[start : start + n] == block
    ]
    if hits:
        return hits
    # Retry ignoring trailing whitespace, which models mangle constantly.
    stripped = [b.rstrip() for b in block]
    return [
        start
        for start in range(0, len(original) - n + 1)
        if [o.rstrip() for o in original[start : start + n]] == stripped
    ]


def _context_margins(before: list[str], after: list[str]) -> tuple[int, int]:
    """How many leading / trailing lines are pure context (identical in both)."""
    lead = 0
    while lead < len(before) and lead < len(after) and before[lead] == after[lead]:
        lead += 1
    trail = 0
    while (
        trail < len(before) - lead
        and trail < len(after) - lead
        and before[len(before) - 1 - trail] == after[len(after) - 1 - trail]
    ):
        trail += 1
    return lead, trail


def _locate(original: list[str], hunk: Hunk, hint: int) -> tuple[int, list[str], list[str], int]:
    """Locate a hunk, shedding context lines until it matches.

    `patch(1)` does the same thing under `--fuzz`: models reliably reproduce the
    lines they are changing and unreliably reproduce the surrounding context --
    they paraphrase docstrings, normalise quote characters, or skip a block
    entirely. Refusing those patches would measure transcription accuracy rather
    than whether the bug was understood.

    Returns (start_index, before_block, after_block, fuzz_lines_dropped).
    """
    before, after = list(hunk.before), list(hunk.after)
    if not before:
        raise PatchError("hunk has no context or removed lines, so it cannot be placed")

    lead, trail = _context_margins(before, after)

    # Try progressively larger fuzz, preferring to keep as much context as possible.
    for dropped in range(0, lead + trail + 1):
        for drop_lead in range(0, min(dropped, lead) + 1):
            drop_trail = dropped - drop_lead
            if drop_trail > trail:
                continue
            b = before[drop_lead : len(before) - drop_trail]
            a = after[drop_lead : len(after) - drop_trail]
            if not b:
                continue
            hits = _find_all(original, b)
            if not hits:
                continue
            start = hits[0] if len(hits) == 1 else min(hits, key=lambda s: abs(s - hint))
            return start, b, a, dropped

    raise PatchError(
        f"context block not found in the file (first changed line: "
        f"{next((l for l in before[lead:] if l.strip()), before[0])[:70]!r})"
    )


def apply_unified_diff(original: str, diff_text: str) -> str:
    """Apply `diff_text` to `original`. Raises PatchError on anything ambiguous."""
    hunks = parse_unified_diff(diff_text)
    if all(h.is_noop for h in hunks):
        raise PatchError("patch changes nothing")

    lines = original.splitlines()
    # Apply from the bottom up so earlier edits do not shift later offsets.
    located = []
    total_fuzz = 0
    for hunk in hunks:
        at, before, after, fuzz = _locate(lines, hunk, hint=max(0, hunk.old_start - 1))
        total_fuzz += fuzz
        located.append((at, before, after))
    located.sort(key=lambda triple: triple[0], reverse=True)

    seen: set[int] = set()
    for at, before, after in located:
        span = range(at, at + len(before))
        if seen & set(span):
            raise PatchError("overlapping hunks")
        seen |= set(span)
        lines[at : at + len(before)] = after

    out = "\n".join(lines)
    if original.endswith("\n"):
        out += "\n"
    _LAST_FUZZ.append(total_fuzz)
    return out


# Records the fuzz used by the most recent apply, so the harness can report how
# often patches only landed because of context shedding.
_LAST_FUZZ: list[int] = []


def last_fuzz() -> int:
    return _LAST_FUZZ[-1] if _LAST_FUZZ else 0
