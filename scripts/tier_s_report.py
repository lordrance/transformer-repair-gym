"""G4: turn the Tier S trajectories into the audit CSV and the measured summary.

Emits `TIER_S_REAL_MODEL_AUDIT.csv` -- one row per episode, every column derived from the
trajectory record -- and prints the aggregates that `LOCALIZATION_SCALE_REPORT.md` cites,
so the report quotes measurements rather than recollections.

The column that carries G4 is `fraction_repo_inspected`. It must be < 1.0 for every
episode: an agent that read all 48 files did not localize anything, it enumerated the
repo, and the tier would be measuring patience rather than diagnosis.

Run: python scripts/tier_s_report.py
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "artifacts" / "tier_s_primary.jsonl"
CSV_OUT = ROOT / "TIER_S_REAL_MODEL_AUDIT.csv"

COLUMNS = [
    "task_id", "episode_id", "model", "n_turns", "submitted", "end_reason",
    "n_files_in_repo", "n_files_inspected", "fraction_repo_inspected",
    "located_relevant_file", "edited_a_relevant_file",
    "files_expected", "files_edited_by_model", "relevant_files_inspected",
    "naive_reward", "hardened_reward", "hidden_failed",
    "prompt_tokens", "completion_tokens", "api_errors",
]


def main() -> int:
    if not SRC.exists():
        raise SystemExit(f"missing {SRC.relative_to(ROOT).as_posix()}; run the Tier S eval first")
    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]

    out = []
    for r in rows:
        ep = r.get("episode") or {}
        usage = r.get("usage") or {}
        out.append({
            "task_id": r.get("task_id"),
            "episode_id": r.get("episode_id"),
            "model": r.get("model"),
            "n_turns": ep.get("n_turns"),
            "submitted": ep.get("submitted"),
            "end_reason": ep.get("end_reason"),
            "n_files_in_repo": r.get("n_files_in_repo"),
            "n_files_inspected": r.get("n_files_inspected"),
            "fraction_repo_inspected": r.get("fraction_repo_inspected"),
            "located_relevant_file": r.get("located_relevant_file"),
            "edited_a_relevant_file": r.get("edited_a_relevant_file"),
            "files_expected": ";".join(r.get("files_expected") or []),
            "files_edited_by_model": ";".join(r.get("files_edited_by_model") or []),
            "relevant_files_inspected": ";".join(r.get("relevant_files_inspected") or []),
            "naive_reward": r.get("naive_reward"),
            "hardened_reward": r.get("hardened_reward"),
            "hidden_failed": ";".join(r.get("hidden_failed") or []),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "api_errors": r.get("api_errors"),
        })

    with CSV_OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(out)

    fr = [float(r["fraction_repo_inspected"]) for r in out]
    hardened = [float(r["hardened_reward"] or 0) for r in out]
    naive = [float(r["naive_reward"] or 0) for r in out]
    located = [bool(r["located_relevant_file"]) for r in out]
    edited = [bool(r["edited_a_relevant_file"]) for r in out]
    turns = [int(r["n_turns"] or 0) for r in out]

    # The budget makes exhaustive reading impossible by construction, not by luck: the
    # adapter permits one tool call per turn, so an agent cannot issue more read_file
    # calls than it has turns. Stating the bound as a computed number means a future
    # change to either the turn budget or the repo size cannot quietly invalidate G4's
    # "cannot exhaustively read the repo" premise.
    max_turns = max(int(r["n_turns"] or 0) for r in out)
    n_files = int(out[0]["n_files_in_repo"] or 0)
    bound = round(max_turns / n_files, 4) if n_files else None

    print(f"episodes                      {len(out)}")
    print(f"files in repo                 {n_files}")
    print(f"max turns observed            {max_turns}")
    print(f"structural ceiling on frac    {bound}  (<=1 tool call per turn)")
    print(f"exhaustive read possible?     {n_files <= max_turns}")
    print(f"fraction inspected  mean/max  {statistics.mean(fr):.3f} / {max(fr):.3f}")
    print(f"every episode < 1.0           {all(f < 1.0 for f in fr)}")
    print(f"located relevant file         {sum(located)}/{len(out)}")
    print(f"edited a relevant file        {sum(edited)}/{len(out)}")
    print(f"hardened pass (full fix)      {sum(1 for h in hardened if h >= 1.0)}/{len(out)}")
    print(f"naive pass (visible only)     {sum(1 for n in naive if n >= 1.0)}/{len(out)}")
    print(f"naive FP (visible ok, hidden not) "
          f"{sum(1 for n, h in zip(naive, hardened) if n >= 1.0 and h < 1.0)}/{len(out)}")
    print(f"mean turns                    {statistics.mean(turns):.1f}")
    print("\nper task:")
    for task in sorted({r['task_id'] for r in out}):
        sub = [r for r in out if r["task_id"] == task]
        sf = [float(r["fraction_repo_inspected"]) for r in sub]
        print(f"  {task:<34} n={len(sub)}  "
              f"frac={statistics.mean(sf):.3f}  "
              f"located={sum(1 for r in sub if r['located_relevant_file'])}/{len(sub)}  "
              f"fixed={sum(1 for r in sub if float(r['hardened_reward'] or 0) >= 1)}/{len(sub)}")
    print(f"\nwrote {CSV_OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
