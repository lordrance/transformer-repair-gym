"""G9 stage C: pull the exact samples that must be read by hand, and dump them.

The contract asks for 3 successful and 3 failed trajectories, 3 fuzz exploits, 3 gold
trees and 3 no-op trees, read end to end. This script does the *selection* -- deterministic
and stated, so nobody can be accused of picking flattering examples -- and prints enough of
each for the reading to happen against real content rather than against a summary.

Selection rule: sort by (task_id, episode_id) and take the first N of each class. No
sampling, no shuffling, no seed to argue about.

Run: python scripts/post_success_spotcheck.py > /tmp/spotcheck_dump.txt
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TRAJECTORY_SOURCES = [
    "artifacts/tier_s_primary.jsonl",
    "artifacts/tier_h_primary.jsonl",
    "artifacts/tier_m_primary.jsonl",
]


def rows_of(rel: str) -> list[dict]:
    path = ROOT / rel
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            row["_source"] = rel
            out.append(row)
    return out


def describe(row: dict) -> str:
    ep = row.get("episode") or {}
    steps = ep.get("steps") or []
    tools = [s.get("tool") for s in steps]
    lines = [
        f"  source            {row['_source']}",
        f"  task/episode      {row.get('task_id')} #{row.get('episode_id')}",
        f"  naive/hardened    {row.get('naive_reward')} / {row.get('hardened_reward')}",
        f"  turns             {ep.get('n_turns')}  end={ep.get('end_reason')}  "
        f"submitted={ep.get('submitted')}",
        f"  tool sequence     {' -> '.join(str(t) for t in tools[:14])}"
        + (" ..." if len(tools) > 14 else ""),
        f"  files expected    {row.get('files_expected')}",
        f"  files edited      {row.get('files_edited_by_model')}",
        f"  hidden failed     {row.get('hidden_failed')}",
    ]
    if "fraction_repo_inspected" in row:
        lines += [
            f"  repo inspected    {row.get('n_files_inspected')}/{row.get('n_files_in_repo')}"
            f" = {row.get('fraction_repo_inspected')}",
            f"  located relevant  {row.get('located_relevant_file')}",
        ]
    if ep.get("summary"):
        lines.append(f"  model summary     {str(ep['summary'])[:300]}")
    return "\n".join(lines)


def main() -> int:
    rows: list[dict] = []
    for rel in TRAJECTORY_SOURCES:
        rows += rows_of(rel)
    rows.sort(key=lambda r: (r["_source"], str(r.get("task_id")), int(r.get("episode_id") or 0)))

    passed = [r for r in rows if float(r.get("hardened_reward") or 0) >= 1.0][:3]
    failed = [r for r in rows if float(r.get("hardened_reward") or 0) < 1.0][:3]

    print("=" * 78)
    print("SUCCESSFUL TRAJECTORIES (hardened_reward == 1.0), first 3 by (source, task, id)")
    print("=" * 78)
    for r in passed:
        print(describe(r), "\n")
    if len(passed) < 3:
        print(f"  ONLY {len(passed)} SUCCESSFUL EPISODES EXIST across all sources.\n")

    print("=" * 78)
    print("FAILED TRAJECTORIES (hardened_reward < 1.0), first 3")
    print("=" * 78)
    for r in failed:
        print(describe(r), "\n")

    print("=" * 78)
    print("FUZZ EXPLOITS, first 3")
    print("=" * 78)
    fuzz = json.loads((ROOT / "artifacts" / "verifier_fuzz_audit.json").read_text(encoding="utf-8"))
    probes = fuzz if isinstance(fuzz, list) else (fuzz.get("probes") or fuzz.get("results") or [])
    for probe in probes[:3]:
        print(json.dumps(probe, indent=2)[:1100], "\n")
    print(f"  total probes: {len(probes)}")

    print("=" * 78)
    print("GOLD / NO-OP SEPARATION, 3 tasks")
    print("=" * 78)
    spec = json.loads((ROOT / "artifacts" / "tier_s_spec.json").read_text(encoding="utf-8"))
    for task in (spec.get("tasks") or [])[:3]:
        print(f"  {task['task_id']}")
        print(f"    files                 {task.get('n_files')}")
        print(f"    gold passes           {task.get('gold_passes')}")
        print(f"    no-op fails           {task.get('noop_fails')}")
        print(f"    buggy failing checks  {task.get('buggy_failing_checks')}")
        print(f"    only relevant differ  {task.get('only_relevant_files_differ')}")
        for rel, d in (task.get("mutation_applied") or {}).items():
            print(f"    {rel}")
            print(f"      gold  {d['gold_sha256'][:16]}")
            print(f"      buggy {d['buggy_sha256'][:16]}  changed={d['changed']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
