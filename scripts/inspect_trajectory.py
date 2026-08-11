"""Print a trajectory step by step, so a run can be read rather than trusted.

Usage:  python scripts/inspect_trajectory.py artifacts/tier_m_primary.jsonl [index|task_id]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def show(rec: dict, verbose: bool = True) -> None:
    ep = rec["episode"]
    print("=" * 78)
    print(
        f"{rec['task_id']}  e{rec['episode_id']}  model={rec.get('model')}\n"
        f"end={ep['end_reason']}  submitted={ep['submitted']}  turns={ep['n_turns']}\n"
        f"naive={rec['naive_reward']:.0f} hardened={rec['hardened_reward']:.0f}  "
        f"edited={rec.get('files_edited_by_model')}  expected={rec['files_expected']}\n"
        f"hidden_failed={rec['hidden_failed']}\n"
        f"usage={rec.get('usage')}  api_errors={rec.get('api_errors')}"
    )
    print("=" * 78)
    if not verbose:
        return
    for s in ep["steps"]:
        args = {
            k: (v[:160].replace("\n", " ⏎ ") if isinstance(v, str) else v)
            for k, v in s["args"].items()
        }
        print(f"\nT{s['turn']:2d}  {s['tool']:12s} ok={s['ok']}  args={args}")
        out = s["output"][:400].replace("\n", "\n      ")
        print(f"      -> {out}")
    if ep.get("summary"):
        print(f"\nSUMMARY: {ep['summary'][:600]}")


def main() -> None:
    path = Path(sys.argv[1])
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    sel = sys.argv[2] if len(sys.argv) > 2 else None

    if sel is None:
        for i, r in enumerate(rows):
            ep = r["episode"]
            print(
                f"[{i:2d}] {r['task_id']:26s} e{r['episode_id']} "
                f"naive={r['naive_reward']:.0f} hardened={r['hardened_reward']:.0f} "
                f"turns={ep['n_turns']:2d} end={ep['end_reason'][:34]:34s} "
                f"edited={r.get('files_edited_by_model')}"
            )
        return

    if sel.isdigit():
        show(rows[int(sel)])
    else:
        for r in rows:
            if r["task_id"] == sel:
                show(r)


if __name__ == "__main__":
    main()
