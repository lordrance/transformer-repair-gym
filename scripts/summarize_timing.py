"""Summarize measured grading wall-times from the Phase-0 audits."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rows = json.loads((ROOT / "artifacts" / "task_audit.json").read_text(encoding="utf-8"))
    times = [r["gold_hidden_wall_time_s"] for r in rows]
    for r in rows:
        print(f"{r['task_id']:38s} hidden_suite={r['gold_hidden_wall_time_s']:6.2f}s")
    print(
        f"\nmean={statistics.mean(times):.2f}s  "
        f"median={statistics.median(times):.2f}s  "
        f"max={max(times):.2f}s"
    )
    print(
        "\nNOTE: this is one full hidden suite per rollout, single process, "
        "no warm interpreter reuse. It is an upper bound on steady-state cost."
    )


if __name__ == "__main__":
    main()
