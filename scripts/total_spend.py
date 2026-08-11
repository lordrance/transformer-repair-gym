"""Total API spend this session, from logged token counts only."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PRIOR = [("phase 0.5 tier E (retained)", 60, 0.1240), ("smoke + probes", 2, 0.0100)]


def main() -> None:
    rows = json.loads((ROOT / "artifacts" / "analysis_summary.json").read_text(encoding="utf-8"))
    total = 0.0
    n_total = 0
    print(f"{'run':30s} {'n':>4s} {'cost':>9s}")
    for r in rows:
        print(f"{r['label']:30s} {r['n_trajectories']:4d} ${r['cost_usd']:8.4f}")
        total += r["cost_usd"]
        n_total += r["n_trajectories"]
    for label, n, cost in PRIOR:
        print(f"{label:30s} {n:4d} ${cost:8.4f}")
        total += cost
        n_total += n
    print(f"{'-' * 45}\n{'TOTAL':30s} {n_total:4d} ${total:8.4f}")

    (ROOT / "artifacts" / "total_spend.json").write_text(
        json.dumps({"total_usd": round(total, 4), "trajectories": n_total}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
