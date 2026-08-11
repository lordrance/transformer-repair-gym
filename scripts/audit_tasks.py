"""Phase-0 task audit: prove every task discriminates gold from buggy.

For each task we build two workspaces -- the mutated one and the gold one -- and
grade both. A task is only usable if:

    gold   -> naive 1.0, hardened 1.0
    buggy  -> hardened 0.0   (naive may pass; that IS the exploit gap)

Run:  python scripts/audit_tasks.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from trgym.tasks.build import build_workspace  # noqa: E402
from trgym.tasks.registry import TASKS  # noqa: E402
from trgym.verifier.reward import grade  # noqa: E402


def main() -> int:
    rows = []
    ok = True

    with tempfile.TemporaryDirectory(prefix="trgym_audit_") as tmp:
        root = Path(tmp)
        for spec in TASKS:
            gold_ws = build_workspace(spec, root / f"{spec.task_id}__gold", gold=True)
            bug_ws = build_workspace(spec, root / f"{spec.task_id}__buggy", gold=False)

            gold = grade(spec, gold_ws)
            buggy = grade(spec, bug_ws)

            gold_ok = gold.naive_reward == 1.0 and gold.hardened_reward == 1.0
            buggy_ok = buggy.hardened_reward == 0.0
            task_ok = gold_ok and buggy_ok
            ok = ok and task_ok

            failed_hidden = [r.name for r in buggy.hidden.results if not r.passed]
            rows.append(
                {
                    "task_id": spec.task_id,
                    "family_id": spec.family_id,
                    "provenance": spec.provenance,
                    "gold_naive": gold.naive_reward,
                    "gold_hardened": gold.hardened_reward,
                    "buggy_naive": buggy.naive_reward,
                    "buggy_hardened": buggy.hardened_reward,
                    "buggy_caught_by": failed_hidden,
                    "buggy_is_naive_exploit_gap": buggy.exploited,
                    "gold_hidden_wall_time_s": gold.metrics["hidden_wall_time_s"],
                    "verdict": "OK" if task_ok else "BROKEN",
                }
            )

            status = "OK  " if task_ok else "FAIL"
            print(
                f"{status} {spec.task_id:36s} "
                f"gold(n={gold.naive_reward:.0f},h={gold.hardened_reward:.0f}) "
                f"buggy(n={buggy.naive_reward:.0f},h={buggy.hardened_reward:.0f}) "
                f"caught_by={failed_hidden}"
            )
            if not gold_ok:
                print(f"     gold failures: {gold.summary()}")
                for v in gold.violations:
                    print(f"       gate: {v}")
                for r in gold.hidden.results:
                    if not r.passed:
                        print(f"       hidden {r.name}: {r.detail}")
                if gold.hidden.stderr:
                    print(f"       stderr: {gold.hidden.stderr[-600:]}")

    out = REPO_ROOT / "artifacts" / "task_audit.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
