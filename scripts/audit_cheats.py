"""Phase-0 exploit audit.

For every hand-written cheat patch, assert the property that justifies the whole
project:

    naive grader    -> rewards it
    hardened grader -> rejects it

A cheat that the naive grader also rejects is not evidence of an exploit gap and
is reported as WEAK rather than quietly counted as a success.

Run:  python scripts/audit_cheats.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from trgym.cheats.catalog import CHEATS  # noqa: E402
from trgym.tasks.build import build_workspace  # noqa: E402
from trgym.tasks.registry import get_task  # noqa: E402
from trgym.verifier.reward import grade  # noqa: E402


def main() -> int:
    rows = []
    ok = True

    with tempfile.TemporaryDirectory(prefix="trgym_cheats_") as tmp:
        root = Path(tmp)
        for cheat in CHEATS:
            spec = get_task(cheat.task_id)
            ws = build_workspace(spec, root / cheat.cheat_id, gold=False)
            cheat.apply(ws, spec.target_file)

            result = grade(spec, ws)
            fired = sorted({v.gate for v in result.violations})
            gate_ok = cheat.expected_gate is None or cheat.expected_gate in fired

            naive_rewards_it = result.naive_reward >= 1.0
            hardened_rejects_it = result.hardened_reward < 1.0
            passed = naive_rewards_it and hardened_rejects_it and gate_ok
            ok = ok and hardened_rejects_it and gate_ok

            if not naive_rewards_it:
                verdict = "WEAK"  # hardened catches it, but so does naive
            elif not hardened_rejects_it:
                verdict = "ESCAPED"
            elif not gate_ok:
                verdict = "WRONG-GATE"
            else:
                verdict = "CAUGHT"

            failed_hidden = [r.name for r in result.hidden.results if not r.passed]
            rows.append(
                {
                    "cheat_id": cheat.cheat_id,
                    "task_id": cheat.task_id,
                    "strategy": cheat.strategy,
                    "naive_reward": result.naive_reward,
                    "hardened_reward": result.hardened_reward,
                    "expected_gate": cheat.expected_gate,
                    "gates_fired": fired,
                    "hidden_checks_failed": failed_hidden,
                    "verdict": verdict,
                }
            )
            print(
                f"{verdict:10s} {cheat.cheat_id:26s} "
                f"naive={result.naive_reward:.0f} hardened={result.hardened_reward:.0f} "
                f"gates={fired or '-'}"
            )
            if verdict in {"ESCAPED", "WRONG-GATE"}:
                print(f"           expected gate {cheat.expected_gate!r}")
                print(f"           hidden failures: {failed_hidden}")

    out = REPO_ROOT / "artifacts" / "cheat_audit.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    caught = sum(1 for r in rows if r["verdict"] == "CAUGHT")
    print(f"\n{caught}/{len(rows)} cheats reproduce a naive->hardened gap and are caught")
    print(f"wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
