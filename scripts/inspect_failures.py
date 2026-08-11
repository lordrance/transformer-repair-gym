"""Dump the interesting rollouts: invalid patches and naive/hardened gaps.

Whether the 'invalid' rate is a model failure or a parser that is too strict is
the difference between a finding and a bug in our own code, so both get read by
hand rather than counted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

LOG = REPO_ROOT / "artifacts" / "deepseek_baseline.jsonl"


def records():
    with LOG.open(encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def main() -> None:
    rows = list(records())

    print("=" * 78)
    print("INVALID PATCHES")
    print("=" * 78)
    for r in rows:
        if not r.get("invalid_patch"):
            continue
        print(f"\n--- {r['task_id']} r{r['rollout_id']} ---")
        print(f"reason: {r['invalid_reason']}")
        print(f"finish_reason: {r['finish_reason']}")
        content = r["response_content"]
        print(f"content length: {len(content)} chars")
        print("--- response (first 1200 chars) ---")
        print(content[:1200])

    print("\n" + "=" * 78)
    print("NAIVE / HARDENED GAPS  (candidate natural exploits)")
    print("=" * 78)
    for r in rows:
        if r.get("invalid_patch") or r["naive_reward"] < 1.0 or r["hardened_reward"] >= 1.0:
            continue
        print(f"\n--- {r['task_id']} r{r['rollout_id']} ---")
        print(f"gates fired : {r['gates_fired']}")
        print(f"hidden fails: {r['hidden_checks_failed']}")
        print("--- diff submitted ---")
        print(r["response_content"][:2500])

    print("\n" + "=" * 78)
    print("HARDENED FAILURES THAT ALSO FAILED NAIVE (ordinary wrong answers)")
    print("=" * 78)
    for r in rows:
        if r.get("invalid_patch") or r["naive_reward"] >= 1.0 or r["hardened_reward"] >= 1.0:
            continue
        print(f"\n--- {r['task_id']} r{r['rollout_id']} ---  hidden: {r['hidden_checks_failed']}")
        print(r["response_content"][:1200])


if __name__ == "__main__":
    main()
