"""Summarize a baseline JSONL: pass rates, protocol failures, tokens, cost inputs.

Separates "the model got it wrong" from "the model ran out of tokens", because
conflating the two would turn a configuration mistake into a false claim about
difficulty.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def summarize(path: Path) -> dict:
    rows = load(path)
    per_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per_task[r["task_id"]].append(r)

    finish = defaultdict(int)
    for r in rows:
        finish[r.get("finish_reason", "?")] += 1

    truncated = [r for r in rows if r.get("finish_reason") == "length"]
    invalid = [r for r in rows if r.get("invalid_patch")]
    invalid_and_truncated = [r for r in invalid if r.get("finish_reason") == "length"]

    prompt_tokens = [r["usage"].get("prompt_tokens", 0) for r in rows if r.get("usage")]
    completion_tokens = [r["usage"].get("completion_tokens", 0) for r in rows if r.get("usage")]
    reasoning_tokens = [
        (r["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
        for r in rows
        if r.get("usage")
    ]
    cache_hit = [r["usage"].get("prompt_cache_hit_tokens", 0) or 0 for r in rows if r.get("usage")]

    print(f"\n{'=' * 72}\n{path.name}  ({rows[0].get('model', '?')})\n{'=' * 72}")
    print(f"{'task':38s} {'naive':>6s} {'hard':>6s} {'invalid':>8s}  rewards")
    for task_id, runs in sorted(per_task.items()):
        naive = [r["naive_reward"] for r in runs]
        hard = [r["hardened_reward"] for r in runs]
        n_inv = sum(1 for r in runs if r.get("invalid_patch"))
        pattern = ",".join(f"{int(h)}" for h in hard)
        print(
            f"{task_id:38s} {statistics.mean(naive):6.2f} {statistics.mean(hard):6.2f} "
            f"{n_inv:8d}  [{pattern}]"
        )

    print(f"\nfinish_reason: {dict(finish)}")
    print(
        f"invalid patches      : {len(invalid)}/{len(rows)}"
        f"  (of which truncated: {len(invalid_and_truncated)})"
    )
    print(f"truncated responses  : {len(truncated)}/{len(rows)}")
    print(
        f"prompt tokens        : total {sum(prompt_tokens):,}  mean {statistics.mean(prompt_tokens):,.0f}"
    )
    print(
        f"completion tokens    : total {sum(completion_tokens):,}  mean "
        f"{statistics.mean(completion_tokens):,.0f}"
    )
    print(
        f"  of which reasoning : total {sum(reasoning_tokens):,}  mean "
        f"{statistics.mean(reasoning_tokens):,.0f}  "
        f"({sum(reasoning_tokens) / max(1, sum(completion_tokens)):.0%} of completion)"
    )
    visible = [c - r for c, r in zip(completion_tokens, reasoning_tokens)]
    print(
        f"  visible answer     : total {sum(visible):,}  mean {statistics.mean(visible):,.0f}"
    )
    print(f"prompt cache hits    : {sum(cache_hit):,}")

    return {
        "file": path.name,
        "model": rows[0].get("model"),
        "n": len(rows),
        "invalid": len(invalid),
        "invalid_truncated": len(invalid_and_truncated),
        "prompt_tokens": sum(prompt_tokens),
        "completion_tokens": sum(completion_tokens),
        "reasoning_tokens": sum(reasoning_tokens),
        "visible_answer_tokens": sum(visible),
    }


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] or [
        REPO_ROOT / "artifacts" / "deepseek_baseline.jsonl",
        REPO_ROOT / "artifacts" / "deepseek_flash_baseline.jsonl",
    ]
    out = [summarize(p) for p in paths if p.exists()]
    (REPO_ROOT / "artifacts" / "run_summaries.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
