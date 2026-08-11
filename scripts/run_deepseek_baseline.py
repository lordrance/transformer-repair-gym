r"""Phase 0.5 — DeepSeek V4 Pro baseline on the 5 repair tasks.

Answers one question: do these tasks form a sane environment for a real LLM?
Not "is DeepSeek good" and NOT "is the difficulty right for the policy we will
train" — see Task 11 of the phase brief. DeepSeek is a strong-model sanity
check; the trainable policy has to be measured separately.

The API key is read from DEEPSEEK_API_KEY and is never written to disk. The
JSONL log contains prompts, responses, patches, usage and rewards -- no
credentials.

Usage
-----
    $env:DEEPSEEK_API_KEY = [Environment]::GetEnvironmentVariable('DEEPSEEK_API_KEY','User')
    .\.venv\Scripts\python.exe scripts/run_deepseek_baseline.py --rollouts 4
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "environments" / "transformer_repair"))

BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("-r", "--rollouts", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=8000)
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "deepseek_baseline.jsonl"))
    ap.add_argument("--append", action="store_true", help="add rollouts to an existing log")
    args = ap.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("error: DEEPSEEK_API_KEY is not set", file=sys.stderr)
        return 2

    from openai import OpenAI

    from transformer_repair import SYSTEM_PROMPT, _grade_submission, build_dataset

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    dataset = build_dataset()

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    existing = 0
    if args.append and out_path.exists():
        existing = sum(1 for _ in out_path.open(encoding="utf-8"))
    mode = "a" if (args.append and out_path.exists()) else "w"

    jobs = [
        (row, rollout)
        for row in dataset
        for rollout in range(existing // len(dataset), existing // len(dataset) + args.rollouts)
    ]

    request_config = {
        "model": args.model,
        "base_url": BASE_URL,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "reasoning": "default (deepseek-v4-pro returns reasoning_content)",
    }
    print(f"config: {json.dumps(request_config)}")
    print(f"running {len(jobs)} trajectories at concurrency {args.concurrency}\n")

    def run_one(job):
        row, rollout_id = job
        task_id = row["info"]["task_id"]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["question"]},
        ]

        started = time.perf_counter()
        error = None
        try:
            resp = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - network/provider errors are data
            return {
                "task_id": task_id,
                "rollout_id": rollout_id,
                "error": f"{type(exc).__name__}: {exc}",
                "latency_s": round(time.perf_counter() - started, 2),
            }
        latency = time.perf_counter() - started

        msg = resp.choices[0].message
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or ""
        usage = resp.usage.model_dump() if resp.usage else {}

        graded = _grade_submission(content, {"task_id": task_id})

        record = {
            "task_id": task_id,
            "rollout_id": rollout_id,
            "model": resp.model,
            "request_config": request_config,
            "finish_reason": resp.choices[0].finish_reason,
            "prompt_system": SYSTEM_PROMPT,
            "prompt_user": row["question"],
            "response_content": content,
            "response_reasoning": reasoning,
            "usage": usage,
            "latency_s": round(latency, 2),
            "naive_reward": graded["naive"],
            "hardened_reward": graded["hardened"],
            "invalid_patch": graded["invalid_patch"],
            "invalid_reason": graded.get("invalid_reason", ""),
            "gates_fired": graded["gates"],
            "hidden_checks_failed": graded["hidden_failed"],
            "grade_metrics": graded.get("metrics", {}),
            "patched_source": graded.get("patched_source", ""),
            "error": error,
        }
        return record

    results = []
    with futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for record in pool.map(run_one, jobs):
            results.append(record)
            if record.get("error"):
                print(f"  {record['task_id']:38s} r{record['rollout_id']}  ERROR {record['error'][:60]}")
            else:
                flag = "INVALID" if record["invalid_patch"] else ""
                print(
                    f"  {record['task_id']:38s} r{record['rollout_id']}  "
                    f"naive={record['naive_reward']:.0f} hardened={record['hardened_reward']:.0f} "
                    f"{flag}"
                )

    with out_path.open(mode, encoding="utf-8") as fh:
        for record in results:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    ok = [r for r in results if not r.get("error")]
    total_prompt = sum(r["usage"].get("prompt_tokens", 0) for r in ok)
    total_completion = sum(r["usage"].get("completion_tokens", 0) for r in ok)
    total_reasoning = sum(
        (r["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
        for r in ok
    )
    print(
        f"\n{len(ok)}/{len(results)} trajectories completed"
        f"\nprompt tokens     {total_prompt:,}"
        f"\ncompletion tokens {total_completion:,}  (of which reasoning {total_reasoning:,})"
        f"\nwrote {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
