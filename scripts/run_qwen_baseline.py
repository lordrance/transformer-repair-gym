r"""Phase 0.6 -- Qwen3.5-4B target-policy calibration (sampling only).

Answers one question: is the frozen 5-task set at a difficulty where the policy
we would actually train produces reward *variance*? A task at 4/4 or 0/4 gives
GRPO exactly zero gradient, so this is a go/no-go measurement, not a scoreboard.

Hard constraints, enforced here and not just documented:

  * SAMPLING ONLY. No TrainingClient, no LoRA, no optim step, no checkpoint.
    The only Tinker object created is a base-model SamplingClient.
  * The renderer and thinking mode are pinned in TARGET_POLICY_CONFIG.md and
    must match whatever a future RL run uses. Do not "fix" a low score by
    changing the template.
  * The task prompts, diff protocol and both verifiers are byte-identical to the
    Phase 0.5 DeepSeek run, so any difference is the model, not the task.

The key is read from TINKER_API_KEY and never written anywhere.

Usage
-----
    $env:TINKER_API_KEY = "<your key>"     # set it in your shell, not in code
    .\.venv\Scripts\python.exe scripts/run_qwen_baseline.py --rollouts 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "environments" / "transformer_repair"))

# Pinned in TARGET_POLICY_CONFIG.md. Changing these invalidates the comparison
# with the DeepSeek baseline and with any future RL run.
MODEL_ID = "Qwen/Qwen3.5-4B"
RENDERER_NAME = "qwen3_5"          # thinking mode ON
PRICE_PREFILL_PER_MTOK = 0.33
PRICE_CACHED_PREFILL_PER_MTOK = 0.066
PRICE_SAMPLE_PER_MTOK = 1.005


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--renderer", default=RENDERER_NAME)
    ap.add_argument("-r", "--rollouts", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    # Generous on purpose: the Phase 0.5 flash run wasted 20 trajectories by
    # truncating mid-reasoning, which looked like a capability failure and was
    # a config failure. Actual usage is reported below.
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "qwen35_4b_baseline.jsonl"))
    args = ap.parse_args()

    if not os.environ.get("TINKER_API_KEY"):
        print(
            "error: TINKER_API_KEY is not set.\n"
            "Set it in your shell (not in any file, not in chat):\n"
            '    $env:TINKER_API_KEY = "<key>"',
            file=sys.stderr,
        )
        return 2

    import tinker
    from tinker_cookbook import renderers

    from transformer_repair import SYSTEM_PROMPT, _grade_submission, build_dataset

    service = tinker.ServiceClient()
    sampling_client = service.create_sampling_client(base_model=args.model)
    tokenizer = sampling_client.get_tokenizer()
    renderer = renderers.get_renderer(args.renderer, tokenizer, model_name=args.model)
    stop_sequences = renderer.get_stop_sequences()

    sampling_config = {
        "model": args.model,
        "renderer": args.renderer,
        "thinking_mode": args.renderer != "qwen3_5_disable_thinking",
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": 1.0,
        "top_k": -1,
        "seed": args.seed,
        "stop": [int(s) if isinstance(s, int) else s for s in stop_sequences],
        "num_samples_per_prompt": args.rollouts,
    }
    print(f"config: {json.dumps(sampling_config)}")

    dataset = build_dataset()
    records = []

    for row in dataset:
        task_id = row["info"]["task_id"]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["question"]},
        ]
        prompt = renderer.build_generation_prompt(messages)
        prompt_tokens = prompt.length

        params = tinker.SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            stop=stop_sequences,
            seed=args.seed,
        )

        started = time.perf_counter()
        response = sampling_client.sample(
            prompt=prompt, num_samples=args.rollouts, sampling_params=params
        ).result()
        latency = time.perf_counter() - started
        cache_hit = int(getattr(response, "prompt_cache_hit_tokens", 0) or 0)

        for rollout_id, seq in enumerate(response.sequences):
            tokens = list(seq.tokens)
            stop_reason = str(getattr(seq, "stop_reason", "?"))
            try:
                message, _termination = renderer.parse_response(tokens)
                content = message.get("content") if isinstance(message, dict) else message.content
                content = content if isinstance(content, str) else str(content)
            except Exception as exc:  # noqa: BLE001 - a malformed sample is data
                content = tokenizer.decode(tokens)
                stop_reason = f"{stop_reason} (parse_response failed: {type(exc).__name__})"

            graded = _grade_submission(content, {"task_id": task_id})

            records.append(
                {
                    "task_id": task_id,
                    "rollout_id": rollout_id,
                    "model": args.model,
                    "renderer": args.renderer,
                    "sampling_config": sampling_config,
                    "prompt_tokens": prompt_tokens,
                    "prompt_cache_hit_tokens_for_batch": cache_hit,
                    "sampled_tokens": len(tokens),
                    "stop_reason": stop_reason,
                    "truncated": len(tokens) >= args.max_tokens,
                    "prompt_system": SYSTEM_PROMPT,
                    "prompt_user": row["question"],
                    "response_content": content,
                    "batch_latency_s": round(latency, 2),
                    "naive_reward": graded["naive"],
                    "hardened_reward": graded["hardened"],
                    "invalid_patch": graded["invalid_patch"],
                    "invalid_reason": graded.get("invalid_reason", ""),
                    "gates_fired": graded["gates"],
                    "hidden_checks_failed": graded["hidden_failed"],
                    "grade_metrics": graded.get("metrics", {}),
                    "patched_source": graded.get("patched_source", ""),
                    "error": None,
                }
            )
            flag = "INVALID" if graded["invalid_patch"] else ""
            trunc = "TRUNCATED" if len(tokens) >= args.max_tokens else ""
            print(
                f"  {task_id:38s} r{rollout_id}  "
                f"naive={graded['naive']:.0f} hardened={graded['hardened']:.0f} "
                f"sampled={len(tokens):6d} {flag} {trunc}"
            )

    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Cost: each prompt is prefilled once per sample() call, then shared across
    # the num_samples completions, so bill prefill per prompt, not per rollout.
    n_prompts = len(dataset)
    prefill_total = sum(r["prompt_tokens"] for r in records) // max(1, args.rollouts)
    cache_hit_total = sum({(r["task_id"]): r["prompt_cache_hit_tokens_for_batch"] for r in records}.values())
    prefill_billed = max(0, prefill_total - cache_hit_total)
    sampled_total = sum(r["sampled_tokens"] for r in records)
    cost = (
        prefill_billed / 1e6 * PRICE_PREFILL_PER_MTOK
        + cache_hit_total / 1e6 * PRICE_CACHED_PREFILL_PER_MTOK
        + sampled_total / 1e6 * PRICE_SAMPLE_PER_MTOK
    )
    truncated = sum(1 for r in records if r["truncated"])

    print(
        f"\n{len(records)} trajectories over {n_prompts} prompts"
        f"\ntruncated            {truncated}/{len(records)}"
        f"\nprefill tokens       {prefill_total:,} (cache hits {cache_hit_total:,})"
        f"\nsampled tokens       {sampled_total:,}  (mean {sampled_total / max(1, len(records)):,.0f})"
        f"\nestimated cost       ${cost:.4f}  (${cost / max(1, len(records)):.5f}/rollout)"
        f"\n  -> 40 rollouts     ${cost / max(1, len(records)) * 40:.4f}"
        f"\n  -> 1,000 rollouts  ${cost / max(1, len(records)) * 1000:.2f}"
        f"\nwrote {out_path}"
    )
    if truncated:
        print(
            "\nWARNING: some samples hit max_tokens. Those are config failures, "
            "not capability failures -- re-run with a larger budget before "
            "drawing any difficulty conclusion."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
