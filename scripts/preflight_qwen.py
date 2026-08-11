r"""Pre-flight for the Qwen3.5-4B calibration -- runs without a Tinker key.

Renders the five task prompts through the *pinned* renderer using the HF
tokenizer, so the prompt token counts and the cost prediction are measured
rather than guessed, and so a template problem is found before any money moves.

Usage:  .\.venv\Scripts\python.exe scripts/preflight_qwen.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "environments" / "transformer_repair"))

MODEL_ID = "Qwen/Qwen3.5-4B"
RENDERERS = ("qwen3_5", "qwen3_5_disable_thinking")

PRICE_PREFILL = 0.33
PRICE_CACHED_PREFILL = 0.066
PRICE_SAMPLE = 1.005
CONTEXT_WINDOW = 65536


def main() -> int:
    from tinker_cookbook import model_info, renderers
    from transformers import AutoTokenizer

    from transformer_repair import SYSTEM_PROMPT, build_dataset

    print(f"model            {MODEL_ID}")
    print(f"recommended      {model_info.get_recommended_renderer_names(MODEL_ID)}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    dataset = build_dataset()
    report: dict = {"model": MODEL_ID, "context_window": CONTEXT_WINDOW, "renderers": {}}

    for name in RENDERERS:
        renderer = renderers.get_renderer(name, tok, model_name=MODEL_ID)
        stops = renderer.get_stop_sequences()
        per_task = {}
        for row in dataset:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["question"]},
            ]
            per_task[row["info"]["task_id"]] = renderer.build_generation_prompt(messages).length

        total = sum(per_task.values())
        report["renderers"][name] = {
            "thinking": name != "qwen3_5_disable_thinking",
            "stop_sequences": [int(s) if isinstance(s, int) else s for s in stops],
            "prompt_tokens_per_task": per_task,
            "prompt_tokens_total": total,
            "max_prompt_tokens": max(per_task.values()),
        }

        print(f"\n--- renderer: {name} ---")
        print(f"stop sequences   {stops}")
        for task_id, n in per_task.items():
            headroom = CONTEXT_WINDOW - n
            print(f"  {task_id:38s} prompt={n:6,d}  headroom={headroom:,d}")
        print(f"  {'TOTAL (5 prompts)':38s}       {total:6,d}")

    # Cost prediction for 5 prompts x 4 samples, under the pinned renderer.
    pinned = report["renderers"]["qwen3_5"]
    prefill = pinned["prompt_tokens_total"]
    print(f"\n{'=' * 70}\ncost prediction, pinned renderer qwen3_5, 5 prompts x 4 samples\n{'=' * 70}")
    print(f"prefill (billed once per prompt): {prefill:,} tokens -> ${prefill / 1e6 * PRICE_PREFILL:.5f}")
    for mean_sampled in (1000, 3000, 6000, 12000):
        sampled = mean_sampled * 20
        cost = prefill / 1e6 * PRICE_PREFILL + sampled / 1e6 * PRICE_SAMPLE
        print(
            f"  if mean sampled = {mean_sampled:6,d} tok/rollout -> "
            f"{sampled:7,d} sampled, total ${cost:.4f}  (${cost / 20:.5f}/rollout)"
        )
    print(
        "\nDeepSeek v4-pro used a mean of 2,151 completion tokens on these prompts,\n"
        "90% of it reasoning. Qwen3.5-4B with thinking on will plausibly use more.\n"
        "max_tokens is set to 16,384 in the runner so truncation cannot be\n"
        "mistaken for incapability -- that error cost 20 trajectories in Phase 0.5."
    )

    out = REPO_ROOT / "artifacts" / "qwen_preflight.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
