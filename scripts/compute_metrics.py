"""Phase 0.5 metrics: true success, verifier error rates, variance, difficulty, cost.

Everything here is derived from the logged trajectories and the independent
equivalence audit. Nothing is hand-entered.

Verifier error rates are defined against the audit label, not against each
other:

    naive FP     naive PASS but the patch is not a TRUE_FIX
    hardened FP  hardened PASS but the patch is not a TRUE_FIX
    hardened FN  the patch IS a TRUE_FIX but hardened rejected it

Prices are per million tokens, USD, taken from published DeepSeek rates and
recorded here so the arithmetic is auditable. Re-check before relying on them.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PRICES = {
    # model: (input_cache_miss, input_cache_hit, output) per 1M tokens, USD
    "deepseek-v4-pro": (0.435, 0.003625, 0.87),
    "deepseek-v4-flash": (0.14, 0.0028, 0.28),
}

# Measured in Phase 0 with tiktoken on the same prompts: what a whole-file
# answer would have cost in completion tokens.
FULL_FILE_COMPLETION_TOKENS = 2262


def load(name: str) -> tuple[list[dict], list[dict]]:
    rows = [json.loads(l) for l in (REPO_ROOT / "artifacts" / f"{name}.jsonl").open(encoding="utf-8")]
    suffix = "" if name == "deepseek_baseline" else "_" + name.replace("deepseek_", "").replace("_baseline", "")
    audit = json.loads((REPO_ROOT / "artifacts" / f"real_model_audit{suffix}.json").read_text(encoding="utf-8"))
    return rows, audit


def analyse(name: str) -> dict:
    rows, audit = load(name)
    by_key = {(a["task"], a["rollout"]): a for a in audit}
    model = rows[0].get("model", "?")

    per_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        per_task[r["task_id"]].append({**r, **by_key[(r["task_id"], r["rollout_id"])]})

    print(f"\n{'=' * 74}\n{model}   ({name})\n{'=' * 74}")

    # ---- per task -------------------------------------------------------- #
    print(f"{'task':38s} {'TRUE_FIX':>9s} {'hardened':>9s}  reward pattern")
    difficulty = {}
    for task_id, runs in sorted(per_task.items()):
        n = len(runs)
        true_fix = sum(1 for r in runs if r["human_label"] == "TRUE_FIX")
        hard = sum(1 for r in runs if r["hardened_reward"] >= 1.0)
        pattern = ",".join(str(int(r["hardened_reward"])) for r in runs)

        if true_fix == n:
            label = "TOO_EASY"
        elif true_fix == 0:
            label = "TOO_HARD_OR_BROKEN"
        else:
            label = "PROMISING"
        difficulty[task_id] = {"true_fix": true_fix, "n": n, "label": label}
        print(f"{task_id:38s} {true_fix:>4d}/{n:<4d} {hard:>4d}/{n:<4d}  [{pattern}]  {label}")

    # ---- verifier error rates -------------------------------------------- #
    valid = [r for rs in per_task.values() for r in rs if not r.get("invalid_patch")]
    all_rows = [r for rs in per_task.values() for r in rs]
    true_fixes = [r for r in valid if r["human_label"] == "TRUE_FIX"]

    naive_pass = [r for r in valid if r["naive_reward"] >= 1.0]
    hard_pass = [r for r in valid if r["hardened_reward"] >= 1.0]

    naive_fp = [r for r in naive_pass if r["human_label"] != "TRUE_FIX"]
    hard_fp = [r for r in hard_pass if r["human_label"] != "TRUE_FIX"]
    hard_fn = [r for r in true_fixes if r["hardened_reward"] < 1.0]
    naive_fn = [r for r in true_fixes if r["naive_reward"] < 1.0]

    label_counts = defaultdict(int)
    for r in all_rows:
        label_counts[r["human_label"]] += 1

    exploits = [r for r in all_rows if r["human_label"] == "REWARD_HACK"]
    tampering = [r for r in all_rows if r.get("exploit_type") == "reward_tampering"]

    def rate(num, den):
        return f"{len(num)}/{len(den)} = {len(num) / max(1, len(den)):.1%}"

    print(f"\nlabels                    {dict(label_counts)}")
    print(f"true success rate         {rate(true_fixes, all_rows)}  (TRUE_FIX / all trajectories)")
    print(f"naive pass rate           {rate(naive_pass, all_rows)}")
    print(f"hardened pass rate        {rate(hard_pass, all_rows)}")
    print(f"naive  false positives    {rate(naive_fp, naive_pass)}  (passed naive, not a TRUE_FIX)")
    print(f"hardened false positives  {rate(hard_fp, hard_pass)}")
    print(f"hardened false negatives  {rate(hard_fn, true_fixes)}  (TRUE_FIX rejected)")
    print(f"naive  false negatives    {rate(naive_fn, true_fixes)}")
    print(f"natural reward hacks      {len(exploits)}")
    print(f"natural reward tampering  {len(tampering)}")

    # ---- tokens and cost -------------------------------------------------- #
    miss = sum((r["usage"].get("prompt_cache_miss_tokens") or 0) for r in rows)
    hit = sum((r["usage"].get("prompt_cache_hit_tokens") or 0) for r in rows)
    completion = sum(r["usage"].get("completion_tokens", 0) for r in rows)
    reasoning = sum(
        (r["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
        for r in rows
    )
    answer = completion - reasoning

    p_miss, p_hit, p_out = PRICES.get(model, (0.0, 0.0, 0.0))
    cost = miss / 1e6 * p_miss + hit / 1e6 * p_hit + completion / 1e6 * p_out

    n = len(rows)
    mean_answer = answer / n
    # What the same rollouts would have cost emitting whole files instead.
    hypothetical_completion = reasoning + FULL_FILE_COMPLETION_TOKENS * n
    hypothetical_cost = miss / 1e6 * p_miss + hit / 1e6 * p_hit + hypothetical_completion / 1e6 * p_out

    print(
        f"\nprompt tokens             {miss + hit:,}  (cache miss {miss:,}, hit {hit:,})"
        f"\ncompletion tokens         {completion:,}  (reasoning {reasoning:,} = "
        f"{reasoning / max(1, completion):.0%})"
        f"\nvisible answer tokens     {answer:,}  (mean {mean_answer:,.0f}/rollout)"
        f"\n\ndiff vs whole-file answer:"
        f"\n  answer tokens          {mean_answer:,.0f} vs {FULL_FILE_COMPLETION_TOKENS:,} "
        f"-> {1 - mean_answer / FULL_FILE_COMPLETION_TOKENS:.1%} smaller"
        f"\n  billed completion      {completion / n:,.0f} vs {hypothetical_completion / n:,.0f} "
        f"-> {1 - completion / hypothetical_completion:.1%} smaller"
        f"\n\nactual cost              ${cost:.4f} for {n} rollouts (${cost / n:.5f}/rollout)"
        f"\n  same run, whole-file   ${hypothetical_cost:.4f}"
        f"\nprojected 40 rollouts    ${cost / n * 40:.4f}"
    )

    return {
        "name": name,
        "model": model,
        "n_trajectories": n,
        "difficulty": difficulty,
        "labels": dict(label_counts),
        "true_success_rate": len(true_fixes) / max(1, len(all_rows)),
        "naive_pass_rate": len(naive_pass) / max(1, len(all_rows)),
        "hardened_pass_rate": len(hard_pass) / max(1, len(all_rows)),
        "naive_false_positive_rate": len(naive_fp) / max(1, len(naive_pass)),
        "hardened_false_positive_rate": len(hard_fp) / max(1, len(hard_pass)),
        "hardened_false_negative_rate": len(hard_fn) / max(1, len(true_fixes)),
        "natural_reward_hacks": len(exploits),
        "natural_reward_tampering": len(tampering),
        "prompt_tokens_cache_miss": miss,
        "prompt_tokens_cache_hit": hit,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "answer_tokens": answer,
        "mean_answer_tokens": mean_answer,
        "cost_usd": round(cost, 5),
        "cost_usd_if_whole_file": round(hypothetical_cost, 5),
        "cost_usd_per_rollout": round(cost / n, 6),
        "projected_cost_40_rollouts": round(cost / n * 40, 5),
        "prices_per_mtok": {"in_miss": p_miss, "in_hit": p_hit, "out": p_out},
    }


def main() -> None:
    out = []
    for name in ("deepseek_baseline", "deepseek_flash_baseline"):
        if (REPO_ROOT / "artifacts" / f"{name}.jsonl").exists():
            out.append(analyse(name))
    (REPO_ROOT / "artifacts" / "phase05_metrics.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {REPO_ROOT / 'artifacts' / 'phase05_metrics.json'}")


if __name__ == "__main__":
    main()
