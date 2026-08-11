"""Re-grade logged trajectories without re-sampling the model.

The first baseline run rejected 4/20 patches as INVALID. Reading them showed the
fault was ours: the applier demanded byte-exact context, and the model had
paraphrased a docstring or skipped one. `patch(1)` handles that with fuzz; we now
do too. The model outputs are fixed data in the JSONL, so re-grading them is a
correction of our measurement, not a re-roll of the experiment.

Both gradings are kept: `invalid_patch_strict` records what the byte-exact
applier said, `invalid_patch` what the fuzzy one says.

Usage:  python scripts/regrade_baseline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "environments" / "transformer_repair"))

LOG = REPO_ROOT / "artifacts" / "deepseek_baseline.jsonl"


def main() -> int:
    from transformer_repair import _grade_submission

    from trgym.patching import last_fuzz

    rows = [json.loads(line) for line in LOG.open(encoding="utf-8")]
    changed = 0

    for r in rows:
        if r.get("error"):
            continue
        before = {
            "naive": r["naive_reward"],
            "hardened": r["hardened_reward"],
            "invalid": r["invalid_patch"],
        }
        graded = _grade_submission(r["response_content"], {"task_id": r["task_id"]})

        r["invalid_patch_strict"] = before["invalid"]
        r["naive_reward"] = graded["naive"]
        r["hardened_reward"] = graded["hardened"]
        r["invalid_patch"] = graded["invalid_patch"]
        r["invalid_reason"] = graded.get("invalid_reason", "")
        r["gates_fired"] = graded["gates"]
        r["hidden_checks_failed"] = graded["hidden_failed"]
        r["grade_metrics"] = graded.get("metrics", {})
        r["patched_source"] = graded.get("patched_source", "")
        r["patch_fuzz_lines"] = last_fuzz() if not graded["invalid_patch"] else None

        if (before["naive"], before["hardened"], before["invalid"]) != (
            graded["naive"],
            graded["hardened"],
            graded["invalid_patch"],
        ):
            changed += 1
            print(
                f"{r['task_id']:38s} r{r['rollout_id']}  "
                f"naive {before['naive']:.0f}->{graded['naive']:.0f}  "
                f"hardened {before['hardened']:.0f}->{graded['hardened']:.0f}  "
                f"invalid {before['invalid']}->{graded['invalid_patch']}"
            )

    with LOG.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    still_invalid = sum(1 for r in rows if r.get("invalid_patch"))
    was_invalid = sum(1 for r in rows if r.get("invalid_patch_strict"))
    fuzzed = sum(1 for r in rows if (r.get("patch_fuzz_lines") or 0) > 0)
    print(
        f"\n{changed} rollouts changed verdict"
        f"\ninvalid under byte-exact applier : {was_invalid}/{len(rows)}"
        f"\ninvalid under fuzzy applier      : {still_invalid}/{len(rows)}"
        f"\npatches that needed fuzz         : {fuzzed}/{len(rows)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
