"""Recompute Tier S patch metrics after the R17 fingerprint fix. No model calls.

`repo_fingerprint` hashed only `tinygpt/*.py`, so for Tier S -- whose implementation lives
in subpackages -- every edit below the top level was invisible. `files_edited_by_model`
and `edited_a_relevant_file` were therefore wrong for all 12 episodes.

The rewards are untouched by this: `naive_reward` and `hardened_reward` come from real
grading of the workspace on disk, which never consulted the fingerprint. Only the patch
metrics were affected, and they can be recovered exactly, because the graded workspaces
still exist under `.sandbox_work/` and the buggy baseline is deterministic from the spec.

This is a correction of a derived column from surviving raw evidence, not a re-run and not
a re-tune. It spends no trajectories. The original values are preserved alongside the
corrected ones so the error stays auditable.

Run: python scripts/tier_s_recompute_patch_metrics.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SRC = ROOT / "artifacts" / "tier_s_primary.jsonl"
WORK = ROOT / ".sandbox_work"


def fingerprint(pkg: Path) -> dict[str, str]:
    import hashlib

    out = {}
    for path in sorted(pkg.rglob("*.py")):
        rel = path.relative_to(pkg).as_posix()
        out[f"tinygpt/{rel}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def main() -> int:
    from trgym.repo.build import build_gold, build_repo
    from trgym.tasks.repo_specs_s import REPO_TASKS_S

    specs = {s.task_id: s for s in REPO_TASKS_S}
    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]

    staging = Path(tempfile.mkdtemp(prefix="tier_s_recompute_"))
    corrected = 0
    try:
        baselines: dict[str, dict[str, str]] = {}
        golds: dict[str, dict[str, str]] = {}
        for task_id, spec in specs.items():
            baselines[task_id] = fingerprint(
                build_repo(spec, staging / f"{task_id}__buggy") / "tinygpt"
            )
            golds[task_id] = fingerprint(
                build_gold(spec, staging / f"{task_id}__gold") / "tinygpt"
            )

        for row in rows:
            task_id = row["task_id"]
            ws = WORK / f"{task_id}__e{row['episode_id']}" / "tinygpt"
            if not ws.exists():
                row["patch_metrics_recomputed"] = False
                row["patch_metrics_note"] = "workspace no longer on disk"
                continue

            after = fingerprint(ws)
            before = baselines[task_id]
            gold = golds[task_id]
            relevant = set(row.get("files_expected") or [])

            edited = sorted(k for k in before if after.get(k) != before[k])
            edited += sorted(k for k in after if k not in before)
            still = sorted(k for k in gold if after.get(k) != gold[k])

            row["files_edited_by_model_ORIGINAL_BUGGY"] = row.get("files_edited_by_model")
            row["files_still_differing_from_gold_ORIGINAL_BUGGY"] = row.get(
                "files_still_differing_from_gold"
            )
            row["edited_a_relevant_file_ORIGINAL_BUGGY"] = row.get("edited_a_relevant_file")

            row["files_edited_by_model"] = sorted(set(edited))
            row["files_still_differing_from_gold"] = still
            row["edited_a_relevant_file"] = bool(set(edited) & relevant)
            row["exactly_matches_gold"] = not still
            row["patch_metrics_recomputed"] = True
            corrected += 1

        SRC.write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8"
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    n_edit_rel = sum(1 for r in rows if r.get("edited_a_relevant_file"))
    n_gold = sum(1 for r in rows if r.get("exactly_matches_gold"))
    n_fixed = sum(1 for r in rows if float(r.get("hardened_reward") or 0) >= 1.0)
    print(f"rows recomputed            {corrected}/{len(rows)}")
    print(f"edited a relevant file     {n_edit_rel}/{len(rows)}   (was 0 -- the bug)")
    print(f"exactly matches gold       {n_gold}/{len(rows)}")
    print(f"hardened pass              {n_fixed}/{len(rows)}   (unchanged; never affected)")
    for r in rows:
        print(f"  {r['task_id']:<34} e{r['episode_id']}  "
              f"hardened={r.get('hardened_reward')}  "
              f"edited={r.get('files_edited_by_model')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
