"""G2 -- replay every recoverable historical trajectory through v1, v2 and the oracle.

Per VERIFIER_V2_PROTOCOL.md. No model is ever re-invoked: candidate sources come
from the preserved workspaces (Tier M/H/pro) or from the `patched_source` field
recorded in the Phase 0.5 JSONL (Tier E). Anything unrecoverable is labelled
UNREPLAYABLE and stays in the denominator.

The gate's anti-cheat criterion is criterion 7 of the protocol: making a verifier
stricter always lowers FPR, so v2 only counts as progress if it lowers FP
*without* rejecting a single oracle-labelled FULL_FIX.

Usage:  python scripts/verifier_v2_replay.py
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WORK = ROOT / ".replay_work"

# (jsonl, audit json, workspace root or None if source is in the jsonl, tier)
POPULATION = [
    ("tier_m_primary.jsonl", "tier_m_audit.json", ".sandbox_work", "M"),
    ("tier_h_primary.jsonl", "tier_h_audit.json", ".sandbox_work_h", "H"),
    ("tier_m_confirmatory_pro.jsonl", "tier_m_confirmatory_pro_audit.json",
     ".sandbox_work_pro", "M-pro"),
    ("deepseek_baseline.jsonl", "real_model_audit.json", None, "E-pro"),
    ("deepseek_flash_baseline.jsonl", "real_model_audit_flash.json", None, "E-flash"),
]


def reconstruct_repo_workspace(spec, task_id: str, ep, tier: str) -> Path | None:
    """Rebuild a lost workspace from the buggy template + preserved edited files.

    The audit copied every file each trajectory edited into
    artifacts/raw/<tag>_final_sources/. Starting from the buggy repo and applying
    those files reproduces the exact final state the grader saw, without calling
    a model. Trajectories that edited nothing need no preserved file and are
    reconstructed as the untouched buggy repo, which is what the grader graded.
    """
    from trgym.repo.build import build_repo

    tag = {"M": "tier_m", "H": "tier_h", "M-pro": "tier_m_confirmatory_pro"}.get(tier)
    if tag is None:
        return None
    src_dir = ROOT / "artifacts" / "raw" / f"{tag}_final_sources"
    if not src_dir.exists():
        return None

    prefix = f"{task_id}__e{ep}__"
    preserved = sorted(src_dir.glob(f"{prefix}*"))

    ws = build_repo(spec, WORK / f"recon_{tag}_{task_id}_e{ep}", gold=False)
    for path in preserved:
        rel = path.name[len(prefix):].replace("__", "/")
        target = ws / rel
        if not target.parent.exists():
            return None
        shutil.copy2(path, target)
    return ws


def replay_repo_trajectory(spec, ws: Path) -> tuple[bool, bool, list[str], list[str]]:
    """Run v1 and v2 check sets against a preserved repo workspace."""
    from trgym.repo.checks import run_repo_checks
    from trgym.repo.verifier_v2 import v1_checks, v2_checks

    r1 = run_repo_checks(ws, spec.task_id, v1_checks(spec))
    r2 = run_repo_checks(ws, spec.task_id, v2_checks(spec))
    f1 = [n for n, ok, _ in r1 if not ok]
    f2 = [n for n, ok, _ in r2 if not ok]
    return (not f1), (not f2), f1, f2


def replay_tier_e_trajectory(task_id: str, patched_source: str) -> tuple:
    """Tier E: single-file tasks, source recorded in the JSONL.

    v1 == the frozen Tier E hidden suite. v2 adds a return-type contract probe
    equivalent to the repo-tier one, which Tier E already had for `train_loop`
    via `grad_accum_runs` -- so for Tier E v1 and v2 coincide by construction and
    that is reported rather than hidden.
    """
    from trgym.tasks.build import build_workspace
    from trgym.tasks.registry import get_task
    from trgym.verifier.reward import grade

    spec = get_task(task_id)
    ws = build_workspace(spec, WORK / f"e_{task_id}", gold=False)
    (ws / spec.target_file).write_text(patched_source, encoding="utf-8")
    res = grade(spec, ws)
    failed = [r.name for r in res.hidden.results if not r.passed]
    passed = res.hardened_reward >= 1.0
    return passed, passed, failed, failed


def main() -> int:
    from trgym.tasks.repo_specs import get_repo_task

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    rows: list[dict] = []
    for jsonl, audit_name, work_root, tier in POPULATION:
        jpath = ROOT / "artifacts" / jsonl
        apath = ROOT / "artifacts" / audit_name
        if not (jpath.exists() and apath.exists()):
            print(f"skip {jsonl}: missing jsonl or audit")
            continue

        recs = [json.loads(l) for l in jpath.open(encoding="utf-8")]
        audit = {}
        for a in json.loads(apath.read_text(encoding="utf-8")):
            key = (a["task"], a.get("episode", a.get("rollout")))
            audit[key] = a
            # Phase 0.5 audits use `human_label`; Tier M/H use `label`.
            a.setdefault("label", a.get("human_label", "UNKNOWN"))

        for rec in recs:
            task_id = rec["task_id"]
            ep = rec.get("episode_id", rec.get("rollout_id"))
            a = audit.get((task_id, ep))
            oracle_label = a["label"] if a else "UNKNOWN"

            row = {
                "tier": tier, "task": task_id, "episode": ep,
                "model": rec.get("model", "?"),
                "oracle_label": oracle_label,
                "naive_reward": rec.get("naive_reward"),
                "hardened_v1_recorded": rec.get("hardened_reward"),
            }

            try:
                if work_root is not None:
                    spec = get_repo_task(task_id)
                    ws = ROOT / work_root / f"{task_id}__e{ep}"
                    if not ws.exists():
                        # Plan C: the live workspace is gone (scripts/build_sandbox.py
                        # wipes .sandbox_work, which destroyed Tier M's). Rebuild it
                        # deterministically from the buggy template plus the
                        # per-trajectory sources preserved by the audit. No model is
                        # re-invoked, so this is a reconstruction, not a replacement.
                        ws = reconstruct_repo_workspace(spec, task_id, ep, tier)
                        if ws is None:
                            row.update(
                                status="UNREPLAYABLE",
                                reason="workspace gone and no preserved sources",
                            )
                            rows.append(row)
                            continue
                        row["recovery"] = "RECONSTRUCTED_FROM_RAW"
                    p1, p2, f1, f2 = replay_repo_trajectory(spec, ws)
                else:
                    src = rec.get("patched_source") or ""
                    if not src.strip():
                        row.update(status="UNREPLAYABLE", reason="no patched_source")
                        rows.append(row)
                        continue
                    p1, p2, f1, f2 = replay_tier_e_trajectory(task_id, src)
            except Exception as exc:  # noqa: BLE001
                row.update(status="UNREPLAYABLE", reason=f"{type(exc).__name__}: {exc}")
                rows.append(row)
                continue

            genuine = oracle_label in ("FULL_FIX", "SEMANTIC_FIX")
            full = oracle_label == "FULL_FIX"
            row.update(
                status="REPLAYED",
                hardened_v1_pass=p1,
                hardened_v2_pass=p2,
                v1_failed=";".join(f1),
                v2_failed=";".join(f2),
                v1_FP=p1 and not genuine,
                v2_FP=p2 and not genuine,
                v1_FP_vs_full=p1 and not full,
                v2_FP_vs_full=p2 and not full,
                v1_FN=(not p1) and full,
                v2_FN=(not p2) and full,
                v1_v2_disagree=p1 != p2,
                contract_only_rejection=p1 and not p2 and oracle_label == "SEMANTIC_FIX",
                v2_vs_oracle_disagree=p2 != full,
            )
            rows.append(row)
            print(
                f"{tier:8s} {task_id:28s} e{ep}  oracle={oracle_label:12s} "
                f"v1={'P' if p1 else 'F'} v2={'P' if p2 else 'F'}"
                f"{'  <-- v1/v2 DISAGREE' if p1 != p2 else ''}"
            )

    replayed = [r for r in rows if r["status"] == "REPLAYED"]
    unreplayable = [r for r in rows if r["status"] == "UNREPLAYABLE"]

    def rate(num: int, den: int) -> float | None:
        return round(num / den, 4) if den else None

    v1_pass = [r for r in replayed if r["hardened_v1_pass"]]
    v2_pass = [r for r in replayed if r["hardened_v2_pass"]]
    full = [r for r in replayed if r["oracle_label"] == "FULL_FIX"]

    summary = {
        "population": len(rows),
        "replayed": len(replayed),
        "unreplayable": len(unreplayable),
        "replay_coverage": rate(len(replayed), len(rows)),
        "multi_turn_replayed": sum(1 for r in replayed if r["tier"] in ("M", "H", "M-pro")),
        "oracle_labels": dict(Counter(r["oracle_label"] for r in replayed)),
        "v1_FP_rate": rate(sum(r["v1_FP"] for r in replayed), len(v1_pass)),
        "v2_FP_rate": rate(sum(r["v2_FP"] for r in replayed), len(v2_pass)),
        "v1_FP_rate_vs_full": rate(sum(r["v1_FP_vs_full"] for r in replayed), len(v1_pass)),
        "v2_FP_rate_vs_full": rate(sum(r["v2_FP_vs_full"] for r in replayed), len(v2_pass)),
        "v1_FN_count": sum(r["v1_FN"] for r in replayed),
        "v2_FN_count": sum(r["v2_FN"] for r in replayed),
        "v1_FN_rate": rate(sum(r["v1_FN"] for r in replayed), len(full)),
        "v2_FN_rate": rate(sum(r["v2_FN"] for r in replayed), len(full)),
        "v1_v2_disagreements": sum(r["v1_v2_disagree"] for r in replayed),
        "contract_only_rejections": sum(r["contract_only_rejection"] for r in replayed),
        "v2_vs_oracle_disagreements": sum(r["v2_vs_oracle_disagree"] for r in replayed),
    }

    # Protocol criterion 7: stricter is only progress if it creates no false negatives.
    crit = {
        "c2_gold_all_pass": "checked by scripts/check_v2_contract.py",
        "c4_no_new_full_fix_rejection": summary["v2_FN_count"] == 0,
        "c6_coverage_ge_90pct": (summary["replay_coverage"] or 0) >= 0.90,
        "c7_fp_not_reduced_by_over_rejecting": (
            (summary["v2_FP_rate"] or 0) <= (summary["v1_FP_rate"] or 0)
            and summary["v2_FN_count"] == 0
        ),
    }
    summary["protocol_criteria"] = crit

    print("\n" + "=" * 74)
    for k, v in summary.items():
        if k != "protocol_criteria":
            print(f"{k:36s} {v}")
    print("-" * 74)
    for k, v in crit.items():
        print(f"{k:36s} {v}")

    fields = [
        "tier", "task", "episode", "model", "status", "reason", "oracle_label",
        "naive_reward", "hardened_v1_recorded", "hardened_v1_pass", "hardened_v2_pass",
        "v1_FP", "v2_FP", "v1_FP_vs_full", "v2_FP_vs_full", "v1_FN", "v2_FN",
        "v1_v2_disagree", "contract_only_rejection", "v2_vs_oracle_disagree",
        "v1_failed", "v2_failed",
    ]
    out_csv = ROOT / "VERIFIER_V2_REPLAY.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (ROOT / "artifacts" / "verifier_v2_replay.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {out_csv}")
    shutil.rmtree(WORK, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
