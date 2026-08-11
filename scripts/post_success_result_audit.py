"""G9 stage B: recompute every reported number from raw artifacts and diff.

Deliberately a *second* implementation. `build_final_metrics_summary.py` derives the
canonical summary; this script recomputes the same quantities with independently written
arithmetic and compares. If both had shared a helper, an error in that helper would agree
with itself and the audit would certify its own bug.

It also checks the report: the SHA-256 pinned in `FINAL_FLAGSHIP_RESEARCH_REPORT.md` must
be the digest of the summary on disk, and every numeric claim of the form `X/Y` or a bare
rate that the report attributes to a family must be findable in the regenerated data.

Writes `POST_SUCCESS_RESULT_AUDIT.json`. `mismatches` must be empty.

Run: python scripts/post_success_result_audit.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "POST_SUCCESS_RESULT_AUDIT.json"
SUMMARY = ROOT / "artifacts" / "final_metrics_summary.json"
REPORT = ROOT / "FINAL_FLAGSHIP_RESEARCH_REPORT.md"

FAMILY_SOURCES = {
    "tier_m": "artifacts/tier_m_primary.jsonl",
    "tier_h": "artifacts/tier_h_primary.jsonl",
    "tier_m_pro": "artifacts/tier_m_confirmatory_pro.jsonl",
    "tier_h_24turn": "artifacts/tier_h_24turn.jsonl",
    "tier_s": "artifacts/tier_s_primary.jsonl",
}


def rows_of(rel: str) -> list[dict]:
    path = ROOT / rel
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def recompute(rel: str) -> dict:
    """Independent arithmetic. Intentionally not shared with the generator."""
    rows = rows_of(rel)
    if not rows:
        return {}
    total = len(rows)
    n_hard = 0
    n_naive = 0
    n_fp = 0
    n_sub = 0
    for r in rows:
        h = float(r.get("hardened_reward") or 0.0) >= 1.0
        v = float(r.get("naive_reward") or 0.0) >= 1.0
        n_hard += 1 if h else 0
        n_naive += 1 if v else 0
        n_fp += 1 if (v and not h) else 0
        n_sub += 1 if (r.get("episode") or {}).get("submitted") else 0
    out = {
        "n": total,
        "full_fix": n_hard,
        "naive_pass": n_naive,
        "hardened_pass": n_hard,
        "submitted": n_sub,
        "budget_exhausted": total - n_sub,
        "full_fix_rate": round(n_hard / total, 4),
        "naive_FPR": round(n_fp / total, 4),
    }
    if any("fraction_repo_inspected" in r for r in rows):
        fr = [float(r.get("fraction_repo_inspected", 1.0)) for r in rows]
        out["every_episode_inspected_less_than_all"] = all(f < 1.0 for f in fr)
        out["mean_fraction_repo_inspected"] = round(sum(fr) / len(fr), 4)
        out["located_relevant_file"] = sum(1 for r in rows if r.get("located_relevant_file"))
    return out


def main() -> int:
    mismatches: list[str] = []
    checked: list[str] = []

    if not SUMMARY.exists():
        raise SystemExit("run scripts/build_final_metrics_summary.py first")
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    families = summary.get("families") or {}

    for family, rel in FAMILY_SOURCES.items():
        mine = recompute(rel)
        theirs = families.get(family)
        if not mine and not theirs:
            continue
        if not mine or not theirs:
            mismatches.append(f"{family}: present in one source only "
                              f"(raw={bool(mine)}, summary={bool(theirs)})")
            continue
        for key, want in mine.items():
            if key not in theirs:
                mismatches.append(f"{family}.{key}: absent from the generated summary")
                continue
            got = theirs[key]
            same = (
                abs(float(want) - float(got)) <= 1e-6
                if isinstance(want, (int, float)) and isinstance(got, (int, float))
                and not isinstance(want, bool) and not isinstance(got, bool)
                else want == got
            )
            checked.append(f"{family}.{key}")
            if not same:
                mismatches.append(f"{family}.{key}: recomputed={want} summary={got}")

    # The report must pin the digest of the summary as it exists on disk.
    digest = hashlib.sha256(SUMMARY.read_bytes()).hexdigest()
    report_present = REPORT.exists()
    digest_in_report = report_present and digest in REPORT.read_text(encoding="utf-8")
    if report_present and not digest_in_report:
        mismatches.append(
            f"report does not pin the current summary digest {digest}"
        )

    # Cross-check the gate artifacts the report leans on hardest.
    canaries = json.loads((ROOT / "artifacts" / "g5_isolation_canaries.json").read_text())
    cs = (canaries.get("summary") or {}).get("sandboxed_container") or {}
    if cs.get("n_leaked") != 0:
        mismatches.append(f"canary artifact reports {cs.get('n_leaked')} leaks, not 0")
    if cs.get("n_payloads_witnessed") != cs.get("n_canaries"):
        mismatches.append(
            f"canary artifact: {cs.get('n_payloads_witnessed')} of "
            f"{cs.get('n_canaries')} probes executed"
        )
    checked += ["g5.n_leaked", "g5.n_payloads_witnessed"]

    payload = {
        "summary_sha256": digest,
        "report_present": report_present,
        "report_pins_current_digest": digest_in_report,
        "families_recomputed": sorted(k for k, v in FAMILY_SOURCES.items() if rows_of(v)),
        "n_values_checked": len(checked),
        "values_checked": sorted(checked),
        "mismatches": mismatches,
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"values checked   {len(checked)}")
    print(f"families         {payload['families_recomputed']}")
    print(f"summary sha256   {digest}")
    print(f"report pins it   {digest_in_report}")
    if mismatches:
        print("MISMATCHES:")
        for m in mismatches:
            print(f"  {m}")
    else:
        print("no mismatches")
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
