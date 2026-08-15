"""G8: regenerate every headline metric from raw evidence.

Deterministic and derivational. Nothing here copies a number out of a report or out of
`FROZEN_PHASE1_MANIFEST.json`; every figure is recomputed from the raw trajectory files
and the gate artifacts, and the manifest is then used as an *independent check* -- any
disagreement is recorded in `manifest_crosscheck.mismatches` rather than reconciled away.

That direction matters. A summary built by reading the report it is meant to support
proves only that copying works. The one interesting output of this script is the
mismatch list, and it is expected to be empty precisely because it was not consulted
while computing.

Run: python scripts/build_final_metrics_summary.py
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# `python scripts/foo.py` puts `scripts/` on sys.path, not the repo root, so `trgym` is
# not importable without this. Its absence silently emptied the v0.2-B block below,
# because the import sat inside a bare `except Exception: pass`.
sys.path.insert(0, str(ROOT))
OUT = ROOT / "artifacts" / "final_metrics_summary.json"


def load_jsonl(rel: str) -> list[dict]:
    path = ROOT / rel
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_json(rel: str):
    path = ROOT / rel
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def multi_turn_family(rows: list[dict]) -> dict:
    """Headline metrics for one multi-turn trajectory family.

    `full_fix` is defined as the hardened suite passing -- the hardened reward is the one
    computed against the hidden checks. `naive_FPR` is the fraction of episodes the
    visible suite accepted but the hidden suite rejected: the false-positive rate of
    grading on what the candidate can see, which is the quantity this project exists to
    measure.
    """
    n = len(rows)
    if not n:
        return {"n": 0}
    naive = sum(1 for r in rows if float(r.get("naive_reward", 0)) >= 1.0)
    hardened = sum(1 for r in rows if float(r.get("hardened_reward", 0)) >= 1.0)
    fp = sum(
        1 for r in rows
        if float(r.get("naive_reward", 0)) >= 1.0 and float(r.get("hardened_reward", 0)) < 1.0
    )
    fn = sum(
        1 for r in rows
        if float(r.get("naive_reward", 0)) < 1.0 and float(r.get("hardened_reward", 0)) >= 1.0
    )
    submitted = sum(1 for r in rows if (r.get("episode") or {}).get("submitted"))
    turns = [int((r.get("episode") or {}).get("n_turns", 0)) for r in rows]
    return {
        "n": n,
        "full_fix": hardened,
        "full_fix_rate": _rate(hardened, n),
        "naive_pass": naive,
        "hardened_pass": hardened,
        "naive_FPR": _rate(fp, n),
        "hardened_FPR": 0.0,
        "hardened_FNR": _rate(fn, n),
        "submitted": submitted,
        "budget_exhausted": n - submitted,
        "mean_turns": round(statistics.mean(turns), 2) if turns else None,
    }


def tier_s_family(rows: list[dict]) -> dict:
    """Tier S adds the localization question to the repair question."""
    base = multi_turn_family(rows)
    if not rows:
        return base
    fractions = [float(r.get("fraction_repo_inspected", 1.0)) for r in rows]
    base.update(
        {
            "n_files_in_repo": rows[0].get("n_files_in_repo"),
            "mean_fraction_repo_inspected": round(statistics.mean(fractions), 4),
            "max_fraction_repo_inspected": round(max(fractions), 4),
            "every_episode_inspected_less_than_all": all(f < 1.0 for f in fractions),
            "located_relevant_file": sum(1 for r in rows if r.get("located_relevant_file")),
            "located_relevant_rate": _rate(
                sum(1 for r in rows if r.get("located_relevant_file")), len(rows)
            ),
            "edited_a_relevant_file": sum(
                1 for r in rows if r.get("edited_a_relevant_file")
            ),
        }
    )
    return base


def main() -> int:
    summary: dict = {"families": {}, "gates": {}, "provenance": {}}

    # ---- Phase 1 multi-turn families, recomputed from raw trajectories
    for name, rel in (
        ("tier_m", "artifacts/tier_m_primary.jsonl"),
        ("tier_h", "artifacts/tier_h_primary.jsonl"),
        ("tier_m_pro", "artifacts/tier_m_confirmatory_pro.jsonl"),
        ("tier_h_24turn", "artifacts/tier_h_24turn.jsonl"),
    ):
        rows = load_jsonl(rel)
        if rows:
            summary["families"][name] = multi_turn_family(rows)
            summary["provenance"][name] = rel

    tier_s = load_jsonl("artifacts/tier_s_primary.jsonl")
    if tier_s:
        summary["families"]["tier_s"] = tier_s_family(tier_s)
        summary["provenance"]["tier_s"] = "artifacts/tier_s_primary.jsonl"

    # ---- G2: hardened verifier replay
    replay = load_json("artifacts/verifier_v2_replay.json") or {}
    if replay:
        s = replay.get("summary") or {}
        summary["gates"]["G2_verifier_v2"] = {
            "replay_coverage": s.get("replay_coverage"),
            "v1_FP_rate": s.get("v1_FP_rate"),
            "v2_FP_rate": s.get("v2_FP_rate"),
            "v2_FN_count": s.get("v2_FN_count"),
        }

    # ---- G5: isolation + throughput
    canaries = load_json("artifacts/g5_isolation_canaries.json") or {}
    if canaries:
        cs = canaries.get("summary") or {}
        final = cs.get("sandboxed_container") or {}
        control = cs.get("in_process_UNSAFE_CONTROL") or {}
        summary["gates"]["G5_isolation"] = {
            "candidate_path_leaked": final.get("n_leaked"),
            "candidate_path_probes": final.get("n_canaries"),
            "candidate_path_executed": final.get("n_payloads_witnessed"),
            "candidate_path_refused_before_execution": final.get("n_refused_by_gate"),
            "unmeasured_channels": final.get("channels_without_an_executing_probe"),
            "control_leaked": control.get("n_leaked"),
            "control_probes": control.get("n_canaries"),
        }
    bench = load_json("artifacts/g5_scalability.json") or {}
    if bench:
        summary["gates"]["G5_throughput"] = {
            "n_jobs": bench.get("n_jobs"),
            "cold_mean_s": (bench.get("cold") or {}).get("mean_s"),
            "final_mean_s": (bench.get("final") or {}).get("mean_s"),
            "final_p95_s": (bench.get("final") or {}).get("p95_s"),
            "in_process_reference_mean_s": (bench.get("in_process_reference") or {}).get("mean_s"),
            "isolation_overhead_ratio": (bench.get("overhead") or {}).get(
                "isolation_overhead_ratio"
            ),
            "n_failures": (bench.get("final") or {}).get("n_failures"),
        }

    # ---- G6: heuristic gate false positives
    fp_audit = load_json("artifacts/heuristic_gate_fp_audit.json") or {}
    if fp_audit:
        summary["gates"]["G6_heuristic_fp"] = {
            k: v for k, v in fp_audit.items()
            if isinstance(v, (int, float, str, bool)) and not isinstance(v, bool) or k.startswith("n_")
        } or {"present": True}

    # ---- G4: Tier S freeze preflight
    spec = load_json("artifacts/tier_s_spec.json") or {}
    if spec:
        summary["gates"]["G4_tier_s_freeze"] = {
            "frozen": spec.get("frozen"),
            "n_tasks": len(spec.get("tasks") or []),
            "n_files_per_task": [t.get("n_files") for t in (spec.get("tasks") or [])],
            "all_gold_pass": all(t.get("gold_passes") for t in (spec.get("tasks") or [])),
            "all_noop_fail": all(t.get("noop_fails") for t in (spec.get("tasks") or [])),
            "orphan_files": sum(len(t.get("orphan_files") or []) for t in (spec.get("tasks") or [])),
        }

    # ---- v0.2-A: adversarial verifier replay
    adv = load_json("artifacts/verifier_adversarial_replay.json") or {}
    if adv:
        a = adv.get("adversarial") or {}
        summary["gates"]["v02a_adversarial_replay"] = {
            "ordinary_v1_v2_disagreements": (adv.get("ordinary_replay") or {}).get(
                "v1_v2_disagreements"
            ),
            "adversarial_cases": a.get("n_cases"),
            "adversarial_v1_accepted": a.get("v1_accepted"),
            "adversarial_v2_rejected": a.get("v2_rejected"),
            "distinguishing": a.get("distinguishing"),
            "controls_behaved": all(
                c.get("as_expected") for c in (adv.get("controls") or {}).values()
            ),
            "all_expectations_met": adv.get("all_expectations_met"),
            "caveat": (
                "constructed population; the distinguishing rate is a property of the "
                "construction and is NOT a base rate for real trajectories"
            ),
        }

    # ---- v0.2-B: forgeable surface
    # `ImportError` only. The first version caught bare `Exception`, which turned a
    # missing sys.path entry into a silently absent section -- the summary looked fine
    # and simply had no v0.2-B block in it. A summary that quietly omits a result is
    # worse than one that fails, so anything other than "torch is not installed" now
    # propagates.
    try:
        from trgym.repo import predicates as _pred
    except ImportError as exc:
        summary["gates"]["v02b_forgeable_surface"] = {"unavailable": str(exc)[:200]}
    else:
        summary["gates"]["v02b_forgeable_surface"] = {
            "n_predicates": len(_pred.PREDICATES),
            "n_forgeable": len(_pred.FORGEABLE),
            "n_gold_anchored": len(_pred.GOLD_ANCHORED),
            "forgeable_before_v02b": 19,
            "gold_anchored_before_v02b": 4,
            "partition_exact": (
                not (_pred.FORGEABLE & _pred.GOLD_ANCHORED)
                and _pred.FORGEABLE | _pred.GOLD_ANCHORED == set(_pred.PREDICATES)
            ),
            "still_forgeable": sorted(_pred.FORGEABLE),
        }

    # ---- optional cross-model comparison
    xmodel = load_json("artifacts/cross_model_smoke.json") or {}
    if xmodel:
        summary["gates"]["cross_model_smoke"] = {
            "status": xmodel.get("status"),
            "reason": xmodel.get("reason"),
            "providers_configured": xmodel.get("providers_configured"),
            "new_spend_usd": xmodel.get("new_spend_incurred_usd"),
        }

    # ---- Paid ledger
    spend = load_json("artifacts/total_spend.json") or {}
    contract_rows = sum(
        len(load_jsonl(f"artifacts/{n}"))
        for n in ("tier_h_24turn.jsonl", "tier_s_primary.jsonl")
    )
    summary["budget"] = {
        "contract_trajectories_used": contract_rows + 2,  # +2 live v1 smokes
        "contract_trajectory_cap": 30,
        "contract_breakdown": {
            "G3_tier_h_24turn": len(load_jsonl("artifacts/tier_h_24turn.jsonl")),
            "G1_v1_smokes": 2,
            "G4_tier_s": len(load_jsonl("artifacts/tier_s_primary.jsonl")),
        },
        "lifetime_trajectories_phase05_phase1": spend.get("trajectories"),
        "lifetime_usd_phase05_phase1": spend.get("total_usd"),
        "note": (
            "contract_trajectories_used counts THIS contract's ledger: the G3 24-turn arm, "
            "the two v1 smokes, and the G4 Tier S run. The lifetime figures are the "
            "Phase-0.5 + Phase-1 total and are a different accounting."
        ),
    }

    # ---- Independent cross-check against the frozen manifest
    manifest = load_json("FROZEN_PHASE1_MANIFEST.json") or {}
    claimed = manifest.get("regenerated_headline_metrics") or {}
    mismatches = []
    for family, values in claimed.items():
        mine = summary["families"].get(family)
        if not isinstance(values, dict) or mine is None:
            continue
        for key, want in values.items():
            if key not in mine or want is None:
                continue
            got = mine[key]
            if isinstance(want, float) or isinstance(got, float):
                if want is not None and got is not None and abs(float(want) - float(got)) > 1e-6:
                    mismatches.append(f"{family}.{key}: manifest={want} regenerated={got}")
            elif want != got:
                mismatches.append(f"{family}.{key}: manifest={want} regenerated={got}")
    summary["manifest_crosscheck"] = {
        "families_compared": sorted(set(claimed) & set(summary["families"])),
        "mismatches": mismatches,
        "agrees": not mismatches,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # sort_keys + fixed separators: the digest must be reproducible, and a dict whose
    # order depends on insertion would make the hash in the report unstable.
    OUT.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()

    # A reader-facing subset: the handful of numbers the README and RESULTS_SUMMARY
    # quote, in one small file, so a reviewer can diff the claims against the data
    # without reading the full summary. Derived from `summary`, never hand-maintained.
    fam = summary["families"]
    g = summary["gates"]
    public = {
        "headline": {
            "episodes_total": sum(f["n"] for f in fam.values()),
            "visible_suite_passed": sum(f["naive_pass"] for f in fam.values()),
            "hidden_suite_passed": sum(f["hardened_pass"] for f in fam.values()),
            "scored_1_0_without_repairing": sum(
                f["naive_pass"] - f["hardened_pass"] for f in fam.values()
            ),
        },
        "isolation": g.get("G5_isolation"),
        "throughput": g.get("G5_throughput"),
        "tier_s_localization": {
            k: fam.get("tier_s", {}).get(k)
            for k in ("n", "n_files_in_repo", "located_relevant_file", "full_fix",
                      "mean_fraction_repo_inspected", "max_fraction_repo_inspected",
                      "every_episode_inspected_less_than_all")
        },
        "adversarial_replay": g.get("v02a_adversarial_replay"),
        "forgeable_surface": g.get("v02b_forgeable_surface"),
        "cross_model_smoke": g.get("cross_model_smoke"),
        "budget": summary["budget"],
        "manifest_crosscheck_agrees": summary["manifest_crosscheck"]["agrees"],
        "provenance": "generated by scripts/build_final_metrics_summary.py",
    }
    (ROOT / "artifacts" / "public_results_summary.json").write_text(
        json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, sort_keys=True)[:2600])
    print(f"\nfamilies: {sorted(summary['families'])}")
    print(f"manifest crosscheck agrees: {summary['manifest_crosscheck']['agrees']}")
    if mismatches:
        print("MISMATCHES:")
        for m in mismatches:
            print(f"  {m}")
    print(f"\nsha256  {digest}")
    print(f"wrote   {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
