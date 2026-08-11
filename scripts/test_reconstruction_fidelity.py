"""Is the G2 replay's reconstruction faithful, or is it the cause of the v2 FN?

The one v2 false negative (`m5_masking_interaction` e1) sits on a row recovered by
`RECONSTRUCTED_FROM_RAW`. Two mutually exclusive explanations:

  (a) v2 correctly caught a contract violation the original audit's oracle missed;
  (b) the reconstruction differs from the workspace the audit actually graded, so
      the replay measured a different artifact.

Discriminating test: run the *oracle* on the reconstruction. The original audit
recorded `contract_ok = True` for this trajectory on the live workspace. If the
oracle now reports a contract problem on the reconstruction, the two artifacts
differ and (b) holds. If the oracle still says the contract is fine while v2 says
it is broken, they genuinely disagree about the same bytes and (a) holds.

Usage:  python scripts/test_reconstruction_fidelity.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WORK = ROOT / ".fidelity_work"
TASK, EP, TIER = "m5_masking_interaction", 1, "M"


def main() -> int:
    from scripts.fuzz_verifier import independent_truth
    from scripts.verifier_v2_replay import reconstruct_repo_workspace
    from trgym.repo.checks import run_repo_checks
    from trgym.repo.verifier_v2 import CONTRACT_CHECKS
    from trgym.tasks.repo_specs import get_repo_task

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)

    # reconstruct_repo_workspace writes under its own module-level WORK; point it here
    import scripts.verifier_v2_replay as replay_mod

    replay_mod.WORK = WORK

    spec = get_repo_task(TASK)
    ws = reconstruct_repo_workspace(spec, TASK, EP, TIER)
    if ws is None:
        print("could not reconstruct; nothing to test")
        return 1
    print(f"reconstructed at {ws}")

    preserved = sorted(
        (ROOT / "artifacts" / "raw" / "tier_m_final_sources").glob(f"{TASK}__e{EP}__*")
    )
    print(f"preserved files applied: {[p.name for p in preserved]}")

    # What the original audit recorded, on the live workspace.
    audit = json.loads((ROOT / "artifacts" / "tier_m_audit.json").read_text(encoding="utf-8"))
    original = next(a for a in audit if a["task"] == TASK and a["episode"] == EP)
    print(
        f"\noriginal audit (live workspace): label={original['label']} "
        f"semantic_ok={original['semantic_ok']} contract_ok={original['contract_ok']} "
        f"files_edited={original['files_edited']}"
    )

    # The oracle, re-run on the reconstruction.
    semantic_ok, sem_notes, contract_notes = independent_truth(spec, ws)
    print(
        f"oracle on reconstruction     : semantic_ok={semantic_ok} "
        f"contract_ok={not contract_notes}"
    )
    if sem_notes:
        print(f"   semantic notes: {sem_notes[:3]}")
    if contract_notes:
        print(f"   contract notes: {contract_notes[:3]}")

    # v2's contract layer, on the same bytes.
    v2 = run_repo_checks(ws, TASK, CONTRACT_CHECKS)
    print(f"v2 contract layer            : {[(n, ok) for n, ok, _ in v2]}")
    for n, ok, detail in v2:
        if not ok:
            print(f"   {n}: {detail[:220]}")

    oracle_contract_ok = not contract_notes
    v2_contract_ok = all(ok for _, ok, _ in v2)

    print("\n" + "=" * 70)
    if oracle_contract_ok != v2_contract_ok:
        print(
            "VERDICT (a): oracle and v2 disagree about the SAME bytes.\n"
            "  The reconstruction is faithful enough to reproduce the original\n"
            "  audit's contract verdict, so the v2 false negative is a genuine\n"
            "  verifier-vs-oracle disagreement and must be explained on its merits."
        )
    elif not oracle_contract_ok and original["contract_ok"]:
        print(
            "VERDICT (b): the oracle NOW reports a contract problem where the\n"
            "  original audit did not. The reconstruction differs from the graded\n"
            "  workspace, so the replay measured a different artifact and the v2\n"
            "  false negative is an artifact of reconstruction, not of v2."
        )
    else:
        print(
            "VERDICT: oracle and v2 agree on the reconstruction; the discrepancy is\n"
            "  between the reconstruction and the original live workspace."
        )
    shutil.rmtree(WORK, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
