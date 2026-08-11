"""G9: break each defence on purpose and prove the corresponding check goes RED.

A test suite that passes tells you the code does what the tests check. It does not tell
you the tests check anything. Mutation testing is the difference: disable a defence, and
if nothing turns red, that defence was never actually verified -- it was decoration, and
this project has shipped decoration twice (R11's vacuously-green verifier, R15's denylist
that refused every canary before it could execute).

Each case below patches real source, runs a designated verification, and requires it to
fail. Then the source is restored byte-for-byte and the suite is re-run clean. A mutant
that SURVIVES is the finding; it means the protection is not tested and the gate that
depends on it is not earned.

Two cases are required beyond the five frozen ones, per PROTOCOL_CHANGELOG:
  * R15 -- remove the result-protocol authentication
  * R16 -- restore the /grader mount into the candidate container

Run: python scripts/post_success_mutation_test.py [--case NAME] [--keep-going]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "post_success_mutation_test.json"
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")

DOCKER_PREAMBLE = [
    "docker", "run", "--rm",
    "-v", "e:/RL:/run/desktop/mnt/host/e/RL",
    "-w", "/run/desktop/mnt/host/e/RL",
    "-e", "PYTHONPATH=/run/desktop/mnt/host/e/RL:/run/desktop/mnt/host/e/RL/environments",
    "-v", "/var/run/docker.sock:/var/run/docker.sock",
    "-v", "/tmp:/tmp",
    "trgym-v1:latest",
]


@dataclass
class Case:
    name: str
    path: str
    find: str
    replace: str
    defence: str
    """What is being disabled, in one line."""
    verify: str
    """`pytest:<target>` -- must exit non-zero.
       `canary:<probe>` -- that probe must LEAK on the sandboxed path."""
    count: int = 1


CASES: list[Case] = [
    # ------------------------------------------------------------------ R16
    Case(
        name="hidden_oracle_protection",
        path="trgym/harness/sandbox.py",
        find='            "-v", f"{staging}:/probe:ro",',
        replace='            "-v", f"{staging}:/probe:ro",\n'
                '            "-v", f"{REPO_ROOT}:/grader:ro",',
        defence="the candidate container holds no oracle (R16)",
        # The canary is the right instrument here: a static test proves the string
        # changed, the canary proves candidate code can actually read the hidden checks.
        verify="canary:oracle_source_read_in_container",
    ),
    Case(
        name="trusted_comparator_boundary",
        path="trgym/harness/sandbox.py",
        find='            "-v", f"{staging}:/probe:ro",',
        replace='            "-v", f"{staging}:/probe:ro",\n'
                '            "-v", f"{REPO_ROOT}:/grader:ro",',
        defence="gold is unreachable from candidate code by import or object graph (R16)",
        verify="canary:gold_oracle_import_evasive",
    ),
    # ------------------------------------------------------------------ R15
    Case(
        name="verdict_protocol_authentication",
        path="trgym/repo/obs_protocol.py",
        find="    at = text.rfind(marker)",
        replace="    at = text.find(marker.split(':')[0])",
        defence="the observation block is nonce-authenticated and last-wins (R15)",
        verify="pytest:tests/test_obs_protocol.py",
    ),
    # ------------------------------------------------------------------ frozen five
    Case(
        name="grading_artifact_isolation",
        path="environments/transformer_repair/grading.py",
        find="    allow_in_process: bool = False,",
        replace="    allow_in_process: bool = True,",
        defence="grading never runs candidate code in the process holding gold (R14)",
        verify="pytest:tests/test_grading_isolation.py",
    ),
    Case(
        name="return_type_contract",
        # Deliberately aimed at checks.py, not predicates.py. `test_repo_tasks.py` drives
        # `run_repo_checks`, the in-process surface; mutating the sandboxed predicate
        # instead would have SURVIVED, and survived for an uninteresting reason -- no test
        # in that suite ever reaches it. The parallel-surface risk this exposes is covered
        # by tests/test_check_surfaces_agree.py.
        path="trgym/repo/checks.py",
        find="        if type(loss) is not float:  # noqa: E721 - exact type is the contract",
        replace="        if False:  # noqa: E721 - exact type is the contract",
        defence="accumulate_gradients must return exactly float, not a subclass",
        verify="pytest:tests/test_repo_tasks.py",
    ),
    Case(
        name="sandboxed_predicate_enforcement",
        path="trgym/repo/predicates.py",
        # Aimed at the LR-schedule predicate specifically, because Tier S task s3 fails on
        # exactly that one check and nothing else. Disabling any predicate that s2 depends
        # on would leave the tree still failing on a sibling check, and the mutant would
        # survive for an uninteresting reason. `repo_contract_return_types` was the first
        # choice and was wrong for precisely that reason -- it is not in HIDDEN_S at all,
        # so Tier S grading never evaluates it.
        find="    want = _gold_lr_trace(task_id, steps)",
        replace="    want = got  # MUTANT: compare the candidate against itself",
        defence="predicates on the production surface actually enforce (R16)",
        # End-to-end: with the predicate neutered, s3's buggy tree stops failing and the
        # freeze preflight reports noop_fails=False.
        verify="freeze:noop_fails",
    ),
    Case(
        name="causal_mask",
        path="trgym/repo_template/tinygpt/attention.py",
        find="    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, "
             "device=scores.device).tril(diagonal=0)",
        replace="    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, "
                "device=scores.device).tril(diagonal=1)",
        defence="the reference implementation is causally correct",
        verify="pytest:tests/test_repo_tasks.py",
    ),
    Case(
        name="cross_job_state_isolation",
        path="trgym/harness/sandbox.py",
        find='            "--tmpfs=/tmp:rw,noexec,nosuid,size=128m",   # ...with a scratch dir',
        replace='            "-v", "/tmp:/tmp",   # MUTANT: shared scratch across jobs',
        defence="no grading job can see another job's temp state",
        verify="canary:temp_dir_persistence",
    ),
]


def run(cmd: list[str], timeout: int = 1800) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
    return proc.returncode, (proc.stdout + proc.stderr)[-4000:]


def verify_went_red(case: Case) -> dict:
    kind, target = case.verify.split(":", 1)
    if kind == "pytest":
        code, tail = run([PY, "-m", "pytest", target, "-q", "-p", "no:cacheprovider"])
        return {
            "kind": "pytest",
            "target": target,
            "exit_code": code,
            "went_red": code != 0,
            "tail": tail[-1200:],
        }
    if kind == "freeze":
        # The Tier S preflight is a true end-to-end exercise of the production grading
        # surface: it builds gold and buggy trees and grades them through
        # grade_workspace -> run_checks -> predicates. Must run in Linux (fcntl).
        code, tail = run(DOCKER_PREAMBLE + ["python", "scripts/tier_s_freeze.py"])
        art = json.loads((ROOT / "artifacts" / "tier_s_spec.json").read_text())
        tasks = art.get("tasks") or []
        still_fails = all(t.get("noop_fails") for t in tasks) if tasks else False
        return {
            "kind": "freeze",
            "target": target,
            "exit_code": code,
            "all_noop_still_fail": still_fails,
            # RED means the buggy tree stopped failing, i.e. the freeze no longer holds.
            "went_red": not still_fails,
            "tail": tail[-800:],
        }
    if kind == "canary":
        code, tail = run(DOCKER_PREAMBLE + ["python", "scripts/g5_isolation_canaries.py"])
        art = json.loads((ROOT / "artifacts" / "g5_isolation_canaries.json").read_text())
        rows = (art.get("results") or {}).get("sandboxed_container") or []
        row = next((r for r in rows if r.get("canary") == target), None)
        return {
            "kind": "canary",
            "target": target,
            "exit_code": code,
            "probe_leaked": bool(row and row.get("leaked")),
            "probe_executed": bool(row and row.get("payload_witnessed")),
            # A probe that leaks is the protection failing, which is what we want to see
            # while the mutant is in place.
            "went_red": bool(row and row.get("leaked")),
            "tail": tail[-800:],
        }
    raise ValueError(f"unknown verify kind {kind!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", action="append", default=None)
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify every anchor resolves exactly once; mutate nothing")
    args = ap.parse_args()

    if args.dry_run:
        bad = 0
        for case in CASES:
            found = (ROOT / case.path).read_text(encoding="utf-8").count(case.find)
            ok = found == case.count
            bad += 0 if ok else 1
            print(f"{'OK ' if ok else 'BAD'} {case.name:<34} {found}x  {case.path}")
        print(f"\n{len(CASES)} cases, {bad} bad anchors")
        return 0 if bad == 0 else 1

    cases = [c for c in CASES if not args.case or c.name in args.case]
    results = []
    all_restored = True

    for case in cases:
        path = ROOT / case.path
        original = path.read_text(encoding="utf-8")
        occurrences = original.count(case.find)
        row: dict = {
            "mutation": case.name,
            "file": case.path,
            "defence": case.defence,
            "verify": case.verify,
            "anchor_occurrences": occurrences,
        }
        if occurrences != case.count:
            # The anchor moved. Recording it as a survivor would be wrong -- nothing was
            # mutated -- but silently skipping would be worse, so it is a loud failure.
            row.update({"applied": False, "tests_went_red": False,
                        "error": f"anchor found {occurrences}x, expected {case.count}"})
            results.append(row)
            print(f"  !! {case.name}: anchor not found ({occurrences}x)")
            if not args.keep_going:
                break
            continue

        print(f"\n=== {case.name} ===\n  disabling: {case.defence}")
        started = time.perf_counter()
        try:
            path.write_text(original.replace(case.find, case.replace), encoding="utf-8")
            outcome = verify_went_red(case)
        finally:
            path.write_text(original, encoding="utf-8")
            restored = path.read_text(encoding="utf-8") == original
            all_restored = all_restored and restored

        row.update(outcome)
        row["applied"] = True
        row["restored"] = restored
        row["seconds"] = round(time.perf_counter() - started, 1)
        results.append(row)
        verdict = "RED (good)" if outcome["went_red"] else "SURVIVED -- DEFENCE UNTESTED"
        print(f"  {verdict}   [{row['seconds']}s, restored={restored}]")
        if not outcome["went_red"] and not args.keep_going:
            break

    survived = [r["mutation"] for r in results if not r.get("tests_went_red")
                and not r.get("went_red")]
    print("\n=== restoring and re-running clean ===")
    code, tail = run([PY, "-m", "pytest", "-q"])
    post_ok = code == 0
    print(tail[-400:])

    payload = {
        "mutations": [
            {**r, "tests_went_red": bool(r.get("went_red"))} for r in results
        ],
        "n_cases": len(results),
        "survived": survived,
        "all_restored": all_restored,
        "post_restore_tests_pass": post_ok,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nsurvived={survived or 'none'}  all_restored={all_restored}  "
          f"post_restore_tests_pass={post_ok}")
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")
    return 0 if (not survived and all_restored and post_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
