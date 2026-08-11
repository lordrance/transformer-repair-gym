import sys, tempfile
from pathlib import Path
sys.path.insert(0, r"e:\RL")
from trgym.repo.build import build_gold, build_repo
from trgym.repo.checks import run_repo_checks
from trgym.repo.verifier_v2 import CONTRACT_CHECKS
from trgym.tasks.repo_specs import REPO_TASKS
from trgym.tasks.repo_specs_h import REPO_TASKS_H
ok=True
with tempfile.TemporaryDirectory() as tmp:
    for spec in REPO_TASKS + REPO_TASKS_H:
        g = run_repo_checks(build_gold(spec, Path(tmp)/f"{spec.task_id}_g"), spec.task_id, CONTRACT_CHECKS)
        b = run_repo_checks(build_repo(spec, Path(tmp)/f"{spec.task_id}_b"), spec.task_id, CONTRACT_CHECKS)
        gf=[n for n,o,d in g if not o]; bf=[(n,d[:90]) for n,o,d in b if not o]
        good = not gf
        ok = ok and good
        print(f"{'OK ' if good else 'BAD'} {spec.task_id:30s} gold_contract_fail={gf or '-'} buggy_contract_fail={[n for n,_ in bf] or '-'}")
        for n,d in bf: print(f"      buggy {n}: {d}")
        for n,o,d in g:
            if not o: print(f"      GOLD FAIL {n}: {d[:160]}")
print("\nCONTRACT LAYER OK" if ok else "\nCONTRACT LAYER BROKEN")
