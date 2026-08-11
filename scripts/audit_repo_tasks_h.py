import sys, tempfile
from pathlib import Path
sys.path.insert(0, r"e:\RL")
from trgym.repo.build import build_gold, build_repo
from trgym.repo.checks import run_repo_checks
from trgym.tasks.repo_specs_h import REPO_TASKS_H
ok_all = True
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    for spec in REPO_TASKS_H:
        gold = build_gold(spec, root / f"{spec.task_id}_g")
        bug = build_repo(spec, root / f"{spec.task_id}_b")
        g = run_repo_checks(gold, spec.task_id, spec.visible_checks + spec.hidden_checks)
        b = run_repo_checks(bug, spec.task_id, spec.visible_checks + spec.hidden_checks)
        gf = [n for n,o,_ in g if not o]
        bh = [n for n,o,_ in b if not o and n in spec.hidden_checks]
        bv = [n for n,o,_ in b if not o and n in spec.visible_checks]
        good = (not gf) and bool(bh) and not bv
        ok_all = ok_all and good
        print(f"{'OK ' if good else 'BAD'} {spec.task_id:30s} files={len(spec.mutations)} gold_fail={gf or '-'} bug_hidden_fail={bh} bug_visible_fail={bv or '-'}")
        for n,o,d in g:
            if not o: print(f"      GOLD FAIL {n}: {d[:150]}")
print("\nALL OK" if ok_all else "\nNEEDS WORK")
