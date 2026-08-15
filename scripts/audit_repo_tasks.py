"""Quick discrimination check for the 5 repo-level medium tasks."""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trgym.repo.build import build_repo, build_gold, changed_files
from trgym.repo.checks import run_repo_checks
from trgym.tasks.repo_specs import REPO_TASKS

ok_all = True
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    for spec in REPO_TASKS:
        gold = build_gold(spec, root / f"{spec.task_id}_gold")
        bug = build_repo(spec, root / f"{spec.task_id}_bug", gold=False)
        touched = changed_files(bug, gold)
        gres = run_repo_checks(gold, spec.task_id, spec.visible_checks + spec.hidden_checks)
        bres = run_repo_checks(bug, spec.task_id, spec.visible_checks + spec.hidden_checks)
        gfail = [n for n, o, _ in gres if not o]
        bfail = [n for n, o, _ in bres if not o]
        vis_bug = [n for n, o, _ in bres if not o and n in spec.visible_checks]
        good = (not gfail) and bool(bfail) and not vis_bug
        ok_all = ok_all and good
        print(f"{'OK ' if good else 'BAD'} {spec.task_id:28s} files={len(touched)} "
              f"gold_fail={gfail or '-'} bug_hidden_fail={[n for n in bfail if n in spec.hidden_checks]} "
              f"bug_visible_fail={vis_bug or '-'}")
        for n, o, d in gres:
            if not o: print(f"      GOLD FAIL {n}: {d[:160]}")
print("\nALL OK" if ok_all else "\nNEEDS WORK")
