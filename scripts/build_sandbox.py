"""Build the sandbox image and verify it grades a task under real isolation."""
import shutil, sys
from pathlib import Path

# Derived, not hardcoded. This used to be the literal string `e:\RL`, which worked on the
# machine it was written on and produced `.../e:\RL/.sandbox_selftest/g:/workspace:rw` on
# Linux -- an invalid Docker mount spec ("too many colons"). Nothing local caught it,
# because locally the constant happened to be correct; CI on ubuntu did.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trgym.harness import sandbox
from trgym.repo.build import build_gold, build_repo
from trgym.tasks.repo_specs import get_repo_task

print("docker available:", sandbox.docker_available())
if not sandbox.image_exists():
    print("building image (pulls the CPU torch wheel, several minutes)...")
    ok, log = sandbox.build_image()
    print("build ok:", ok)
    if not ok:
        print(log[-2500:]); raise SystemExit(1)
print("image present:", sandbox.image_exists())

# Docker Desktop cannot bind-mount from %LOCALAPPDATA%\Temp, so the scratch area
# lives under the repo. It must NOT be an evidence directory: this script rmtree's
# it, and using .sandbox_work here destroyed Tier M's 20 graded workspaces.
work = ROOT / ".sandbox_selftest"   # NEVER an evidence dir; see PROTOCOL_CHANGELOG R5
if work.exists(): shutil.rmtree(work)
work.mkdir(parents=True)

spec = get_repo_task("m1_attention_regression")
rc = 0
for label, ws in (("gold", build_gold(spec, work / "g")), ("buggy", build_repo(spec, work / "b"))):
    r = sandbox.run_checks(ws, spec.task_id, spec.hidden_checks)
    failed = [n for n, ok, _ in r.results if not ok]
    print(f"{label:6s} backend={r.backend} ok={r.ok} {r.wall_time_s:.1f}s failed={failed}")
    if r.stderr.strip(): print("   stderr:", r.stderr[-500:])
    if label == "gold" and not r.ok: rc = 1
    if label == "buggy" and r.ok: rc = 1
print("\nSANDBOX OK" if rc == 0 else "\nSANDBOX NEEDS WORK")
shutil.rmtree(work, ignore_errors=True)
raise SystemExit(rc)
