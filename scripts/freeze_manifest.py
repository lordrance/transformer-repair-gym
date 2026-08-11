"""Freeze the experiment configuration before any real-model run.

The point is to make "we changed the task after seeing the results" detectable
rather than a matter of trust. Every file that can influence a score is hashed:
task definitions, prompts, harness, both verifiers, the oracle, the sandbox.

Re-running this after an evaluation and diffing the manifest shows exactly what
moved.

Usage:  python scripts/freeze_manifest.py [--label primary]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Everything whose content can change a reward. Ordered by layer.
TRACKED = [
    # task definitions
    "trgym/tasks/spec.py",
    "trgym/tasks/registry.py",
    "trgym/tasks/repo_specs.py",
    # the pristine reference the tasks are generated from
    "trgym/reference/tiny_gpt.py",
    "trgym/reference/train_loop.py",
    "trgym/repo_template/tinygpt/config.py",
    "trgym/repo_template/tinygpt/norm.py",
    "trgym/repo_template/tinygpt/positional.py",
    "trgym/repo_template/tinygpt/attention.py",
    "trgym/repo_template/tinygpt/model.py",
    "trgym/repo_template/tinygpt/data.py",
    "trgym/repo_template/tinygpt/optim.py",
    "trgym/repo_template/tinygpt/train.py",
    # workspace construction and prompts
    "trgym/tasks/build.py",
    "trgym/repo/build.py",
    # verifiers
    "trgym/verifier/visible.py",
    "trgym/verifier/hidden.py",
    "trgym/verifier/anticheat.py",
    "trgym/verifier/reward.py",
    "trgym/repo/checks.py",
    # harness and sandbox
    "trgym/harness/tools.py",
    "trgym/harness/session.py",
    "trgym/harness/sandbox.py",
    "trgym/patching.py",
    "docker/Dockerfile",
    # the evaluation driver, which contains the system prompt
    "scripts/run_deepseek_repo_eval.py",
    # exploit catalogue
    "trgym/cheats/catalog.py",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_state() -> dict:
    def run(*args: str) -> str | None:
        try:
            proc = subprocess.run(
                ["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=20
            )
            return proc.stdout.strip() if proc.returncode == 0 else None
        except Exception:  # noqa: BLE001
            return None

    head = run("rev-parse", "HEAD")
    if head is None:
        return {"repo": "not a git repository", "note": "hashes are the only provenance"}
    return {
        "commit": head,
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="primary")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from trgym.harness import sandbox
    from trgym.harness.tools import ALLOWED_COMMANDS, Budget
    from trgym.repo.checks import CHECKS as REPO_CHECKS
    from trgym.repo.checks import LEVELS as REPO_LEVELS
    from trgym.tasks.registry import TASKS
    from trgym.tasks.repo_specs import REPO_TASKS
    from trgym.verifier.anticheat import GATES

    import scripts.run_deepseek_repo_eval as driver  # noqa: PLC0415

    files = {}
    missing = []
    for rel in TRACKED:
        path = REPO_ROOT / rel
        if path.exists():
            files[rel] = sha256(path)
        else:
            missing.append(rel)

    budget = Budget()
    manifest = {
        "label": args.label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "python": sys.version.split()[0],
        "torch": __import__("torch").__version__,
        "verifiers_pin": "0.3.0",
        "file_hashes": files,
        "missing_files": missing,
        "tasks": {
            "tier_E": [t.task_id for t in TASKS],
            "tier_M": [
                {
                    "task_id": t.task_id,
                    "family_id": t.family_id,
                    "provenance": t.provenance,
                    "files_mutated": sorted(t.mutations),
                    "requires_training_run": t.requires_training_run,
                    "visible_checks": list(t.visible_checks),
                    "hidden_checks": list(t.hidden_checks),
                }
                for t in REPO_TASKS
            ],
        },
        "verifier": {
            "repo_checks": sorted(REPO_CHECKS),
            "repo_check_levels": REPO_LEVELS,
            "anticheat_gates": [g.__name__ for g in GATES],
        },
        "harness": {
            "allowed_commands": sorted(ALLOWED_COMMANDS),
            "max_turns": budget.max_turns,
            "max_commands": budget.max_commands,
            "max_wall_s": budget.max_wall_s,
            "max_command_s": budget.max_command_s,
        },
        "sandbox": {
            "image": sandbox.IMAGE,
            "image_present": sandbox.image_exists(),
            "memory": sandbox.DEFAULT_MEMORY,
            "cpus": sandbox.DEFAULT_CPUS,
            "pids_limit": sandbox.DEFAULT_PIDS,
            "network": "none",
            "read_only_rootfs": True,
        },
        "model_config": {
            "provider": "deepseek",
            "base_url": "https://api.deepseek.com",
            "system_prompt_sha256": hashlib.sha256(
                driver.SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest(),
            "system_prompt": driver.SYSTEM_PROMPT,
            "temperature": 1.0,
            "max_tokens": 16000,
            "note": "model id recorded per-trajectory in the JSONL",
        },
    }

    out = Path(args.out) if args.out else REPO_ROOT / "EXPERIMENT_MANIFEST.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"froze {len(files)} files ({len(missing)} missing) -> {out}")
    if missing:
        print("missing:", missing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
