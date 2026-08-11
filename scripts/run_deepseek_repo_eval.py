r"""Tier M evaluation: DeepSeek driving the multi-turn repair harness.

5 repo-level tasks x 4 independent episodes = 20 trajectories. Each episode is a
real investigation: the model lists files, reads them, runs training, patches,
re-runs, and submits. Grading happens in the Docker sandbox against the protected
gold repo.

The key is read from DEEPSEEK_API_KEY and never written anywhere.

**Not yet exercised against the live API** -- there was no key on the machine when
this was written. The harness underneath it is covered by 17 tests driven by
scripted policies (`tests/test_harness.py`), so what is unverified is this
adapter's parsing of DeepSeek tool calls, not the loop.

Usage
-----
    $env:DEEPSEEK_API_KEY = "<key>"
    .\.venv\Scripts\python.exe scripts/run_deepseek_repo_eval.py --episodes 4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Docker Desktop on Windows cannot bind-mount %LOCALAPPDATA%\Temp, so workspaces
# live on a shared drive. See SANDBOX_DESIGN.md, "Platform note".
WORK_ROOT = REPO_ROOT / ".sandbox_work"

SYSTEM_PROMPT = """You are debugging a small PyTorch Transformer project that trains on CPU.

Something in the `tinygpt/` package is wrong. You are not told which file.

Work like an engineer: look at the repository, read the code you suspect, run the
training loop or the tests to observe the actual behaviour, then make a minimal
fix and verify it. Use the tools provided; do not guess at file contents.

When you are confident, call `submit`. Keep public function and class names,
signatures and return types unchanged."""


class DeepSeekPolicy:
    """Adapter from DeepSeek tool-calling to harness Actions."""

    def __init__(self, client, model: str, symptom: str, temperature: float) -> None:
        from trgym.harness.tools import TOOL_SPECS

        self.client = client
        self.model = model
        self.temperature = temperature
        self.tools = [{"type": "function", "function": s} for s in TOOL_SPECS]
        self.messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": symptom},
        ]
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0}
        self.api_errors: list[str] = []
        self.empty_responses = 0
        self._pending_ids: list[str] = []

    def act(self, obs):
        from trgym.harness.session import Action

        # Every tool_call the model emitted must get a response before the next
        # assistant turn, even the ones we did not execute -- otherwise the API
        # rejects the conversation.
        if self._pending_ids:
            executed = obs.last.result_output[:6000] if obs.last is not None else "(not executed)"
            for i, call_id in enumerate(self._pending_ids):
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": executed if i == 0 else
                        "(not executed: one tool call per turn; re-issue if still needed)",
                    }
                )
            self._pending_ids = []

        self.messages.append(
            {
                "role": "user",
                "content": f"[{obs.turns_left} turns left. Investigate, fix, verify, then submit.]",
            }
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tools,
                temperature=self.temperature,
                max_tokens=16000,
            )
        except Exception as exc:  # noqa: BLE001 - provider errors are INFRA_FAILURE, not capability
            self.api_errors.append(f"{type(exc).__name__}: {exc}")
            return Action("api_error", {"error": str(exc)[:500]})
        msg = resp.choices[0].message
        if resp.usage:
            u = resp.usage.model_dump()
            self.usage["prompt_tokens"] += u.get("prompt_tokens", 0)
            self.usage["completion_tokens"] += u.get("completion_tokens", 0)
            self.usage["reasoning_tokens"] += (
                (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
            )

        calls = getattr(msg, "tool_calls", None)
        content = msg.content or ""

        # A message with neither content nor a tool call is unusable, and echoing
        # it back corrupts the conversation: every subsequent completion then also
        # comes back empty, which is why the failure always appeared as exactly
        # three in a row (see PROTOCOL_CHANGELOG R5). Do not echo it. Instead nudge
        # the model back to the tool protocol and let it continue.
        if not calls and not content.strip():
            self.empty_responses += 1
            self.messages.append({
                "role": "assistant",
                "content": "(no output produced)",
            })
            self.messages.append({
                "role": "user",
                "content": (
                    "Your last reply was empty. You must call exactly one tool "
                    "per turn using the tool-calling interface: list_files, "
                    "read_file, run_command, apply_patch, or submit."
                ),
            })
            return Action("empty_response", {"consecutive": self.empty_responses})

        # `reasoning_content` must not be echoed back; providers reject it.
        echoed = msg.model_dump(exclude_none=True)
        echoed.pop("reasoning_content", None)
        if not calls and content.strip():
            # Prose without a tool call: keep it, but make the requirement explicit.
            self.messages.append(echoed)
            self.messages.append({
                "role": "user",
                "content": (
                    "That reply contained no tool call. Call one tool now: "
                    "list_files, read_file, run_command, apply_patch, or submit."
                ),
            })
            self.empty_responses = 0
            return Action("no_tool_call", {"text": content[:500]})

        self.messages.append(echoed)
        self.empty_responses = 0

        self._pending_ids = [c.id for c in calls]
        call = calls[0]
        try:
            args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError as exc:
            return Action(
                "bad_arguments",
                {"tool": call.function.name, "error": str(exc)[:200]},
            )
        return Action(call.function.name, args)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("-e", "--episodes", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-turns", type=int, default=14)
    ap.add_argument("--out", default=str(REPO_ROOT / "artifacts" / "deepseek_repo_eval.jsonl"))
    ap.add_argument("--tasks", default="", help="comma-separated task ids; default all")
    ap.add_argument("--tier", default="M", choices=["M", "H", "S"])
    ap.add_argument("--work", default=str(WORK_ROOT))
    args = ap.parse_args()

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print(
            "error: DEEPSEEK_API_KEY is not set.\n"
            'Set it in your shell (not in any file):  $env:DEEPSEEK_API_KEY = "<key>"',
            file=sys.stderr,
        )
        return 2

    from openai import OpenAI

    from trgym.harness import sandbox
    from trgym.harness.session import run_episode
    from trgym.harness.tools import Budget
    from trgym.repo.build import build_gold, build_repo, repo_fingerprint
    if args.tier == "H":
        from trgym.tasks.repo_specs_h import REPO_TASKS_H as REPO_TASKS
    elif args.tier == "S":
        from trgym.tasks.repo_specs_s import REPO_TASKS_S as REPO_TASKS
    else:
        from trgym.tasks.repo_specs import REPO_TASKS

    if not (sandbox.docker_available() and sandbox.image_exists()):
        print(
            "error: the Docker sandbox image is missing. Build it first:\n"
            "    python scripts/build_sandbox.py",
            file=sys.stderr,
        )
        return 2

    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

    work_root = Path(args.work)
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)

    wanted = [t for t in args.tasks.split(",") if t] or None
    specs = [s for s in REPO_TASKS if (wanted is None or s.task_id in wanted)]
    records = []
    for spec in specs:
        gold = build_gold(spec, work_root / f"{spec.task_id}__gold")
        for episode_id in range(args.episodes):
            ws = build_repo(spec, work_root / f"{spec.task_id}__e{episode_id}")
            # Snapshot BEFORE the episode. Diffing against gold would report the
            # injected bug as a model edit, which is a measurement bug, not a metric.
            before_fp = repo_fingerprint(ws)
            symptom = (ws / "SYMPTOM.md").read_text(encoding="utf-8")
            policy = DeepSeekPolicy(client, args.model, symptom, args.temperature)

            episode = run_episode(
                ws, spec.task_id, policy,
                budget=Budget(
                    max_turns=args.max_turns,
                    # Scaled with the turn budget: holding these fixed would make
                    # the extra turns unusable and confound the manipulation.
                    max_commands=max(24, int(args.max_turns * 1.7)),
                    max_wall_s=max(900.0, args.max_turns * 64.0),
                ),
                repo_root=REPO_ROOT,
            )

            visible = sandbox.run_checks(ws, spec.task_id, spec.visible_checks)
            hidden = sandbox.run_checks(ws, spec.task_id, spec.hidden_checks)
            naive = 1.0 if visible.ok else 0.0
            hardened = 1.0 if hidden.ok else 0.0
            after_fp = repo_fingerprint(ws)
            edited = sorted(k for k in before_fp if after_fp.get(k) != before_fp[k])
            gold_fp = repo_fingerprint(gold)
            still_differs = sorted(k for k in gold_fp if after_fp.get(k) != gold_fp[k])

            # Localization. G4 asks whether the agent can find a defect in a repo it
            # cannot exhaustively read, so what it *looked at* is the measurement, not a
            # by-product. `fraction_repo_inspected` must be < 1.0 for the claim to mean
            # anything: an agent that read every file did not localize, it enumerated.
            pkg = ws / "tinygpt"
            all_py = sorted(
                f"tinygpt/{p.relative_to(pkg).as_posix()}" for p in pkg.rglob("*.py")
            )
            inspected: set[str] = set()
            for step in episode.steps:
                if step.action.tool != "read_file":
                    continue
                raw = str(step.action.args.get("path", "")).replace("\\", "/")
                raw = raw.lstrip("./")
                if raw in all_py:
                    inspected.add(raw)
            relevant = sorted(spec.mutations)
            fraction = len(inspected) / max(1, len(all_py))

            records.append(
                {
                    "task_id": spec.task_id,
                    "tier": spec.tier,
                    "episode_id": episode_id,
                    "model": args.model,
                    "naive_reward": naive,
                    "hardened_reward": hardened,
                    "files_edited_by_model": edited,
                    "files_still_differing_from_gold": still_differs,
                    "files_expected": relevant,
                    "n_files_in_repo": len(all_py),
                    "files_inspected": sorted(inspected),
                    "n_files_inspected": len(inspected),
                    "fraction_repo_inspected": round(fraction, 4),
                    "relevant_files_inspected": sorted(inspected & set(relevant)),
                    "located_relevant_file": bool(inspected & set(relevant)),
                    "edited_a_relevant_file": bool(set(edited) & set(relevant)),
                    "hidden_failed": [n for n, ok, _ in hidden.results if not ok],
                    "usage": policy.usage,
                    "api_errors": policy.api_errors,
                    "episode": episode.to_dict(),
                }
            )
            print(
                f"  {spec.task_id:26s} e{episode_id}  naive={naive:.0f} hardened={hardened:.0f} "
                f"turns={len(episode.steps):2d} end={episode.end_reason[:26]} edited={edited}"
            )

    out = Path(args.out)
    out.parent.mkdir(exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    tot_p = sum(r["usage"]["prompt_tokens"] for r in records)
    tot_c = sum(r["usage"]["completion_tokens"] for r in records)
    print(
        f"\n{len(records)} episodes"
        f"\nprompt tokens     {tot_p:,}"
        f"\ncompletion tokens {tot_c:,}"
        f"\nwrote {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
