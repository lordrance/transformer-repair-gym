"""The multi-turn repair session.

    prompt -> inspect -> execute -> observe -> edit -> retest -> submit

A `Policy` is anything with `act(observation) -> Action`. That keeps the loop
independent of who is driving it: a real model over an OpenAI-compatible API, or
a scripted policy in the tests. The scripted policy is how the harness is
verified without spending a token, and it is the reason the loop can be trusted
before any API key exists.

The episode ends when the policy submits, when a budget runs out, or when the
policy emits an unparseable action too many times in a row. Grading always runs
on whatever state the workspace is in at that point -- there is no "no submission"
special case, because a policy that burns its turns and leaves the bug in place
should score exactly what that is worth.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from trgym.harness.tools import (
    Budget,
    ToolResult,
    apply_patch,
    list_files,
    read_file,
    run_command,
)

MAX_CONSECUTIVE_PARSE_FAILURES = 3

# Actions a policy adapter emits when it could not produce a tool call. These are
# INFRA/PROTOCOL events, not the policy choosing an invalid tool, so they must not
# count toward the unusable-action limit -- doing so turned a provider hiccup into
# a capability verdict on h2 (PROTOCOL_CHANGELOG R5).
ADAPTER_NONACTIONS = frozenset({"api_error", "empty_response", "no_tool_call", "noop"})


@dataclass
class Action:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class Step:
    turn: int
    action: Action
    result_ok: bool
    result_output: str
    duration_s: float

    def to_dict(self) -> dict:
        return {
            "turn": self.turn,
            "tool": self.action.tool,
            "args": {
                k: (v if len(str(v)) < 2000 else str(v)[:2000] + "...[clipped]")
                for k, v in self.action.args.items()
            },
            "ok": self.result_ok,
            "output": self.result_output,
            "duration_s": round(self.duration_s, 3),
        }


@dataclass
class Observation:
    """What the policy sees before choosing its next action."""

    symptom: str
    turns_left: int
    last: Step | None
    history: list[Step]


class Policy(Protocol):
    def act(self, obs: Observation) -> Action: ...


@dataclass
class Episode:
    task_id: str
    steps: list[Step]
    end_reason: str
    submitted: bool
    summary: str
    budget: dict

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "end_reason": self.end_reason,
            "submitted": self.submitted,
            "summary": self.summary,
            "n_turns": len(self.steps),
            "budget": self.budget,
            "steps": [s.to_dict() for s in self.steps],
        }


def run_episode(
    workspace: Path,
    task_id: str,
    policy: Policy,
    *,
    budget: Budget | None = None,
    repo_root: Path | None = None,
) -> Episode:
    workspace = Path(workspace)
    budget = budget or Budget()
    symptom = (workspace / "SYMPTOM.md").read_text(encoding="utf-8")

    steps: list[Step] = []
    parse_failures = 0
    end_reason = "budget"
    submitted = False
    summary = ""

    while True:
        exhausted = budget.exhausted()
        if exhausted:
            end_reason = exhausted
            break

        obs = Observation(
            symptom=symptom,
            turns_left=budget.max_turns - budget.turns_used,
            last=steps[-1] if steps else None,
            history=steps,
        )
        budget.turns_used += 1

        try:
            action = policy.act(obs)
        except Exception as exc:  # noqa: BLE001 - a broken policy is data
            end_reason = f"policy raised {type(exc).__name__}: {exc}"
            break

        if action.tool == "submit":
            submitted = True
            summary = str(action.args.get("summary", ""))
            end_reason = "submitted"
            steps.append(
                Step(budget.turns_used, action, True, "episode submitted", 0.0)
            )
            break

        if action.tool in ADAPTER_NONACTIONS:
            # Recorded in the trace, but it does not consume the unusable-action
            # allowance and it does not end the episode.
            steps.append(
                Step(budget.turns_used, action, False,
                     f"adapter non-action: {action.tool}", 0.0)
            )
            continue

        result = _dispatch(workspace, action, budget, repo_root)
        if result is None:
            parse_failures += 1
            result = ToolResult(
                ok=False,
                output=f"unknown tool {action.tool!r}; nothing was executed",
            )
            if parse_failures >= MAX_CONSECUTIVE_PARSE_FAILURES:
                steps.append(
                    Step(budget.turns_used, action, False, result.output, 0.0)
                )
                end_reason = "too many unusable actions"
                break
        else:
            parse_failures = 0

        steps.append(
            Step(
                turn=budget.turns_used,
                action=action,
                result_ok=result.ok,
                result_output=result.render(),
                duration_s=result.duration_s,
            )
        )

    return Episode(
        task_id=task_id,
        steps=steps,
        end_reason=end_reason,
        submitted=submitted,
        summary=summary,
        budget=asdict(budget),
    )


def _dispatch(
    workspace: Path, action: Action, budget: Budget, repo_root: Path | None
) -> ToolResult | None:
    args = action.args
    if action.tool == "list_files":
        return list_files(workspace, args.get("pattern", "**/*.py"))
    if action.tool == "read_file":
        return read_file(
            workspace, args["path"], int(args.get("start", 1)),
            int(args["end"]) if args.get("end") is not None else None,
        )
    if action.tool == "run_command":
        return run_command(workspace, args.get("name", ""), budget, repo_root)
    if action.tool == "apply_patch":
        return apply_patch(workspace, args.get("path", ""), args.get("diff", ""))
    return None


def save_episode(episode: Episode, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(episode.to_dict(), indent=2), encoding="utf-8")
