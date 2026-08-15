"""v0.2-A: replay verifier v1 against v2 on cases built to separate them.

The problem this answers. G2's ordinary replay reports `v1_v2_disagreements == 0` across 89
real trajectories: v2 never once decided differently from v1. That is a true statement about
*agent behaviour*, not about the verifier — v2 = v1 plus two L1 contract checks
(`repo_contract_return_types`, `repo_contract_public_api`), and ordinary agents simply never
produce a semantically-correct tree that violates the documented interface. G9 stage D found
the same hole from the other side: nothing exercised those two checks at all.

So this suite constructs the missing population. Each adversarial case starts from **gold**
and applies one edit that is semantically harmless but breaks the contract:
`accumulate_gradients` returning a Tensor instead of `float`, `loss_sum` returning a list
instead of a 2-tuple, a public signature gaining a parameter, and so on. If v2 is doing
anything, v1 accepts these and v2 rejects them.

Two controls are included and both matter:
  * `control_gold` -- unmodified gold; both verifiers must ACCEPT, or the harness is broken.
  * `control_semantic_defect` -- a real causal-mask bug; both must REJECT, showing v1 is
    not simply blind.

These cases were written from the contract's documented types before the suite was first
run, and are not tuned afterwards. Cases where v1 also rejects are reported as
NOT-DISTINGUISHING rather than removed, because "the edit was not as harmless as intended"
is a result, not a defect to hide.

Run: python scripts/verifier_adversarial_replay.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "artifacts" / "verifier_adversarial_replay.json"
TASK = "m1_attention_regression"


@dataclass
class Case:
    name: str
    category: str
    rel: str
    find: str
    replace: str
    intent: str
    expect_v1_accept: bool = True
    expect_v2_accept: bool = False
    edits: list = field(default_factory=list)


NUMPY_IMPORT = "import numpy as _np; "

CASES: list[Case] = [
    # ---------------------------------------------------------------- return types
    Case(
        "accum_returns_tensor", "return_type", "tinygpt/train.py",
        "    return total_loss / total_tokens",
        "    return torch.tensor(total_loss / total_tokens)",
        "accumulate_gradients returns Tensor; contract documents exactly float. "
        "This is verbatim the v1 hole VERIFIER_V2_PROTOCOL.md H1 was written for.",
    ),
    Case(
        "accum_returns_numpy_scalar", "return_type", "tinygpt/train.py",
        "    return total_loss / total_tokens",
        f"    {NUMPY_IMPORT}return _np.float64(total_loss / total_tokens)",
        "numpy.float64 subclasses float, so isinstance() accepts it and json.dumps "
        "later refuses it. Only an exact-type check catches this.",
    ),
    Case(
        "history_loss_numpy", "return_type", "tinygpt/train.py",
        '        history["loss"].append(loss)',
        f'        {NUMPY_IMPORT}history["loss"].append(_np.float64(loss))',
        "train() history must be a list of Python float; numpy scalars break "
        "serialisation downstream.",
    ),
    Case(
        "final_loss_numpy", "return_type", "tinygpt/train.py",
        '    history["final_loss"] = history["loss"][-1] if history["loss"] else float("nan")',
        f'    {NUMPY_IMPORT}history["final_loss"] = _np.float64('
        'history["loss"][-1] if history["loss"] else float("nan"))',
        "final_loss is documented as a Python float.",
    ),
    Case(
        "loss_sum_returns_list", "return_type", "tinygpt/model.py",
        "    return loss_sum, n_tokens",
        "    return [loss_sum, n_tokens]",
        "A list unpacks exactly like a 2-tuple at every call site, so nothing "
        "semantic notices. The contract says tuple.",
    ),
    Case(
        "n_supervised_returns_tensor", "return_type", "tinygpt/data.py",
        "        return int((self.labels[:, 1:] != -100).sum())",
        "        return (self.labels[:, 1:] != -100).sum()",
        "Batch.n_supervised documented as int; a 0-d Tensor behaves like one "
        "everywhere except the interface.",
    ),
    Case(
        "scheduler_missing_get_last_lr", "return_type", "tinygpt/optim.py",
        "    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)",
        "    class _Sched:\n"
        "        def __init__(self, inner): self._inner = inner\n"
        "        def step(self, *a, **k): return self._inner.step(*a, **k)\n"
        "    return _Sched(torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda))",
        "A scheduler shim that steps correctly but drops get_last_lr(). Training "
        "runs fine; the documented surface is incomplete.",
    ),
    # ---------------------------------------------------------------- public API
    Case(
        "public_api_extra_param", "public_api", "tinygpt/optim.py",
        "def make_optimizer(model: torch.nn.Module, cfg: Config) -> torch.optim.Optimizer:",
        "def make_optimizer(model: torch.nn.Module, cfg: Config, lr: float | None = None"
        ") -> torch.optim.Optimizer:",
        "A defaulted extra parameter is source-compatible for every caller and "
        "still changes the public signature.",
    ),
    Case(
        "public_api_objective_signature", "public_api", "tinygpt/model.py",
        "def shifted_cross_entropy_sum(\n    logits: torch.Tensor, labels: torch.Tensor, "
        "ignore_index: int\n) -> tuple[torch.Tensor, torch.Tensor]:",
        "def shifted_cross_entropy_sum(\n    logits: torch.Tensor, labels: torch.Tensor, "
        "ignore_index: int, reduction: str = 'sum'\n) -> tuple[torch.Tensor, torch.Tensor]:",
        "Same trick on the objective's public signature.",
    ),
    Case(
        "public_api_norm_default", "public_api", "tinygpt/norm.py",
        "def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:",
        "def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6"
        ") -> torch.Tensor:",
        "Adding a default value changes the signature without changing behaviour.",
    ),
    Case(
        "public_api_method_signature", "public_api", "tinygpt/model.py",
        "    def loss_sum(\n        self,\n        input_ids: torch.Tensor,",
        "    def loss_sum(\n        self,\n        *,\n        input_ids: torch.Tensor,",
        "Making a documented positional parameter keyword-only.",
        expect_v1_accept=False,   # callers pass positionally; predicted to break v1 too
    ),
    # ---------------------------------------------------------------- controls
    Case(
        "control_gold", "control", "tinygpt/train.py",
        "    return total_loss / total_tokens",
        "    return total_loss / total_tokens",
        "Unmodified gold. Both verifiers must ACCEPT or the harness is broken.",
        expect_v1_accept=True, expect_v2_accept=True,
    ),
    Case(
        "control_semantic_defect", "control", "tinygpt/attention.py",
        "device=scores.device).tril(diagonal=0)",
        "device=scores.device).tril(diagonal=1)",
        "A real causal-mask defect. Both must REJECT: v1 is not blind, it is "
        "narrow.",
        expect_v1_accept=False, expect_v2_accept=False,
    ),
]


def evaluate(case: Case) -> dict:
    from trgym.repo.build import build_gold
    from trgym.repo.checks import run_repo_checks
    from trgym.repo.verifier_v2 import v1_checks, v2_checks
    from trgym.tasks.repo_specs import get_repo_task

    spec = get_repo_task(TASK)
    work = Path(tempfile.mkdtemp(prefix=f"advreplay_{case.name}_"))
    try:
        ws = build_gold(spec, work / "tree")
        target = ws / case.rel
        src = target.read_text(encoding="utf-8")
        n = src.count(case.find)
        if n != 1:
            return {"case": case.name, "category": case.category, "error":
                    f"anchor found {n}x in {case.rel}, expected 1", "applied": False}
        if case.find != case.replace:
            target.write_text(src.replace(case.find, case.replace), encoding="utf-8")

        v1 = run_repo_checks(ws, TASK, list(v1_checks(spec)))
        v2 = run_repo_checks(ws, TASK, list(v2_checks(spec)))
        v1_accept = all(ok for _, ok, _ in v1)
        v2_accept = all(ok for _, ok, _ in v2)
        v2_failed = [n for n, ok, _ in v2 if not ok]
        return {
            "case": case.name,
            "category": case.category,
            "intent": case.intent,
            "file": case.rel,
            "applied": True,
            "v1_accept": v1_accept,
            "v2_accept": v2_accept,
            "distinguishing": bool(v1_accept and not v2_accept),
            "v1_failed_checks": [n for n, ok, _ in v1 if not ok],
            "v2_failed_checks": v2_failed,
            "v2_rejected_by": [c for c in v2_failed if c.startswith("repo_contract_")],
            "expected_v1_accept": case.expect_v1_accept,
            "expected_v2_accept": case.expect_v2_accept,
            "matched_expectation": (v1_accept == case.expect_v1_accept
                                    and v2_accept == case.expect_v2_accept),
            "detail": next((d for n, ok, d in v2 if not ok), "")[:220],
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main() -> int:
    rows = [evaluate(c) for c in CASES]

    adversarial = [r for r in rows if r.get("category") != "control"]
    controls = [r for r in rows if r.get("category") == "control"]
    distinguishing = [r for r in adversarial if r.get("distinguishing")]
    not_distinguishing = [r for r in adversarial if r.get("applied")
                          and not r.get("distinguishing")]

    ordinary = {}
    prev = ROOT / "artifacts" / "verifier_v2_replay.json"
    if prev.exists():
        s = (json.loads(prev.read_text(encoding="utf-8")).get("summary") or {})
        ordinary = {
            "population": s.get("population"),
            "replayed": s.get("replayed"),
            "v1_v2_disagreements": s.get("v1_v2_disagreements"),
            "v1_FP_rate": s.get("v1_FP_rate"),
            "v2_FP_rate": s.get("v2_FP_rate"),
        }

    by_cat: dict = {}
    for r in adversarial:
        c = r["category"]
        by_cat.setdefault(c, {"n": 0, "v1_accept": 0, "v2_reject": 0})
        by_cat[c]["n"] += 1
        by_cat[c]["v1_accept"] += 1 if r.get("v1_accept") else 0
        by_cat[c]["v2_reject"] += 1 if r.get("v2_accept") is False else 0

    payload = {
        "task": TASK,
        "ordinary_replay": ordinary,
        "adversarial": {
            "n_cases": len(adversarial),
            "v1_accepted": sum(1 for r in adversarial if r.get("v1_accept")),
            "v2_rejected": sum(1 for r in adversarial if r.get("v2_accept") is False),
            "distinguishing": len(distinguishing),
            "not_distinguishing": [r["case"] for r in not_distinguishing],
            "by_category": by_cat,
        },
        "controls": {
            r["case"]: {"v1_accept": r.get("v1_accept"), "v2_accept": r.get("v2_accept"),
                        "as_expected": r.get("matched_expectation")}
            for r in controls
        },
        "all_expectations_met": all(r.get("matched_expectation") for r in rows
                                    if r.get("applied")),
        "cases": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"ordinary replay disagreements : {ordinary.get('v1_v2_disagreements')}")
    print(f"adversarial cases             : {len(adversarial)}")
    print(f"  v1 accepted                 : {payload['adversarial']['v1_accepted']}")
    print(f"  v2 rejected                 : {payload['adversarial']['v2_rejected']}")
    print(f"  v1-accept AND v2-reject     : {len(distinguishing)}")
    print()
    for r in rows:
        if not r.get("applied"):
            print(f"  !! {r['case']:<34} {r.get('error')}")
            continue
        mark = "DISTINGUISHING" if r.get("distinguishing") else (
            "control" if r["category"] == "control" else "not-distinguishing")
        flag = " " if r.get("matched_expectation") else "*"
        print(f" {flag}{r['case']:<34} v1={'A' if r['v1_accept'] else 'R'} "
              f"v2={'A' if r['v2_accept'] else 'R'}  {mark}")
        if r.get("v2_rejected_by"):
            print(f"    rejected by: {r['v2_rejected_by']}")
    print("\n* = outcome differed from the pre-registered expectation")
    print(f"wrote {OUT.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
