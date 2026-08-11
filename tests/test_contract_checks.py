"""The L1 contract checks must actually fail on a contract violation.

These exist because G9 stage D found that nothing exercised them. `repo_contract_return_types`
and `repo_contract_public_api` were added by verifier v2 to close a confirmed hole
(VERIFIER_V2_PROTOCOL.md H1: v1 accepted a submission that returned a Tensor where the
contract says float). But no Tier E/M/H/S task plants a return-type defect, and no test
called the checks directly — so the checks could have been deleted outright and the whole
suite would still have been green.

That is the same shape as R11's vacuously-green verifier, one level up: not a check that
always passes, but a check that is never asked anything. A mutation that disables it has to
turn something red, and before this file, nothing did.

Everything here runs in-process against a gold tree we build ourselves, so it needs no
Docker and no candidate.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch  # noqa: F401  -- imported so the mutated fixture below can reference it

from trgym.repo.build import build_gold
from trgym.repo.checks import run_repo_checks
from trgym.tasks.repo_specs import get_repo_task

TASK = "m1_attention_regression"


def _gold(tmp_path: Path) -> Path:
    return build_gold(get_repo_task(TASK), tmp_path / "gold")


def _run(ws: Path, name: str) -> tuple[bool, str]:
    results = run_repo_checks(ws, TASK, [name])
    assert len(results) == 1, results
    _, ok, detail = results[0]
    return ok, detail


# --------------------------------------------------------------------------- #
# repo_contract_return_types
# --------------------------------------------------------------------------- #
def test_return_types_pass_on_gold(tmp_path: Path) -> None:
    ok, detail = _run(_gold(tmp_path), "repo_contract_return_types")
    assert ok, f"the reference implementation must satisfy its own contract: {detail}"


def test_return_types_fail_when_accumulate_returns_a_tensor(tmp_path: Path) -> None:
    """The exact defect verifier v2 was built for: a Tensor where float is documented.

    `numpy.float64` and `torch.Tensor` both survive `isinstance(x, float)`-style checks in
    the first case and break `json.dumps` downstream in both, which is why the contract is
    an exact type and why this check has to exist.
    """
    ws = _gold(tmp_path)
    train = ws / "tinygpt" / "train.py"
    src = train.read_text(encoding="utf-8")
    needle = "    return total_loss / total_tokens"
    assert src.count(needle) == 1, "template changed; retarget this fixture"
    train.write_text(
        src.replace(needle, "    return torch.tensor(total_loss / total_tokens)"),
        encoding="utf-8",
    )

    ok, detail = _run(ws, "repo_contract_return_types")
    assert not ok, "returning a Tensor from accumulate_gradients must fail the contract"
    assert "accumulate_gradients" in detail, detail


def test_return_types_fail_when_history_is_not_json_serialisable(tmp_path: Path) -> None:
    """The documented history must survive serialisation."""
    ws = _gold(tmp_path)
    train = ws / "tinygpt" / "train.py"
    src = train.read_text(encoding="utf-8")
    needle = '    history["loss"].append(loss)'
    assert src.count(needle) == 1, "template changed; retarget this fixture"
    train.write_text(
        src.replace(needle, '    history["loss"].append(complex(loss, 0))'),
        encoding="utf-8",
    )

    ok, _ = _run(ws, "repo_contract_return_types")
    assert not ok, "a non-float in train() history must fail the contract"


# --------------------------------------------------------------------------- #
# repo_contract_public_api
# --------------------------------------------------------------------------- #
def test_public_api_passes_on_gold(tmp_path: Path) -> None:
    ok, detail = _run(_gold(tmp_path), "repo_contract_public_api")
    assert ok, f"gold must match its own public API: {detail}"


def test_public_api_fails_when_a_signature_changes(tmp_path: Path) -> None:
    ws = _gold(tmp_path)
    optim = ws / "tinygpt" / "optim.py"
    src = optim.read_text(encoding="utf-8")
    needle = "def make_optimizer(model: torch.nn.Module, cfg: Config)"
    assert src.count(needle) == 1, "template changed; retarget this fixture"
    optim.write_text(
        src.replace(needle, "def make_optimizer(model: torch.nn.Module, cfg: Config, lr=None)"),
        encoding="utf-8",
    )

    ok, detail = _run(ws, "repo_contract_public_api")
    assert not ok, "an added parameter changes the public signature and must be caught"
    assert "make_optimizer" in detail, detail


def test_public_api_fails_when_a_public_symbol_is_removed(tmp_path: Path) -> None:
    ws = _gold(tmp_path)
    optim = ws / "tinygpt" / "optim.py"
    src = optim.read_text(encoding="utf-8")
    marker = "def make_scheduler("
    assert marker in src
    optim.write_text(src[: src.index(marker)] + "\n", encoding="utf-8")

    ok, detail = _run(ws, "repo_contract_public_api")
    assert not ok, "removing a documented public function must be caught"
    assert "make_scheduler" in detail, detail


@pytest.mark.parametrize("check", ["repo_contract_return_types", "repo_contract_public_api"])
def test_check_is_registered_on_both_surfaces(check: str) -> None:
    """Guards against the check being quietly dropped rather than quietly disabled."""
    from trgym.repo import predicates
    from trgym.repo.checks import CHECKS, LEVELS

    assert check in LEVELS
    assert check in CHECKS
    assert check in predicates.PREDICATES
