"""The visible smoke layer, self-contained so it can be planted in a workspace.

Split out of `trgym/repo/checks.py` during the G1 migration. The planted runner used to
do `from trgym.repo.checks import run_repo_checks`, which meant the candidate's own smoke
test depended on the module that also defines the hidden L2/L3 oracle and `gold_repo()`.
On the host that was invisible because `trgym` happened to be importable; in a v1
`Runtime` it fails outright, and the honest reading is that the dependency should never
have existed. See PROTOCOL_CHANGELOG R9.

Nothing here reads gold. All five visible checks operate purely on the candidate's own
modules, which is why the split is possible at all without weakening anything: this file
contains no ground truth, so planting it in the candidate's workspace reveals nothing.

`trgym/repo/checks.py` imports these definitions rather than restating them, so there is
exactly one definition of each visible check and of `RepoModules`.
"""

from __future__ import annotations

import importlib
import itertools
import math
import shutil
import sys
from pathlib import Path

import torch

_COUNTER = itertools.count()

# The one batch shape the visible suite is allowed to use. The hidden suite
# deliberately never reuses it.
VISIBLE_SHAPE = (2, 16)


class CheckFailure(AssertionError):
    """A graded check failed. The message is surfaced to the trace."""


def _purge_bytecode(root: Path) -> None:
    """Delete `__pycache__` under `root` before importing from it.

    `importlib.invalidate_caches()` is not enough, and the difference is a real grading
    defect rather than a tidiness concern. `invalidate_caches` resets the *finder* caches;
    whether a cached `.pyc` is reused is decided separately, by comparing the source
    mtime-and-size pair recorded in the `.pyc` header.

    A candidate patch that changes `tril(diagonal=1)` to `tril(diagonal=0)` leaves the file
    **exactly the same size**. If it lands in the same mtime second as the write that
    preceded it -- routine on a fast machine, and mtime granularity varies by filesystem --
    Python considers the cached bytecode current and silently runs the OLD code. The repair
    is then graded as if it had never been applied.

    That is a false negative against a correct submission, which is the worst direction for
    this project's error budget: it makes a working fix look broken. It surfaced as
    `test_a_policy_that_fixes_the_bug_passes_the_hidden_suite` failing intermittently on CI
    -- first on windows/py3.11, then on both ubuntu jobs -- while passing locally, which is
    exactly the signature of an mtime-resolution race.
    """
    for cache in root.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)


class RepoModules:
    """Import `tinygpt` from a specific directory under a unique alias."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.alias = f"_trgym_repo_{next(_COUNTER)}"
        self._saved_path = None

    def __enter__(self) -> "RepoModules":
        self._saved_path = list(sys.path)
        sys.path.insert(0, str(self.root))
        for name in [n for n in sys.modules if n == "tinygpt" or n.startswith("tinygpt.")]:
            del sys.modules[name]
        _purge_bytecode(self.root)
        importlib.invalidate_caches()
        try:
            self.pkg = importlib.import_module("tinygpt")
            self.config = importlib.import_module("tinygpt.config")
            self.norm = importlib.import_module("tinygpt.norm")
            self.positional = importlib.import_module("tinygpt.positional")
            self.attention = importlib.import_module("tinygpt.attention")
            self.model = importlib.import_module("tinygpt.model")
            self.data = importlib.import_module("tinygpt.data")
            self.optim = importlib.import_module("tinygpt.optim")
            self.train = importlib.import_module("tinygpt.train")
        except Exception as exc:  # noqa: BLE001 - candidate code is arbitrary
            self.__exit__(None, None, None)
            raise CheckFailure(f"repo does not import: {type(exc).__name__}: {exc}")
        return self

    def __exit__(self, *_exc) -> None:
        for name in [n for n in sys.modules if n == "tinygpt" or n.startswith("tinygpt.")]:
            del sys.modules[name]
        if self._saved_path is not None:
            sys.path[:] = self._saved_path


def _seeded(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# --------------------------------------------------------------------------- #
# The visible suite. One fixed public fixture each -- these are smoke tests the
# buggy code is expected to pass, not a grader.
# --------------------------------------------------------------------------- #
def check_repo_visible_smoke(ws: Path, task_id: str) -> None:
    """It imports, a forward pass runs, the shape is right, nothing is NaN."""
    with RepoModules(ws) as cand:
        cfg = cand.config.Config()
        torch.manual_seed(0)
        model = cand.model.TinyGPT(cfg)
        b, s = VISIBLE_SHAPE
        ids = torch.randint(1, cfg.vocab_size, (b, s), generator=_seeded(1234))
        out = model(ids)
        if tuple(out.shape) != (b, s, cfg.vocab_size):
            raise CheckFailure(f"logits shape {tuple(out.shape)} is wrong")
        if not torch.isfinite(out).all():
            raise CheckFailure("logits contain NaN or inf")


def check_repo_visible_loss_is_finite(ws: Path, task_id: str) -> None:
    with RepoModules(ws) as cand:
        cfg = cand.config.Config()
        torch.manual_seed(0)
        model = cand.model.TinyGPT(cfg)
        batches = cand.data.make_batches(cfg, 1, seed=1234)
        loss_sum, n_tokens = model.loss_sum(
            batches[0].input_ids, batches[0].labels, batches[0].padding_mask
        )
        if not torch.isfinite(loss_sum) or int(n_tokens) <= 0:
            raise CheckFailure("loss is not a finite number over a positive token count")


def check_repo_visible_single_token_attention(ws: Path, task_id: str) -> None:
    with RepoModules(ws) as cand:
        q = torch.randn(1, 1, 1, 16, generator=_seeded(11))
        out = cand.attention.causal_attention(q, q, q)
        if not torch.allclose(out, q, atol=1e-6):
            raise CheckFailure("attention over a single token must return that token's value")


def check_repo_visible_rope_position_zero(ws: Path, task_id: str) -> None:
    """Holds under either pairing convention -- a genuinely lazy smoke test."""
    with RepoModules(ws) as cand:
        cos, sin = cand.positional.build_rope_cache(4, 16, 10000.0)
        if not torch.allclose(cos[0], torch.ones(16), atol=1e-6):
            raise CheckFailure("cos at position 0 must be all ones")
        if not torch.allclose(sin[0], torch.zeros(16), atol=1e-6):
            raise CheckFailure("sin at position 0 must be all zeros")


def check_repo_visible_short_train_runs(ws: Path, task_id: str) -> None:
    """Five steps produce finite numbers. Too short to reveal a divergence."""
    with RepoModules(ws) as cand:
        cfg = cand.config.Config()
        history = cand.train.train(cfg, steps=5, verbose=False)
        if not all(math.isfinite(v) for v in history["loss"]):
            raise CheckFailure(f"loss became non-finite within 5 steps: {history['loss']}")


VISIBLE_CHECKS = {
    "repo_visible_smoke": check_repo_visible_smoke,
    "repo_visible_loss_is_finite": check_repo_visible_loss_is_finite,
    "repo_visible_single_token_attention": check_repo_visible_single_token_attention,
    "repo_visible_rope_position_zero": check_repo_visible_rope_position_zero,
    "repo_visible_short_train_runs": check_repo_visible_short_train_runs,
}


def run_visible_checks(ws: Path, task_id: str, names) -> list[tuple[str, bool, str]]:
    """Run named visible checks. Returns (name, ok, message) per check.

    Raises KeyError for an unknown name rather than skipping it: a typo in a planted
    runner must be loud, not silently reduce the suite.
    """
    out: list[tuple[str, bool, str]] = []
    for name in names:
        fn = VISIBLE_CHECKS[name]
        try:
            fn(Path(ws), task_id)
            out.append((name, True, ""))
        except Exception as exc:  # noqa: BLE001 - a failing check is the measurement
            out.append((name, False, f"{type(exc).__name__}: {exc}"))
    return out
