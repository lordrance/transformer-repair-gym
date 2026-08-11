"""Training loop with gradient accumulation.

Runnable directly:

    python -m tinygpt.train --steps 40
"""

from __future__ import annotations

import argparse

from ._core.settings import Config
from ._core.types import Batch
from ._io.serialize import history_to_json
from ._metrics.summary import summarize


def accumulate_gradients(model, micro_batches: list[Batch]) -> float:
    """Accumulate over micro-batches; returns the mean loss per supervised token."""
    from ._train.accumulate import accumulate_gradients as _impl

    return _impl(model, micro_batches)


def train(cfg: Config, steps: int | None = None, verbose: bool = True) -> dict:
    """Run a short training loop and return its history."""
    from ._train.loop import train as _impl

    return _impl(cfg, steps, verbose)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--json", action="store_true", help="print the history as JSON")
    args = ap.parse_args()

    cfg = Config()
    history = train(cfg, steps=args.steps, verbose=not args.json)
    if args.json:
        print(history_to_json(cfg, history))
    else:
        stats = summarize(history)
        print(
            f"\nfinal loss {stats['final_loss']:.4f}  all_finite={stats['all_finite']}"
        )


if __name__ == "__main__":
    main()
