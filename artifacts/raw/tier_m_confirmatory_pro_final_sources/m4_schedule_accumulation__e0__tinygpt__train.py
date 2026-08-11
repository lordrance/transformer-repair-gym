"""Training loop with gradient accumulation.

Runnable directly, which is how the multi-turn harness lets a candidate observe
training dynamics:

    python -m tinygpt.train --steps 40
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch

from .config import Config
from .data import Batch, make_batches
from .model import TinyGPT
from .optim import make_optimizer, make_scheduler


def accumulate_gradients(model: TinyGPT, micro_batches: list[Batch]) -> float:
    """Accumulate over micro-batches; returns the mean loss per supervised token.

    Two passes. The first only counts supervised tokens so the denominator is
    fixed before any backward call; the second does the backward with that
    global denominator. Normalizing per micro-batch instead would make the
    result depend on how the data happened to be split.
    """
    model.zero_grad(set_to_none=True)

    total_tokens = sum(mb.n_supervised for mb in micro_batches)
    if total_tokens == 0:
        return 0.0

    total_loss = 0.0
    for mb in micro_batches:
        loss_sum, _ = model.loss_sum(mb.input_ids, mb.labels, mb.padding_mask)
        (loss_sum / total_tokens).backward()
        total_loss += float(loss_sum.detach())
    return total_loss / total_tokens


def train(cfg: Config, steps: int | None = None, verbose: bool = True) -> dict:
    """Run a short training loop and return its history."""
    torch.manual_seed(cfg.seed)
    steps = steps if steps is not None else cfg.total_steps

    model = TinyGPT(cfg)
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg)

    batches = make_batches(cfg, steps * cfg.grad_accum_steps, seed=cfg.seed + 1)
    history = {"loss": [], "lr": [], "grad_norm": []}

    for step in range(steps):
        window = batches[step * cfg.grad_accum_steps : (step + 1) * cfg.grad_accum_steps]
        loss = accumulate_gradients(model, window)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()
        scheduler.step()

        history["loss"].append(loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["grad_norm"].append(float(grad_norm))

        if verbose:
            print(
                f"step {step:3d}  loss {loss:.4f}  lr {history['lr'][-1]:.2e}  "
                f"grad_norm {history['grad_norm'][-1]:.3f}",
                flush=True,
            )

    finite = all(
        v == v and abs(v) != float("inf") for v in history["loss"] + history["grad_norm"]
    )
    history["all_finite"] = finite
    history["final_loss"] = history["loss"][-1] if history["loss"] else float("nan")
    return history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--json", action="store_true", help="print the history as JSON")
    args = ap.parse_args()

    cfg = Config()
    history = train(cfg, steps=args.steps, verbose=not args.json)
    if args.json:
        print(json.dumps({"config": asdict(cfg), "history": history}))
    else:
        print(
            f"\nfinal loss {history['final_loss']:.4f}  all_finite={history['all_finite']}"
        )


if __name__ == "__main__":
    main()
