"""The training loop proper."""

from __future__ import annotations

import torch

from .._core.registry import resolve
from .._core.settings import Config
from .._data.batching import make_batches
from .._model.gpt import TinyGPT
from .._optim.clipping import clip_gradients
from .._optim.schedule import make_scheduler
from .._util.reporting import format_step
from .accumulate import accumulate_gradients
from .history import finalize, new_history, record


def train(cfg: Config, steps: int | None = None, verbose: bool = True) -> dict:
    """Run a short training loop and return its history."""
    torch.manual_seed(cfg.seed)
    steps = steps if steps is not None else cfg.total_steps

    model = TinyGPT(cfg)
    optimizer = resolve("optimizer", "adamw")(model, cfg)
    scheduler = make_scheduler(optimizer, cfg)

    batches = make_batches(cfg, steps * cfg.grad_accum_steps, seed=cfg.seed + 1)
    history = new_history()

    for step in range(steps):
        window = batches[step * cfg.grad_accum_steps : (step + 1) * cfg.grad_accum_steps]
        loss = accumulate_gradients(model, window)
        grad_norm = clip_gradients(model, cfg.grad_clip)

        # optimizer.step() must come before scheduler.step(); the reverse order
        # silently skips the first entry of the schedule.
        optimizer.step()
        scheduler.step()

        record(history, loss, optimizer.param_groups[0]["lr"], grad_norm)
        if verbose:
            print(format_step(step, loss, history["lr"][-1], history["grad_norm"][-1]),
                  flush=True)

    return finalize(history)
