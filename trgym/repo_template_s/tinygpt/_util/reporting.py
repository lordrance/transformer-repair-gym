"""Console formatting for the training loop."""

from __future__ import annotations


def format_step(step: int, loss: float, lr: float, grad_norm: float) -> str:
    return f"step {step:3d}  loss {loss:.4f}  lr {lr:.2e}  grad_norm {grad_norm:.3f}"
