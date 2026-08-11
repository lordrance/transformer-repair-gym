"""One-line run summary, used by the CLI."""

from __future__ import annotations

from .._ops.reductions import global_norm
from .curves import loss_slope


def summarize(history: dict) -> dict:
    return {
        "final_loss": float(history.get("final_loss", float("nan"))),
        "slope": loss_slope(history.get("loss", [])),
        "all_finite": bool(history.get("all_finite", False)),
        "grad_norm_scale": global_norm(
            [__import__("torch").tensor(history.get("grad_norm", [0.0]))]
        ),
    }
