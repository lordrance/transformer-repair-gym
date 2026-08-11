"""Training history bookkeeping.

Every value is coerced to a Python float: the documented contract is that the history
is JSON-serialisable, and a numpy scalar looks identical until `json.dumps` refuses it.
"""

from __future__ import annotations


def new_history() -> dict:
    return {"loss": [], "lr": [], "grad_norm": []}


def record(history: dict, loss: float, lr: float, grad_norm: float) -> None:
    history["loss"].append(float(loss))
    history["lr"].append(float(lr))
    history["grad_norm"].append(float(grad_norm))


def finalize(history: dict) -> dict:
    finite = all(
        v == v and abs(v) != float("inf") for v in history["loss"] + history["grad_norm"]
    )
    history["all_finite"] = finite
    history["final_loss"] = history["loss"][-1] if history["loss"] else float("nan")
    return history
