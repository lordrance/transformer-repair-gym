"""Diagnostics computed from a finished run."""

from .curves import loss_slope
from .summary import summarize

__all__ = ["loss_slope", "summarize"]
