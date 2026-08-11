"""JSON serialization of a run."""

from __future__ import annotations

import json
from dataclasses import asdict

from .._core.settings import Config


def history_to_json(cfg: Config, history: dict) -> str:
    return json.dumps({"config": asdict(cfg), "history": history})
