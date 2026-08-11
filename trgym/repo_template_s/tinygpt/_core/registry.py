"""A tiny name -> factory registry.

Used by the CLI to select a component without importing every implementation at
start-up. Small, but real: `_train.loop` resolves the optimizer factory through it.
"""

from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")

_REGISTRY: dict[str, dict[str, Callable]] = {}


def register(kind: str, name: str) -> Callable[[T], T]:
    def deco(fn: T) -> T:
        _REGISTRY.setdefault(kind, {})[name] = fn
        return fn

    return deco


def resolve(kind: str, name: str) -> Callable:
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        raise KeyError(f"no {kind!r} registered under {name!r}")


def available(kind: str) -> list[str]:
    return sorted(_REGISTRY.get(kind, {}))
