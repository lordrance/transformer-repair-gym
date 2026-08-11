def metric(func: F | None = None, priority: int = 0) -> F | Callable[[F], F]:
    """Mark a `Task`/`Harness` metric `(self, trace) -> float` (recorded, not
    summed) — per-trace judgement; it declares what it needs by name (`task`,
    `trace`, `runtime`). Cross-agent judgement is an `Env`'s `finalize()`,
    imperatively."""
    decorator = mark("metric", metric_priority=priority)
    return decorator if func is None else decorator(func)
