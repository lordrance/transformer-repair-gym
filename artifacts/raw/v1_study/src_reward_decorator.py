def reward(
    func: F | None = None, weight: float = 1.0, priority: int = 0
) -> F | Callable[[F], F]:
    """Mark a weighted `Task` reward returning a float or keyed scores — per-trace
    judgement over the trace's own run. Cross-agent judgement is an
    `Env`'s `finalize()`, imperatively."""
    decorator = mark("reward", reward_priority=priority, _vf_weight=weight)
    return decorator if func is None else decorator(func)
