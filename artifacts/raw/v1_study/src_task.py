class Task(Generic[DataT, StateT, ConfigT]):
    NEEDS_CONTAINER: ClassVar[bool] = False
    """Whether the task needs a containerized environment (isolated filesystem, ...)."""

    def __init__(self, data: DataT, config: ConfigT | None = None) -> None:
        self.data = data
        self.config = config if config is not None else self.config_type()()

    def with_system_prompt(self, system_prompt: str) -> Self:
        clone = copy.copy(self)
        clone.data = self.data.model_copy(update={"system_prompt": system_prompt})
        return clone

    async def setup(self, trace: Trace, runtime: Runtime) -> None:
        return None

    async def finalize(self, trace: Trace, runtime: Runtime) -> None:
        return None

    async def validate(self, runtime: Runtime) -> bool:
        return True

    async def score(
        self,
        trace: Trace,
        runtime: Runtime | None = None,
    ) -> None:
        def requires_runtime(fn) -> bool:
            param = inspect.signature(fn).parameters.get("runtime")
            # A defaulted runtime parameter can still be called offline with None.
            return param is not None and param.default is inspect.Parameter.empty

        judges = self.plugged_judges()
        available = {"task": self.data, "trace": trace}
        if runtime is not None:
            available["runtime"] = runtime

        async with boundary(TaskError, f"task {type(self).__name__} scoring"):
            metrics = discover_decorated(self, "metric")
            rewards = discover_decorated(self, "reward")
            if runtime is None:
                skipped = [
                    fn.__name__ for fn in (*metrics, *rewards) if requires_runtime(fn)
                ] + [
                    judge.reward_name
                    for judge in judges
                    if requires_runtime(judge.score)
                ]
                if skipped:
                    logger.info(
                        "score: no runtime — skipped runtime-dependent signals: %s",
                        skipped,
                    )
                metrics = [fn for fn in metrics if not requires_runtime(fn)]
                rewards = [fn for fn in rewards if not requires_runtime(fn)]
                judges = [
                    judge for judge in judges if not requires_runtime(judge.score)
                ]

            seed(trace.metrics, (fn.__name__ for fn in metrics))
            seed(trace.rewards, (fn.__name__ for fn in rewards))
            seed(trace.rewards, (judge.reward_name for judge in judges))

            metric_results = await invoke_all(metrics, available)
            for fn, result in zip(metrics, metric_results):
                if isinstance(result, Mapping):
                    unseed(trace.metrics, fn.__name__)
                    trace.record_metrics(result)
                else:
                    trace.record_metric(fn.__name__, result)
            reward_results = await invoke_all(rewards, available)
            for fn, result in zip(rewards, reward_results):
                weight = getattr(fn, "_vf_weight", 1.0)
                if isinstance(result, Mapping):
                    unseed(trace.rewards, fn.__name__)
                    items = result.items()
                else:
                    items = [(fn.__name__, result)]
                for key, value in items:
                    trace.record_reward(key, value, weight)
            judge_results = await invoke_all(
                [judge.score for judge in judges], available
            )
            for judge, result in zip(judges, judge_results):
                if isinstance(result, Mapping):
                    unseed(trace.rewards, judge.reward_name)
                    items = result.items()
                else:
                    items = [(judge.reward_name, result)]
                for key, value in items:
                    trace.record_reward(key, value, judge.config.weight)

    @classmethod
    def data_type(cls) -> type[TaskData]:
        return concrete_type(cls, TaskData) or TaskData

    @classmethod
    def config_type(cls) -> type[TaskConfig]:
        return concrete_type(cls, TaskConfig) or TaskConfig

    def plugged_judges(self) -> list[Judge]:
        from verifiers.v1.utils.loaders import load_judge

        return [load_judge(config) for config in self.config.judges]

    @classmethod
    def toolsets(cls, config: ConfigT) -> list[Toolset]:
        """Tool servers launched per rollout, each constructed with its config off
        `config` — override and wire explicitly:

            @classmethod
            def toolsets(cls, config: MyTaskConfig) -> list[vf.Toolset]:
                return [SearchToolset(config.tools)]

        A classmethod so consumers can size placement (tunnels) off the class
        before any task instance exists."""
        return []
