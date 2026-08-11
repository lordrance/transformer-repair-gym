class Trace(BaseModel, Generic[DataT, StateT, AgentConfigT]):
    version: int = TRACE_VERSION
    """The trace schema this trace serializes as."""
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    """Unique ID for this trace, auto-generated."""
    verifiers: VersionInfo = Field(default_factory=_current_build)
    """The verifiers version that produced this trace."""
    run: RunInfo | None = None
    """The run this trace belongs to (eval or train), consumer-stamped."""

    task: TraceTask[DataT]
    """The task data that seeded this trace."""
    agent: AgentInfo[AgentConfigT]
    """The agent (harness x model x runtime) that produced this trace."""
    tools: list[Tool] = Field(default_factory=list)
    """The tools advertised to the agent, automatically recorded from last intercepted turn."""

    nodes: list[MessageNode] = Field(default_factory=list)
    """The message graph; branches are derived views and storage stays linear in turns."""
    calls: list[ModelCall] = Field(default_factory=list)
    """Every model call; automatically recorded at intercept time + linked into `nodes`."""

    rewards: dict[str, Reward | None] = Field(default_factory=dict)
    """Named, weighted rewards; `None` means scoring didn't run (e.g. because of a
    preceding error)."""
    metrics: dict[str, float | None] = Field(default_factory=dict)
    """Unweighted, named metrics; `None` as in `rewards`."""
    info: dict[str, Any] = Field(default_factory=dict)
    """Scratch space for task-specific metadata."""
    state: StateT = Field(default_factory=State, exclude=True)
    """Runtime (possibly, non-serializable) state shared across runtimes; excluded from serialization."""

    extra_usage: list[Usage] = Field(default_factory=list)
    """Usage from judges and other calls outside the agent's message graph."""

    is_completed: bool = False
    """Whether the trace completed."""
    ok: bool = False
    """Whether the trace completed successfully."""
    stop_condition: str | None = None
    """What stopped the trace."""
    errors: list[Error] = Field(default_factory=list)
    """Every error captured across attempts, oldest to newest."""
    timing: Timing = Field(default_factory=Timing)

    _head_index: dict = PrivateAttr(default_factory=dict)
    """`(parent, msg_hash) -> node_id` for the graph builder."""

    @property
    def reward(self) -> float:
        return sum(r.value for r in self.rewards.values() if r is not None)

    @property
    def has_error(self) -> bool:
        return not self.ok

    @property
    def num_input_tokens(self) -> int:
        """Fed-in tokens (system + user + tool), counted once, summed across branches."""
        return sum(branch.num_input_tokens for branch in self.branches)

    @property
    def num_output_tokens(self) -> int:
        """Model-generated tokens across all turns, summed across branches."""
        return sum(branch.num_output_tokens for branch in self.branches)

    @property
    def num_total_tokens(self) -> int:
        """Final sequence lengths (last prompt + completion) summed across branches."""
        return sum(branch.num_total_tokens for branch in self.branches)

    @property
    def usage(self) -> Usage | None:
        """Provider-reported usage summed once per actual model call in this rollout."""
        return Usage.aggregate(c.usage for c in self.calls if c.usage is not None)

    @property
    def branches(self) -> list[Branch]:
        """One root-to-leaf path per graph leaf, its calls attached in path order."""
        by_node = {c.node: c for c in self.calls if c.node is not None}
        branches: list[Branch] = []
        for i, leaf in enumerate(graph.leaves(self)):
            path: list[int] = []
            nid: int | None = leaf
            while nid is not None:
                path.append(nid)
                nid = self.nodes[nid].parent
            path.reverse()
            branches.append(
                Branch(
                    index=i,
                    nodes=[self.nodes[n] for n in path],
                    calls=[by_node[n] for n in path if n in by_node],
                )
            )
        return branches

    @property
    def num_branches(self) -> int:
        return len(graph.leaves(self))

    @property
    def num_turns(self) -> int:
        return sum(1 for n in self.nodes if n.sampled)

    @property
    def is_truncated(self) -> bool:
        """True for framework limits or a length-finished final response."""
        if self.stop_condition in (
            "max_turns",
            "max_input_tokens",
            "max_output_tokens",
            "max_total_tokens",
            "context_length",
        ):
            return True
        last = next((c for c in reversed(self.calls) if c.error is None), None)
        return bool(last and last.finish_reason == "length")

    @property
    def assistant_messages(self) -> list[AssistantMessage]:
        """Every model response, in order — one per turn, branch-independent. Excludes
        prompt-supplied assistant messages (`sampled` is the provenance signal)."""
        return [
            n.message
            for n in self.nodes
            if n.sampled and isinstance(n.message, AssistantMessage)
        ]

    @property
    def tool_messages(self) -> list[ToolMessage]:
        """Every tool result, in order — branch-independent."""
        return [n.message for n in self.nodes if isinstance(n.message, ToolMessage)]

    @property
    def last_reply(self) -> str:
        """The last recorded model response, in text format."""
        msgs = self.assistant_messages
        return (msgs[-1].content or "").strip() if msgs else ""

    @property
    def last_error(self) -> Error | None:
        """The last error captured across attempts."""
        return self.errors[-1] if self.errors else None

    @property
    def transcript(self) -> str:
        """Text transcript of the last branch's messages; tool outputs are omitted."""
        branches = self.branches
        blocks: list[str] = []
        for message in branches[-1].messages if branches else []:
            lines = [f"[{message.role}]"]
            if isinstance(message, AssistantMessage):
                if message.content:
                    lines.append(message.content)
                lines.extend(
                    f"[tool_call {call.name}({call.arguments})]"
                    for call in message.tool_calls or []
                )
            else:
                if isinstance(message, ToolMessage) and message.name:
                    lines[0] = f"[{message.role} {message.name}]"
                if text := content_text(message.content):
                    lines.append(text)
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def record_metric(self, name: str, value: float) -> None:
        self.metrics[name] = float(value)

    def record_metrics(self, values: Mapping[str, float]) -> None:
        for name, value in values.items():
            self.record_metric(name, value)

    def record_reward(self, name: str, value: float, weight: float = 1.0) -> None:
        reward = Reward(score=float(value), weight=float(weight))
        self.rewards[name] = reward

    def record_judge(self, response: JudgeResponse) -> None:
        self.info.setdefault("judge", []).append(response.model_dump())
        if response.usage is not None:
            self.extra_usage.append(response.usage)

    def record_run(self, run: RunInfo | None = None, **info: Any) -> None:
        """Record the run identity (eval / train), and optional extra info."""
        if run is not None:
            self.run = run
        self.info.update(info)

    def stop(self, condition: str) -> None:
        """Stop the trace, optionally with a stop condition."""
        self.is_completed = True
        if self.stop_condition is None:
            self.stop_condition = condition

    def split_agent_time(self) -> None:
        """Split the agent span into model and harness time."""
        span = self.timing.agent
        if not span.end:
            return
        model = sum(call.time.duration for call in self.calls)
        span.model.duration = min(model, span.duration)
        span.harness.duration = span.duration - span.model.duration

    def record_error(self, error: Exception) -> None:
        """Record an error, and stop the trace as failed."""
        self.errors.append(
            Error(
                type=type(error).__name__,
                message=str(error),
                status_code=getattr(error, "status_code", None),
                # Provider errors already carry the actionable upstream diagnostic.
                # Keep full tracebacks for every other failure.
                traceback=None
                if isinstance(error, ProviderError)
                else traceback.format_exc(),
            )
        )
        self.ok = False
        self.stop("error")

    def to_record(self) -> dict[str, Any]:
        """JSON record without raw tensors, which remain available on the msgpack wire."""
        return self.model_dump(mode="json", exclude=EXCLUDE_FIELDS)
