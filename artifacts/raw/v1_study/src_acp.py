class ACP:
    """Run one-shot ACP agents or create rollout-scoped ACP sessions."""

    async def setup(self, harness: Harness, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(
            ACP_SOURCE, {**harness.config.resolved_env, "UV_FROZEN": "false"}
        )

    def session(
        self,
        harness: Harness,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
        *,
        env: dict[str, str],
        command: list[str],
        prompt: str | Messages | None,
        system_prompt: str | None = None,
        session_meta: dict | None = None,
    ) -> "ACPHarnessSession":
        """Create a persistent ACP-backed handle owned by one rollout."""
        return ACPHarnessSession(
            harness,
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            mcp_urls,
            data,
            env=env,
            command=command,
            prompt=prompt,
            system_prompt=system_prompt,
            session_meta=session_meta,
        )

    async def run(
        self,
        runtime: Runtime,
        env: dict[str, str],
        command: list[str],
        prompt: str | Messages | None,
        *,
        mcp_urls: dict[str, str] | None = None,
        system_prompt: str | None = None,
        session_path: str | None = None,
        session_meta: dict | None = None,
        allow_empty_tool_reply: bool = False,
    ) -> ProgramResult:
        """Run one ACP segment without retaining its process."""
        return await self._run(
            runtime,
            env,
            command,
            prompt,
            mcp_urls=mcp_urls,
            system_prompt=system_prompt,
            session_path=session_path,
            session_meta=session_meta,
            allow_empty_tool_reply=allow_empty_tool_reply,
        )

    async def _run(
        self,
        runtime: Runtime,
        env: dict[str, str],
        command: list[str],
        prompt: str | Messages | None,
        *,
        mcp_urls: dict[str, str] | None = None,
        system_prompt: str | None = None,
        session_path: str | None = None,
        session_meta: dict | None = None,
        allow_empty_tool_reply: bool = False,
    ) -> ProgramResult:
        if prompt is None:
            raise ValueError("ACP requires a prompt")
        messages = (
            [{"role": "user", "content": prompt}]
            if isinstance(prompt, str)
            else [message_to_wire(message) for message in prompt]
        )
        config = {
            "command": command,
            "messages": messages,
            "mcp_urls": mcp_urls or {},
            "system_prompt": system_prompt or "",
            "session_path": session_path,
            "session_meta": session_meta or {},
            "allow_empty_tool_reply": allow_empty_tool_reply,
        }
        program = await runtime.prepare_uv_script(
            ACP_SOURCE,
            {**env, "UV_FROZEN": "false"},
            activate=False,
        )
        directory = f".vf-acp-{secrets.token_hex(8)}"
        created = await runtime.run(["mkdir", "-m", "700", directory], {})
        if created.exit_code != 0:
            raise RuntimeError(f"ACP config directory failed: {created.stderr.strip()}")
        path = f"{directory}/config.json"
        try:
            await runtime.write(path, json.dumps(config).encode())
            return await runtime.run_program([*program, "once", path], env)
        finally:
            await run_shielded(runtime.run(["rm", "-rf", directory], {}))
