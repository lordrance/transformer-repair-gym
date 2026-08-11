class Runtime(ABC):
    is_local: ClassVar[bool] = True
    """Whether this runtime exchanges host-local URLs without a public tunnel. True for
    subprocess and Docker (directly or through Docker's policy proxy); remote runtimes
    override to False and use a host `Tunnel` inward plus `expose` outward."""

    info: BaseRuntimeInfo

    @property
    def supports_live_processes(self) -> bool:
        """Whether `open_process()` is implemented for this runtime instance."""
        return type(self).open_process is not Runtime.open_process

    def __init__(self, name: str | None = None) -> None:
        self.name = name or f"vf-{uuid.uuid4().hex[:12]}"
        self._uv_interpreters: dict[str, str] = {}
        self._uv_script_locks: dict[str, asyncio.Lock] = {}
        self._setup_claimed = False
        self.stopped = False
        """Whether teardown has begun (set by `stop`). A stopped runtime is dead: a rollout
        refuses to borrow one — the owner tore it down, so any use is a lifetime bug in the
        borrowing program, caught up front instead of failing opaquely mid-harness."""

    @property
    def type(self) -> str:
        return self.config.type

    @abstractmethod
    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        """Free the provisioned resource on the normal path (the owner's `finally`),
        shielded from cancellation: a Ctrl-C / SIGTERM cancels that `finally` mid-await,
        and an interrupted teardown leaks the container / paid sandbox. Runs `teardown`
        to completion, then re-raises the cancellation. Framework method — override
        `teardown`, not this."""
        self.stopped = True  # before the await: no new borrows once teardown begins
        await run_shielded(self.teardown())

    async def teardown(self) -> None:
        """Free the provisioned resource, off the event loop. Override only for teardown
        that must be async (e.g. a remote API call); `stop` shields it from cancellation.
        Best-effort and idempotent, like `cleanup`. An override must not consume state
        `cleanup` keys off before its first await: if the event loop dies mid-teardown
        (second Ctrl-C), the atexit backstop must still find the resource."""
        await asyncio.to_thread(self.cleanup)

    def cleanup(self) -> None:
        """Synchronously free the provisioned resource — best-effort and idempotent. The
        source of truth for teardown: usable from the atexit backstop where async machinery
        is dead, and run off the event loop by `stop` on the normal path. Default no-op."""

    @abstractmethod
    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        pass

    async def alive(self) -> bool:
        """Whether the box still executes anything. Not every runtime raises when
        the box is gone — some surface it as `exec`'s own non-zero result,
        indistinguishable from the command failing. One probe on the failure path
        tells the two apart before we blame anyone."""
        try:
            return (await self.run(["true"], {})).exit_code == 0
        except Exception:  # noqa: BLE001 - failing to exec at all means the box is gone
            return False

    async def run_program(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        """Run the harness's MAIN program — the rollout itself (a possibly long-lived, stateful,
        agentic run) — as opposed to the short idempotent infra ops (write / mv / install /
        provisioning) that go through `run`. No framework layer may replay this argv: doing so
        against the rollout's persistent trace would fork a duplicate branch. Provider SDKs may
        still retry individual safe transport operations underneath `run`."""
        return await self.run(argv, env)

    async def open_process(
        self, argv: list[str], env: dict[str, str]
    ) -> RuntimeProcess:
        """Start a live process for a rollout-scoped harness session."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support live processes"
        )

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        """Start `argv` as a background process in the runtime (combined output to
        `log`, a path in the workspace) and return immediately. It runs until `stop()`
        tears the runtime down. Used to host a tool server colocated with the harness."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support run_background"
        )

    async def prepare_uv_script(
        self,
        script: str | bytes,
        env: dict[str, str] | None = None,
        *,
        activate: bool = True,
    ) -> list[str]:
        """Prepare a PEP 723 script and return an argv that runs it.

        ``activate=False`` invokes the resolved script interpreter directly, keeping
        its environment variables out of child processes spawned by the script.
        """
        data = script.encode() if isinstance(script, str) else script
        digest = hashlib.sha256(data).hexdigest()
        path = f"/tmp/vf-scripts/{digest}.py"
        if digest not in self._uv_interpreters:
            async with self._uv_script_locks.setdefault(digest, asyncio.Lock()):
                if digest not in self._uv_interpreters:
                    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
                    await self.write(tmp, data)
                    command = (
                        f"mv -f {shlex.quote(tmp)} {shlex.quote(path)} "
                        f"&& {{ {_ENSURE_UV}; }} "
                        f"&& uv sync --script {shlex.quote(path)} -q --no-config "
                        f"&& uv python find --script {shlex.quote(path)} --no-config"
                    )
                    result = await self.run(["sh", "-c", command], env or {})
                    if result.exit_code != 0:
                        raise RuntimeError(
                            "failed to prepare uv script: "
                            f"{result.stderr.strip()[-2000:]}"
                        )
                    self._uv_interpreters[digest] = result.stdout.strip().splitlines()[
                        -1
                    ]
        interpreter = self._uv_interpreters[digest]
        if not activate:
            return [interpreter, path]
        venv = str(PurePosixPath(interpreter).parent.parent)
        command = (
            'export VIRTUAL_ENV="$1" PATH="${1}/bin:$HOME/.local/bin:$PATH" '
            'UV_INSTALL_DIR="$HOME/.local/bin" UV_RUN_RECURSION_DEPTH=1; '
            'shift; exec "$@"'
        )
        return [
            "sh",
            "-c",
            command,
            "uv-script",
            venv,
            interpreter,
            path,
        ]

    async def run_uv_script(
        self,
        script: str | bytes,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> ProgramResult:
        """The script is written to a stable, content-addressed path rather than the per-rollout
        workspace: uv keys its per-script environment by the script's full path, so a
        unique path per call would mint a fresh env every rollout. A path derived from the
        content means identical scripts share one path → uv reuses one env, bounded by the
        number of distinct scripts. Published via a unique temp + atomic `mv`, so
        concurrent rollouts writing the same content never race a half-written read."""
        argv = await self.prepare_uv_script(script, env)
        return await self.run([*argv, *(args or [])], env or {})

    async def read(self, path: str, max_bytes: int | None = None) -> bytes:
        """Read `path` into host memory. `max_bytes` caps the transfer, raising past
        the cap — for a file written by something we don't control, whose size we
        can't assume. The cap is enforced inside the box rather than after the
        transfer, and base64 because `run` returns decoded text. Framework method —
        override `_read`, not this."""
        if max_bytes is None:
            return await self._read(path)
        # Through a temp file, not a pipe: `head | base64` exits with base64's 0
        # even when the path is missing, and a missing file must raise here just
        # as it does from `_read`.
        result = await self.run(
            [
                "sh",
                "-c",
                (
                    "t=$(mktemp) || exit 1; "
                    'head -c "$1" -- "$2" > "$t" || { rm -f "$t"; exit 1; }; '
                    'base64 < "$t"; rc=$?; rm -f "$t"; exit $rc'
                ),
                "sh",
                str(max_bytes + 1),
                path,
            ],
            {},
        )
        if result.exit_code:
            raise SandboxError(f"read {path!r}: {result.stderr.strip()[-500:]}")
        data = base64.b64decode(result.stdout)
        if len(data) > max_bytes:
            raise SandboxError(f"read {path!r}: over the {max_bytes} byte limit")
        return data

    @abstractmethod
    async def _read(self, path: str) -> bytes:
        """Read the whole file at `path`; `read` adds the optional transfer cap."""

    @abstractmethod
    async def write(self, path: str, data: bytes) -> None:
        pass

    def host_url(self, url: str) -> str:
        """The URL a program inside this runtime uses to reach a host-bound `url`."""
        return url

    async def prepare_setup(self) -> None:
        """Open egress for trusted setup when reusing a restricted runtime."""
        if not self.network_restricted:
            return
        if self._setup_claimed:
            await self.prepare_execution(None)
        self._setup_claimed = True

    async def prepare_execution(self, routes: list[str] | None) -> None:
        """Last setup step, right before the agent starts. Restricted runtimes enforce
        their policy here; `routes` identifies the interception and MCP endpoints.
        None restores unrestricted egress for another trusted setup phase."""

    @property
    def network_restricted(self) -> bool:
        """Whether the agent phase uses filtered networking (see `prepare_execution`)."""
        return (
            isinstance(self.config, NetworkPolicyConfig)
            and self.config.network_restricted
        )

    @property
    def published_port(self) -> int | None:
        """A fixed port this runtime exposes to the outside at startup, declared up front to the
        provider (Modal forwards only ports named at `Sandbox.create`). When set, a server placed
        here binds it instead of a host-chosen free port, and `expose` returns its public URL.
        `None` for local runtimes (subprocess/docker), which pick a free port."""
        return None

    async def expose(self, port: int) -> str | None:
        """Publish a port running *inside this runtime* to a URL reachable from the host/outside,
        or None when local. A remote runtime overrides this with the provider's native port
        exposure (modal `tunnels()`, prime `client.expose`), torn down with the sandbox in
        `stop()`. The reverse of a host `Tunnel` (interception.tunnel, which reaches a host
        port from inside a runtime)."""
        return None
