"""Real, bounded readiness probes; no business queries or persistent substitutes."""

import asyncio
from typing import Protocol

import httpx2
import structlog
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel
from sqlalchemy import text

from app.core.config_models import HealthSettings, MCPSettings
from app.core.errors import HealthProbeError, HealthProbeTimeoutError
from app.db.session import Database

logger = structlog.get_logger(__name__)


class HealthChecks(BaseModel):
    """The only dependency state exposed by Step 1.1."""

    postgresql: bool = False
    mcp: bool = False
    model_runtime: bool | None = None

    @property
    def ready(self) -> bool:
        """Whether both required services responded successfully."""
        return self.postgresql and self.mcp and self.model_runtime is not False


class Probe(Protocol):
    """A probe owns its resources and supports partially initialized cleanup."""

    async def check(self) -> None:
        """Check availability or raise a typed failure."""
        ...

    async def aclose(self) -> None:
        """Release any resources acquired by this instance."""
        ...


class PostgreSQLProbe:
    """Borrow the application's pool for SELECT 1; never own a second engine."""

    def __init__(self, database: Database) -> None:
        self._database = database

    async def check(self) -> None:
        """Connect asynchronously; SQLAlchemy reconnects after dependency recovery."""
        try:
            async with self._database.engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
        except Exception as exc:
            raise HealthProbeError from exc

    async def aclose(self) -> None:
        """The application lifespan releases the shared pool after probes stop."""


class MCPProbe:
    """Use a short authenticated SDK session; never invoke business tools."""

    def __init__(self, settings: MCPSettings, timeout_s: float) -> None:
        self._settings = settings
        self._timeout_s = timeout_s

    async def check(self) -> None:
        """Initialize, list tools, and close in the same task/context scope."""
        try:
            async with (
                httpx2.AsyncClient(
                    headers={
                        "Authorization": "Bearer " + self._settings.auth_token.get_secret_value()
                    },
                    timeout=self._timeout_s,
                    trust_env=False,
                ) as http,
                streamable_http_client(str(self._settings.base_url), http_client=http) as (
                    read,
                    write,
                ),
                ClientSession(read, write, read_timeout_seconds=self._timeout_s) as session,
            ):
                await session.initialize()
                await session.list_tools()
        except Exception as exc:
            raise HealthProbeError from exc

    async def aclose(self) -> None:
        """Sessions are closed by each check's context managers, including cancellation."""


class HealthService:
    """Coordinate probes without retaining stale dependency success across requests."""

    def __init__(
        self, postgresql: Probe, mcp: Probe, settings: HealthSettings, model: Probe | None = None
    ) -> None:
        self._postgresql = postgresql
        self._mcp = mcp
        self._settings = settings
        self.active = False
        self._model = model

    async def start(self) -> HealthChecks:
        """Activate and check dependencies without failing on transient outages."""
        self.active = True
        return await self.check()

    async def check(self) -> HealthChecks:
        """Run both full operations concurrently with independent budgets and no retry."""
        if not self.active:
            return HealthChecks()
        async with asyncio.TaskGroup() as group:
            database = group.create_task(
                self._check_one("postgresql", self._postgresql, self._settings.postgresql_timeout_s)
            )
            mcp = group.create_task(self._check_one("mcp", self._mcp, self._settings.mcp_timeout_s))
            model = (
                group.create_task(self._check_one("model_runtime", self._model, 2))
                if self._model
                else None
            )
        return HealthChecks(
            postgresql=database.result(),
            mcp=mcp.result(),
            model_runtime=model.result() if model else None,
        )

    @staticmethod
    async def _bounded_check(probe: Probe, timeout_s: float) -> None:
        try:
            async with asyncio.timeout(timeout_s):
                await probe.check()
        except TimeoutError as exc:
            raise HealthProbeTimeoutError from exc

    @classmethod
    async def _check_one(cls, name: str, probe: Probe, timeout_s: float) -> bool:
        try:
            await cls._bounded_check(probe, timeout_s)
        except Exception:
            # Both typed adapter failures and unexpected SDK failures fail closed.
            logger.exception("readiness_check_failed", dependency=name)
            return False
        return True

    async def aclose(self) -> None:
        """Attempt both cleanups even when one fails; the lifespan bounds total time."""
        self.active = False
        async with asyncio.TaskGroup() as group:
            if self._model is not None:
                group.create_task(self._close_one("model_runtime", self._model))
            group.create_task(self._close_one("mcp", self._mcp))
            group.create_task(self._close_one("postgresql", self._postgresql))

    @staticmethod
    async def _close_one(name: str, probe: Probe) -> None:
        try:
            await probe.aclose()
        except Exception:
            logger.exception("readiness_cleanup_failed", dependency=name)
