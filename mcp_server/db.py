"""Bounded async psycopg pool; the Step 1.7 approved driver exception."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import psycopg
from psycopg.conninfo import make_conninfo
from psycopg_pool import AsyncConnectionPool, PoolTimeout

from app.core.errors import McpUnavailableError
from mcp_server.config import BusinessSettings


class BusinessDatabase:
    """Own one runtime pool with read-only credentials and bounded cleanup."""

    def __init__(self, settings: BusinessSettings) -> None:
        self.settings = settings
        self.pool = AsyncConnectionPool(
            conninfo=make_conninfo(
                host=settings.host,
                port=settings.port,
                dbname=settings.database,
                user=settings.user,
                password=settings.password.get_secret_value(),
                connect_timeout=settings.connect_timeout_s,
            ),
            min_size=0,
            max_size=settings.pool_size,
            open=False,
            timeout=settings.connect_timeout_s,
            reconnect_timeout=settings.connect_timeout_s,
            kwargs={"autocommit": True, "prepare_threshold": None},
        )

    async def start(self) -> None:
        """Open the pool without an unbounded startup connectivity wait."""
        await self.pool.open()

    async def aclose(self) -> None:
        """Release the pool within the configured connection budget."""
        await self.pool.close(timeout=self.settings.connect_timeout_s)

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[psycopg.AsyncConnection[tuple[object, ...]]]:
        """Map pool/connection failures without leaking connection information."""
        try:
            async with self.pool.connection() as connection:
                yield connection
        except (PoolTimeout, psycopg.OperationalError) as exc:
            if isinstance(exc, psycopg.errors.QueryCanceled):
                raise
            raise McpUnavailableError() from exc

    async def check(self) -> None:
        """Readiness checks the real database without policy retries."""
        async with asyncio.timeout(self.settings.connect_timeout_s), self.connection() as conn:
            await conn.execute("SELECT 1")
