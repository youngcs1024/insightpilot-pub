"""Application-owned async pool, request sessions and database failure translation."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg  # type: ignore[import-untyped]  # asyncpg 0.31.0 ships no typing marker/stubs.
import structlog
from fastapi import Request
from sqlalchemy.exc import DBAPIError, InterfaceError, OperationalError, SQLAlchemyError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config_models import DatabaseSettings
from app.core.errors import (
    ConflictError,
    DatabaseError,
    DatabaseTimeoutError,
    InsightPilotError,
    UpstreamUnavailableError,
)

logger = structlog.get_logger(__name__)


def translate_database_error(exc: Exception) -> InsightPilotError:
    """Classify driver fields and exception types, never internal error prose."""
    if isinstance(exc, (TimeoutError, PoolTimeoutError)):
        return DatabaseTimeoutError()
    source = exc.orig if isinstance(exc, DBAPIError) else exc
    sqlstate = getattr(source, "sqlstate", None)
    if sqlstate == "23505":
        return ConflictError()
    if sqlstate == "57014":
        return DatabaseTimeoutError()
    if (
        (isinstance(exc, DBAPIError) and exc.connection_invalidated)
        or isinstance(exc, (OperationalError, InterfaceError, asyncpg.InterfaceError, OSError))
        or sqlstate
        in {
            "08000",
            "08001",
            "08003",
            "08004",
            "08006",
            "08007",
            "08P01",
            "28000",
            "28P01",
            "3D000",
            "57P01",
            "57P02",
            "57P03",
            "53300",
        }
    ):
        return UpstreamUnavailableError()
    return DatabaseError()


class Database:
    """Own one pool per application instance; never retain a shared session."""

    def __init__(self, settings: DatabaseSettings) -> None:
        self._settings = settings
        self._engine: AsyncEngine | None = None
        self._sessions: async_sessionmaker[AsyncSession] | None = None

    def start(self) -> None:
        """Configure the pool without connecting, so transient outages allow startup."""
        if self._engine is not None:
            return
        self._engine = create_async_engine(
            self._settings.app_url,
            pool_pre_ping=True,
            pool_size=self._settings.pool_size,
            max_overflow=self._settings.max_overflow,
            pool_recycle=self._settings.pool_recycle_s,
            pool_timeout=self._settings.pool_timeout_s,
            echo=self._settings.echo_sql,
            hide_parameters=True,
            connect_args={
                "timeout": self._settings.connect_timeout_s,
                "command_timeout": self._settings.command_timeout_s,
            },
        )
        self._sessions = async_sessionmaker(self._engine, expire_on_commit=False)
        logger.info("database_pool_started", pool_size=self._settings.pool_size)

    @property
    def engine(self) -> AsyncEngine:
        """Provide the shared readiness engine only while the application is active."""
        if self._engine is None:
            raise UpstreamUnavailableError()
        return self._engine

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield an independent session; closing rolls back any uncommitted transaction."""
        if self._sessions is None:
            raise UpstreamUnavailableError()
        try:
            async with self._sessions() as session:
                yield session
        except (
            SQLAlchemyError,
            asyncpg.PostgresError,
            asyncpg.InterfaceError,
            OSError,
            TimeoutError,
        ) as exc:
            failure = translate_database_error(exc)
            logger.exception("database_operation_failed", code=failure.code)
            raise failure from exc

    async def aclose(self) -> None:
        """Release this application's pool; the caller bounds lifecycle cleanup."""
        engine, self._engine, self._sessions = self._engine, None, None
        if engine is not None:
            await engine.dispose()
            logger.info("database_pool_closed")


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Inject a request session; the calling service owns begin/commit boundaries."""
    database: Database = request.app.state.database
    async with database.session() as session:
        yield session
