"""Shared Alembic adapter with one configuration contract for online and offline DDL."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import asyncpg  # type: ignore[import-untyped]  # asyncpg has no typing marker.
import structlog
from sqlalchemy import pool, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context
from app.core.errors import DatabaseError
from app.db.base import Base
from app.db.models import Conversation, Turn, User
from data.seed.schema import BUSINESS_METADATA
from scripts.migration_settings import MigrationSettings, MigrationTarget

if TYPE_CHECKING:
    from alembic.runtime.environment import (
        EnvironmentContext,
        NameFilterParentNames,
        NameFilterType,
    )
    from sqlalchemy import Connection
    from sqlalchemy.schema import SchemaItem

LOGGER = structlog.get_logger()
EXCLUDE_TABLES = frozenset(
    {"checkpoints", "checkpoint_blobs", "checkpoint_writes", "checkpoint_migrations"}
)
# Explicit imports above register every application table, without importing API settings.
APP_MODELS = (User, Conversation, Turn)


class MigrationError(DatabaseError):
    """DDL failed; rollback and operator inspection are required, never automatic replay."""

    code = "MIGRATION_FAILED"
    user_message = "Migration failed; inspect the selected database and migration revision."


def include_object(
    obj: SchemaItem,
    name: str | None,
    type_: str,
    reflected: bool,
    compare_to: SchemaItem | None,
) -> bool:
    """Leave the four externally managed checkpoint tables untouched."""
    return not (type_ == "table" and name in EXCLUDE_TABLES)


def include_business_name(
    name: str | None,
    type_: NameFilterType,
    parent_names: NameFilterParentNames,
) -> bool:
    """Reflect only the business and migration-owned operational schemas."""
    return type_ != "schema" or name in {"biz", "ops", "mcp"}


def configure_context(
    environment: EnvironmentContext,
    target: MigrationTarget,
    *,
    connection: Connection | None = None,
    url: str | None = None,
) -> None:
    """Keep schema comparison and offline SQL settings identical across entrypoints."""
    business = target == MigrationTarget.BUSINESS
    environment.configure(
        connection=connection,
        url=url,
        target_metadata=BUSINESS_METADATA if business else Base.metadata,
        version_table="alembic_version_biz" if business else "alembic_version_app",
        version_table_schema="biz" if business else None,
        include_object=include_object,
        include_name=include_business_name if business else None,
        include_schemas=business,
        compare_type=True,
        compare_server_default=True,
        literal_binds=connection is None,
        dialect_opts={"paramstyle": "named"},
    )


def owner_statement(target: MigrationTarget) -> str:
    """Return fixed SQL identifiers; connection configuration cannot inject an owner."""
    return (
        "SET LOCAL ROLE biz_owner"
        if target == MigrationTarget.BUSINESS
        else "SET LOCAL ROLE app_owner"
    )


def migrate_connection(connection: Connection, target: MigrationTarget) -> None:
    """Run one transactional revision chain under its object creator role."""
    configure_context(context, target, connection=connection)  # type: ignore[arg-type]  # Alembic module proxies EnvironmentContext.
    with context.begin_transaction():
        connection.execute(text(owner_statement(target)))
        context.run_migrations()


async def run_online(settings: MigrationSettings, target: MigrationTarget) -> None:
    """Bound every database call and dispose connections even when migration execution fails."""
    database = settings.migration
    engine = create_async_engine(
        database.url(target),
        poolclass=pool.NullPool,
        hide_parameters=True,
        connect_args={
            "timeout": database.connect_timeout_s,
            "command_timeout": database.command_timeout_s,
            "server_settings": {
                "statement_timeout": str(int(database.command_timeout_s * 1000)),
                "lock_timeout": str(int(database.lock_timeout_s * 1000)),
            },
        },
    )
    try:
        async with engine.connect() as connection:
            await connection.run_sync(migrate_connection, target)
    except (
        SQLAlchemyError,
        asyncpg.PostgresError,
        asyncpg.InterfaceError,
        OSError,
        TimeoutError,
    ) as exc:
        # Driver messages and exception chains can include SQL/credentials; log only safe fields.
        LOGGER.exception(
            "migration_failed",
            target=target.value,
            exception_type=type(exc).__name__,
            exc_info=False,
        )
        raise MigrationError("Database migration failed.") from None
    finally:
        await engine.dispose()
    LOGGER.info("migration_completed", target=target.value)


def run_environment(target: MigrationTarget) -> None:
    """Dispatch Alembic CLI commands without constructing application runtime resources."""
    if context.is_offline_mode():
        configure_context(context, target, url="postgresql+asyncpg://")  # type: ignore[arg-type]  # Alembic context proxy.
        with context.begin_transaction():
            context.execute(owner_statement(target))
            context.run_migrations()
    else:
        asyncio.run(run_online(MigrationSettings.load(), target))
