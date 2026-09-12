"""Operator-only checkpoint migrations; API credentials never acquire DDL privileges."""

import asyncio

import psycopg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row

from scripts.migration_environment import MigrationError
from scripts.migration_settings import MigrationSettings


async def setup_checkpointer(settings: MigrationSettings) -> None:
    """Serialize library-managed DDL under app_owner outside a transaction."""
    db = settings.migration
    try:
        async with asyncio.timeout(120):
            conn = await psycopg.AsyncConnection.connect(
                make_conninfo(
                    host=db.host,
                    port=db.port,
                    user=db.user,
                    password=db.password.get_secret_value(),
                    dbname="insightpilot_app",
                ),
                autocommit=True,
                prepare_threshold=None,
                row_factory=dict_row,
                connect_timeout=5,
                options=f"-c statement_timeout={int(db.command_timeout_s * 1000)}",
            )
            async with conn:
                await conn.execute("SET ROLE app_owner")
                await conn.execute("SELECT pg_advisory_lock(192837465)")
                await AsyncPostgresSaver(conn).setup()
                await conn.execute("REVOKE UPDATE, DELETE ON public.data_evidence FROM app_rw")
    except (psycopg.Error, TimeoutError) as exc:
        raise MigrationError("Checkpoint initialization failed") from exc
