"""Typed read-only execution, independent of MCP transport and SQL validation."""

import asyncio
import math
import time as clock
from datetime import date, datetime, time
from decimal import Decimal
from uuid import UUID, uuid4

import psycopg
import structlog

from app.core.errors import McpResultError, SqlExecutionError, SqlTimeoutError
from app.schemas.mcp import ColumnSpec, QueryResultPayload, SqlErrorKind, SqlValue
from mcp_server.db import BusinessDatabase
from mcp_server.policy.sql_validator import clamp_result_cap

logger = structlog.get_logger(__name__)


def encode_value(value: object) -> SqlValue:
    """Encode supported scalar values without float-rounding Decimal data."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, Decimal) and value.is_finite():
        return str(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    raise McpResultError()


class QueryExecutor:
    """Database defence in depth still applies when callers bypass validation."""

    def __init__(self, database: BusinessDatabase) -> None:
        self.database = database

    async def execute(
        self, sql: str, *, max_rows: int = 1000, limit_applied: bool = False
    ) -> QueryResultPayload:
        """Execute once; no retries of a database transaction or SQL timeout."""
        cap = clamp_result_cap(max_rows)
        try:
            async with asyncio.timeout(self.database.settings.operation_timeout_s):
                return await self._execute(sql, cap, limit_applied)
        except (psycopg.errors.QueryCanceled, TimeoutError) as exc:
            logger.exception("sql_timeout")
            raise SqlTimeoutError() from exc
        except psycopg.Error as exc:
            logger.exception("sql_execution_failed", sqlstate=exc.sqlstate)
            raise SqlExecutionError(
                {
                    "42P01": SqlErrorKind.UNDEFINED_TABLE,
                    "42703": SqlErrorKind.UNDEFINED_COLUMN,
                    "42804": SqlErrorKind.TYPE_MISMATCH,
                }.get(exc.sqlstate or "", SqlErrorKind.OTHER)
            ) from exc

    async def _execute(self, sql: str, cap: int, limit_applied: bool) -> QueryResultPayload:
        started = clock.monotonic()
        async with self.database.connection() as conn, conn.transaction():
            await conn.execute("SET TRANSACTION READ ONLY")
            await conn.execute("SET LOCAL statement_timeout = '10s'")
            await conn.execute("SET LOCAL search_path = biz, pg_catalog")
            # The validator bounds returned SQL rows; bypassing it still cannot bypass READ ONLY.
            async with conn.cursor() as cursor:
                await cursor.execute(sql)
                rows = await cursor.fetchmany(cap + 1)
                if cursor.description is None:
                    raise McpResultError()
                columns = []
                for column in cursor.description:
                    info = conn.adapters.types.get(column.type_code)
                    columns.append(
                        ColumnSpec(
                            name=column.name, type=info.name if info else f"oid:{column.type_code}"
                        )
                    )
        result = QueryResultPayload(
            executed_sql=sql,
            limit_applied=limit_applied,
            execution_ms=int((clock.monotonic() - started) * 1000),
            mcp_call_id=str(uuid4()),
            columns=columns,
            rows=[[encode_value(value) for value in row] for row in rows[:cap]],
            row_count=min(len(rows), cap),
            result_truncated=len(rows) > cap,
        )
        logger.info("sql_executed", rows=result.row_count, result_truncated=result.result_truncated)
        return result
