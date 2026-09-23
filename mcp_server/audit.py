"""Write-only, bounded audit persistence for authenticated MCP tool calls."""

import asyncio
import hashlib
import json
import time
from collections.abc import Mapping
from enum import StrEnum
from uuid import UUID

import structlog
from mcp.types import CallToolResult
from psycopg.conninfo import make_conninfo
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, ValidationError

from app.schemas.mcp import McpErrorCode, McpErrorPayload, PolicyReason, QueryResultPayload
from app.schemas.schema_tools import SchemaResponse
from mcp_server.config import AuditSettings

logger = structlog.get_logger(__name__)


class AuditOutcome(StrEnum):
    """Closed set of persisted MCP outcomes."""

    OK = "ok"
    POLICY_REJECTED = "policy_rejected"
    EXECUTION_ERROR = "execution_error"
    TIMEOUT = "timeout"


class AuditEvent(BaseModel):
    """Values permitted to cross into the write-only audit connection."""

    caller: str
    correlation_id: str | None
    tool: str
    arguments_sha256: str
    sql_text: str | None
    outcome: AuditOutcome
    reject_reasons: list[str] | None = None
    rows_returned: int | None = None
    duration_ms: int


def hash_arguments(arguments: dict[str, object]) -> str:
    """Hash the complete wire arguments without retaining their raw values."""
    canonical = json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def correlation_header(headers: Mapping[str, str] | None) -> str | None:
    """Accept only the API's UUID request identifier from the HTTP header."""
    if headers is None:
        return None
    supplied = next(
        (value for key, value in headers.items() if key.lower() == "x-request-id"), None
    )
    if supplied is None:
        return None
    try:
        return UUID(supplied).hex
    except ValueError:
        return None


def classify_result(
    tool: str, result: CallToolResult
) -> tuple[AuditOutcome, list[str] | None, int | None]:
    """Derive audit fields from typed tool results, never exception prose."""
    content = result.structured_content
    if result.is_error:
        return _classify_error(content)
    if tool == "get_schema":
        return _classify_schema(content)
    if tool == "execute_readonly_query":
        return _classify_query(content)
    return AuditOutcome.OK, None, None


def _classify_error(content: object) -> tuple[AuditOutcome, list[str] | None, int | None]:
    try:
        error = McpErrorPayload.model_validate(content)
    except ValidationError:
        return AuditOutcome.EXECUTION_ERROR, None, None
    if error.code is McpErrorCode.POLICY_REJECTED:
        return AuditOutcome.POLICY_REJECTED, [reason.value for reason in error.reasons], None
    if error.code is McpErrorCode.SQL_TIMEOUT:
        return AuditOutcome.TIMEOUT, None, None
    return AuditOutcome.EXECUTION_ERROR, None, None


def _classify_schema(content: object) -> tuple[AuditOutcome, list[str] | None, int | None]:
    try:
        schema = SchemaResponse.model_validate(content)
    except ValidationError:
        return AuditOutcome.EXECUTION_ERROR, None, None
    if schema.rejected:
        return AuditOutcome.POLICY_REJECTED, [PolicyReason.TABLE_NOT_ALLOWED.value], None
    return AuditOutcome.OK, None, None


def _classify_query(content: object) -> tuple[AuditOutcome, list[str] | None, int | None]:
    try:
        query = QueryResultPayload.model_validate(content)
    except ValidationError:
        return AuditOutcome.EXECUTION_ERROR, None, None
    return AuditOutcome.OK, None, query.row_count


class AuditWriter:
    """Insert through a dedicated role; failures remain visible but never change tool results."""

    def __init__(self, settings: AuditSettings) -> None:
        self.settings = settings
        self.write_failures_total = 0
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
        """Open the independent pool without a startup connectivity wait."""
        await self.pool.open()

    async def aclose(self) -> None:
        """Close audit connections within their own bounded budget."""
        await self.pool.close(timeout=self.settings.connect_timeout_s)

    async def record(self, event: AuditEvent) -> None:
        """Insert once outside the business read-only transaction; never retry unknown commits."""
        try:
            async with (
                asyncio.timeout(self.settings.operation_timeout_s),
                self.pool.connection() as conn,
            ):
                await conn.execute(
                    """INSERT INTO mcp.audit_log
                       (caller, correlation_id, tool, arguments_sha256, sql_text,
                        outcome, reject_reasons, rows_returned, duration_ms)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (
                        event.caller,
                        event.correlation_id,
                        event.tool,
                        event.arguments_sha256,
                        event.sql_text,
                        event.outcome.value,
                        Jsonb(event.reject_reasons) if event.reject_reasons is not None else None,
                        event.rows_returned,
                        event.duration_ms,
                    ),
                )
        except Exception as exc:
            self.write_failures_total += 1
            logger.exception(
                "audit_write_failed",
                failure_count=self.write_failures_total,
                error_type=type(exc).__name__,
            )


def elapsed_ms(started: float) -> int:
    """Measure the tool call itself, excluding audit persistence."""
    return max(0, int((time.monotonic() - started) * 1000))
