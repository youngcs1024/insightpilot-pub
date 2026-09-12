"""Authenticated MCP v2 business-data server on the private Compose network."""

import asyncio
import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

import structlog
from mcp.server import MCPServer
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.types import CallToolResult, TextContent
from pydantic import AnyHttpUrl, Field, SecretStr
from starlette.requests import Request
from starlette.responses import JSONResponse
from structlog.types import EventDict, WrappedLogger

from app.core.errors import InsightPilotError, McpPolicyRejected, SqlExecutionError
from app.schemas.mcp import (
    McpErrorCode,
    McpErrorPayload,
    PositiveRows,
    QueryResultPayload,
    SqlErrorKind,
    ValidationStatus,
)
from app.schemas.schema_catalog import BusinessSchemaResponse
from mcp_server.config import McpServerSettings
from mcp_server.db import BusinessDatabase
from mcp_server.policy.sql_validator import SQLValidator, clamp_result_cap
from mcp_server.tools.business_schema import BusinessSchemaReader
from mcp_server.tools.execute_query import QueryExecutor

logger = structlog.get_logger(__name__)


class StaticTokenVerifier:
    """Constant-time token comparison without OAuth infrastructure."""

    def __init__(self, token: SecretStr) -> None:
        self._token = token

    async def verify_token(self, token: str) -> AccessToken | None:
        """Authenticate a single internal service principal."""
        if not secrets.compare_digest(token.encode(), self._token.get_secret_value().encode()):
            return None
        return AccessToken(token=token, client_id="insightpilot-api", scopes=["query"])


def error_result(exc: InsightPilotError) -> CallToolResult:
    """Return typed structured errors; never depend on SDK exception prose."""
    payload = McpErrorPayload(
        code=McpErrorCode(exc.code),
        sql_error=exc.kind if isinstance(exc, SqlExecutionError) else SqlErrorKind.OTHER,
        message=exc.user_message,
        status=exc.status if isinstance(exc, McpPolicyRejected) else None,
        reasons=exc.reasons if isinstance(exc, McpPolicyRejected) else [],
    )
    return CallToolResult(
        is_error=True,
        structured_content=payload.model_dump(mode="json"),
        content=[TextContent(type="text", text=payload.model_dump_json())],
    )


def create_server(settings: McpServerSettings) -> MCPServer[None]:
    """Build the supported tools and their pool; imports never open a connection."""
    database = BusinessDatabase(settings.business)
    executor = QueryExecutor(database)
    validator = SQLValidator()
    schema_reader = BusinessSchemaReader(database)

    @asynccontextmanager
    async def lifespan(server: MCPServer[None]) -> AsyncIterator[None]:
        await database.start()
        try:
            yield
        finally:
            async with asyncio.timeout(settings.mcp.shutdown_timeout_s):
                await database.aclose()

    server: MCPServer[None] = MCPServer(
        "insightpilot-business",
        log_level="WARNING",
        lifespan=lifespan,
        token_verifier=StaticTokenVerifier(settings.mcp.auth_token),
        auth=AuthSettings(
            issuer_url=AnyHttpUrl(f"http://mcp:{settings.mcp.port}"),
            resource_server_url=AnyHttpUrl(f"http://mcp:{settings.mcp.port}/mcp"),
            required_scopes=["query"],
        ),
    )

    @server.tool()
    async def execute_readonly_query(
        sql: Annotated[str, Field(min_length=1, max_length=32000)],
        max_rows: PositiveRows = 1000,
    ) -> Annotated[CallToolResult, QueryResultPayload]:
        """Execute one read-only PostgreSQL query and return typed, bounded rows."""
        try:
            cap = clamp_result_cap(max_rows)
            outcome = validator.validate(sql, result_cap=cap)
            if outcome.status is not ValidationStatus.VALID:
                # SQL literals can contain arbitrary secrets/PII: log no user literals.
                logger.warning("sql_policy_rejected", reasons=outcome.reasons, sql="<redacted>")
                raise McpPolicyRejected(outcome.status, outcome.reasons)
            result = await executor.execute(
                outcome.rewritten_sql, max_rows=cap, limit_applied=outcome.limit_applied
            )
            return CallToolResult(
                structured_content=result.model_dump(mode="json"),
                content=[TextContent(type="text", text=result.model_dump_json())],
            )
        except InsightPilotError as exc:
            return error_result(exc)

    @server.tool()
    async def get_business_schema(
        known_revision: Annotated[str | None, Field(min_length=1, max_length=64)] = None,
    ) -> Annotated[CallToolResult, BusinessSchemaResponse]:
        """Return physical metadata for eight fixed tables, never business rows."""
        try:
            result = await schema_reader.read(known_revision)
            logger.info(
                "business_schema_read", revision=result.revision, unchanged=result.unchanged
            )
            return CallToolResult(
                structured_content=result.model_dump(mode="json"),
                content=[TextContent(type="text", text=result.model_dump_json())],
            )
        except InsightPilotError as exc:
            logger.exception("business_schema_failed", code=exc.code)
            return error_result(exc)

    @server.custom_route("/health", methods=["GET"])  # type: ignore[untyped-decorator]
    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @server.custom_route("/ready", methods=["GET"])  # type: ignore[untyped-decorator]
    async def ready(request: Request) -> JSONResponse:
        try:
            await database.check()
        except Exception:
            logger.exception("mcp_readiness_failed")
            return JSONResponse({"ready": False}, status_code=503)
        return JSONResponse({"ready": True})

    return server


def safe_exception(_: WrappedLogger, __: str, event: EventDict) -> EventDict:
    """Never render driver exception prose from this credential-holding process."""
    event.pop("exc_info", None)
    return event


def main() -> None:
    """Run the private server with safe stdout-only application events."""
    settings = McpServerSettings.load()
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.add_log_level,
            safe_exception,
            structlog.processors.JSONRenderer(),
        ]
    )
    # SDK and pool diagnostics can carry connection metadata; project events are sufficient.
    logging.getLogger("psycopg.pool").disabled = True
    create_server(settings).run(
        transport="streamable-http",
        host=settings.mcp.host,
        port=settings.mcp.port,
        streamable_http_path="/mcp",
        stateless_http=False,
    )


if __name__ == "__main__":
    main()
