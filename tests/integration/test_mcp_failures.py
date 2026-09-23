"""Step 5.6 failure contracts through the real MCP process where applicable."""

# ruff: noqa: PLR2004 -- fixed failure counts and cooldown are acceptance assertions.

import socket
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from mcp import ClientSession
from mcp.types import CallToolResult
from pydantic import BaseModel, SecretStr
from structlog.testing import capture_logs

from app.agents.failures import FailureKind
from app.agents.nodes.common import node_failure
from app.clients.mcp_client import CircuitBreaker, McpClient
from app.core.config_models import MCPSettings
from app.core.deadline import Deadline
from app.core.errors import (
    McpAuthenticationError,
    McpCallTimeoutError,
    McpPolicyRejected,
    McpUnavailableError,
    SqlExecutionError,
    SqlTimeoutError,
)
from app.schemas.mcp import QueryArguments, SqlErrorKind
from tests.integration.mcp_support import client, mcp_endpoint, query

pytestmark = pytest.mark.integration
__all__ = ["client", "mcp_endpoint"]


class CountingMcpClient(McpClient):
    """Observe logical transport entries without replacing the real SDK or server."""

    def __init__(self, settings: MCPSettings) -> None:
        super().__init__(settings)
        self.session_attempts = 0
        self.raw_calls = 0

    @asynccontextmanager
    async def _session_context(self) -> AsyncIterator[ClientSession]:
        self.session_attempts += 1
        async with super()._session_context() as session:
            yield session

    async def _raw_call(
        self, name: str, args: BaseModel, *, omit_unset: bool = False
    ) -> CallToolResult:
        self.raw_calls += 1
        return await super()._raw_call(name, args, omit_unset=omit_unset)


async def test_connection_refused_retries_then_fails(mcp_endpoint: MCPSettings) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    settings = MCPSettings(
        base_url=f"http://127.0.0.1:{port}/mcp",
        auth_token=mcp_endpoint.auth_token,
        timeout_s=0.5,
    )
    client = CountingMcpClient(settings)
    try:
        with pytest.raises(McpUnavailableError):
            await query(client, "SELECT 1")
        assert client.session_attempts == 3
        assert client.breaker.failures == 1
    finally:
        await client.aclose()


async def test_timeout_not_retried(mcp_endpoint: MCPSettings) -> None:
    settings = mcp_endpoint.model_copy(update={"timeout_s": 2.0})
    client = CountingMcpClient(settings)
    try:
        await client.connect()
        settings.timeout_s = 0.05
        sql = (
            "SELECT sum(a.i * b.i) FROM generate_series(1,100000) a(i) "
            "CROSS JOIN generate_series(1,100000) b(i)"
        )
        with pytest.raises(McpCallTimeoutError):
            await query(client, sql)
        assert client.raw_calls == 1
        assert client.breaker.failures == 1
    finally:
        await client.aclose()


async def test_policy_rejection_not_retried(mcp_endpoint: MCPSettings) -> None:
    client = CountingMcpClient(mcp_endpoint)
    try:
        with pytest.raises(McpPolicyRejected):
            await query(client, "SELECT * FROM pg_catalog.pg_user")
        assert client.raw_calls == 1
        assert client.breaker.failures == 0
    finally:
        await client.aclose()


async def test_policy_rejection_does_not_open_breaker(mcp_endpoint: MCPSettings) -> None:
    client = CountingMcpClient(mcp_endpoint)
    try:
        for _ in range(10):
            with pytest.raises(McpPolicyRejected):
                await query(client, "SELECT * FROM pg_catalog.pg_user")
        assert client.breaker.failures == 0
        assert client.breaker.opened_at is None
        assert (await query(client, "SELECT 1")).rows == [[1]]
        assert client.raw_calls == 11
    finally:
        await client.aclose()


async def test_auth_failure_logged_as_error(mcp_endpoint: MCPSettings) -> None:
    settings = mcp_endpoint.model_copy(update={"auth_token": SecretStr("incorrect-token")})
    client = CountingMcpClient(settings)
    try:
        with capture_logs() as logs, pytest.raises(McpAuthenticationError):
            await query(client, "SELECT 1")
        assert client.session_attempts == 1
        assert client.breaker.failures == 0
        assert any(
            event["event"] == "mcp_authentication_failed" and event["log_level"] == "error"
            for event in logs
        )
        failure_kind = node_failure("execute_sql", McpAuthenticationError()).kind
        assert failure_kind is FailureKind.MCP_UNAVAILABLE
    finally:
        await client.aclose()


async def test_statement_timeout_not_correctable(client: McpClient) -> None:
    sql = (
        "SELECT sum(a.i * b.i) FROM generate_series(1,100000) a(i) "
        "CROSS JOIN generate_series(1,100000) b(i)"
    )
    with pytest.raises(SqlTimeoutError) as caught:
        await query(client, sql)
    assert node_failure("execute_sql", caught.value).kind is FailureKind.SQL_TIMEOUT
    assert client.breaker.failures == 0


async def test_execution_error_is_correctable(client: McpClient) -> None:
    with pytest.raises(SqlExecutionError) as caught:
        await query(client, "SELECT missing_column")
    assert caught.value.kind is SqlErrorKind.UNDEFINED_COLUMN
    assert node_failure("execute_sql", caught.value).kind is FailureKind.SQL_EXECUTION_FAILED
    assert client.breaker.failures == 0


async def test_breaker_open_fails_fast(client: McpClient) -> None:
    clock = [0.0]
    client.breaker = CircuitBreaker(clock=lambda: clock[0])
    for _ in range(5):
        client.breaker.failed()
    with pytest.raises(McpUnavailableError):
        await client.call_tool(
            "execute_readonly_query",
            QueryArguments(sql="SELECT 1"),
            deadline=Deadline(time.monotonic() + 5),
        )
    assert client.breaker.failures == 5


async def test_breaker_half_opens_after_cooldown(client: McpClient) -> None:
    clock = [0.0]
    client.breaker = CircuitBreaker(clock=lambda: clock[0])
    for _ in range(5):
        client.breaker.failed()
    clock[0] = 30.0
    assert (await query(client, "SELECT 1")).rows == [[1]]
    assert client.breaker.failures == 0
    assert client.breaker.opened_at is None


async def test_health_requires_ready_server_and_valid_token(mcp_endpoint: MCPSettings) -> None:
    client = McpClient(mcp_endpoint)
    try:
        await client.health()
        assert client.tools_loaded
        assert client.tools
        assert client.breaker.failures == 0
    finally:
        await client.aclose()
    invalid = McpClient(
        mcp_endpoint.model_copy(update={"auth_token": SecretStr("incorrect-token")})
    )
    try:
        with pytest.raises(McpAuthenticationError):
            await invalid.health()
        assert not invalid.tools_loaded
        assert invalid.breaker.failures == 0
    finally:
        await invalid.aclose()
