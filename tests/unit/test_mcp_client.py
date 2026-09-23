"""Deterministic transport, retry and circuit-breaker behavior without sockets."""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http import HTTPStatus
from unittest.mock import AsyncMock

import httpx2
import pytest
from asgi_correlation_id import correlation_id
from mcp import ClientSession
from mcp.shared.exceptions import MCPError
from mcp.types import CONNECTION_CLOSED, CallToolResult
from pydantic import SecretStr

from app.clients.mcp_client import CircuitBreaker, McpClient, decode_result, transport_failure
from app.core.config_models import MCPSettings
from app.core.deadline import Deadline
from app.core.errors import (
    DeadlineExceededError,
    McpPolicyRejected,
    McpResultError,
    McpUnavailableError,
    SchemaDriftError,
    SchemaMetadataError,
    SqlTimeoutError,
)
from app.schemas.mcp import QueryArguments
from app.schemas.metric_tools import MetricFragment
from app.schemas.schema_tools import GetSchemaArgs
from tests.factories import business_schema
from tests.factories import mcp_success as success
from tests.metric_tool_support import metric_args

TWO = 2


@pytest.fixture
def settings() -> MCPSettings:
    return MCPSettings(auth_token=SecretStr("test-token"), timeout_s=0.1)


def failure(code: str) -> CallToolResult:
    return CallToolResult(
        content=[],
        is_error=True,
        structured_content={
            "code": code,
            "message": "Ignored upstream prose",
        },
    )


@pytest.mark.parametrize(
    ("code", "exception"),
    [
        ("MCP_POLICY_REJECTED", McpPolicyRejected),
        ("SQL_TIMEOUT", SqlTimeoutError),
        ("MCP_UNAVAILABLE", McpUnavailableError),
    ],
)
def test_error_mapping(code: str, exception: type[Exception]) -> None:
    with pytest.raises(exception):
        decode_result(failure(code))


def test_invalid_result_rejected() -> None:
    bad = success()
    bad.structured_content["row_count"] = 2
    with pytest.raises(McpResultError):
        decode_result(bad)


async def test_metric_policy_rejection_is_not_retried_or_counted(settings: MCPSettings) -> None:
    session = AsyncMock(spec=ClientSession)
    session.call_tool.return_value = CallToolResult(
        content=[],
        is_error=True,
        structured_content={
            "code": "MCP_POLICY_REJECTED",
            "message": "ignored",
            "status": "invalid",
            "reasons": ["unknown_column"],
            "column_name": "imagined_amount",
        },
    )

    @asynccontextmanager
    async def factory() -> AsyncIterator[ClientSession]:
        yield session

    client = McpClient(settings, session_factory=factory)
    try:
        with pytest.raises(McpPolicyRejected) as caught:
            await client.resolve_metric(metric_args(), deadline=Deadline(time.monotonic() + 5))
        assert caught.value.column_name == "imagined_amount"
        assert session.call_tool.await_count == 1
        assert client.breaker.failures == 0
    finally:
        await client.aclose()


async def test_metric_client_accepts_complete_normalized_sql(settings: MCPSettings) -> None:
    session = AsyncMock(spec=ClientSession)
    response = MetricFragment(
        select_fragment="SELECT ...",
        from_fragment="FROM ...",
        where_fragment="WHERE ...",
        group_by_fragment="",
        normalized_sql="SELECT 1",
        normalized=True,
    )
    session.call_tool.return_value = CallToolResult(
        content=[], structured_content=response.model_dump(mode="json")
    )

    @asynccontextmanager
    async def factory() -> AsyncIterator[ClientSession]:
        yield session

    client = McpClient(settings, session_factory=factory)
    try:
        args = metric_args()
        assert (
            await client.resolve_metric(args, deadline=Deadline(time.monotonic() + 5)) == response
        )
        session.call_tool.assert_awaited_once_with(
            "resolve_metric", arguments=args.model_dump(mode="json")
        )
    finally:
        await client.aclose()


def test_breaker_recovery_single_probe() -> None:
    now = [0.0]
    breaker = CircuitBreaker(clock=lambda: now[0])
    for _ in range(5):
        breaker.enter()
        breaker.failed()
    with pytest.raises(McpUnavailableError):
        breaker.enter()
    now[0] = 30
    assert breaker.enter()
    with pytest.raises(McpUnavailableError):
        breaker.enter()
    breaker.success()
    assert not breaker.enter()


async def test_session_reused_and_closed_same_task(settings: MCPSettings) -> None:
    session = AsyncMock(spec=ClientSession)
    session.call_tool.return_value = success()
    owners = []

    @asynccontextmanager
    async def factory() -> AsyncIterator[ClientSession]:
        owners.append(asyncio.current_task())
        try:
            yield session
        finally:
            owners.append(asyncio.current_task())

    client = McpClient(settings, session_factory=factory)
    await client.connect()
    try:
        for _ in range(2):
            result = await client.call_tool(
                "execute_readonly_query",
                QueryArguments(sql="SELECT 1"),
                deadline=Deadline(time.monotonic() + 5),
            )
            assert result.rows == [[1]]
    finally:
        await client.aclose()
    assert len(owners) == TWO
    assert owners[0] is owners[1]


async def test_request_id_is_sent_per_tool_call(settings: MCPSettings) -> None:
    session = AsyncMock(spec=ClientSession)
    session.call_tool.return_value = success()

    @asynccontextmanager
    async def factory() -> AsyncIterator[ClientSession]:
        yield session

    client = McpClient(settings, session_factory=factory)
    try:
        for value in ("a" * 32, "b" * 32):
            token = correlation_id.set(value)
            try:
                await client.call_tool(
                    "execute_readonly_query",
                    QueryArguments(sql="SELECT 1"),
                    deadline=Deadline(time.monotonic() + 5),
                )
            finally:
                correlation_id.reset(token)
        assert [
            call.kwargs["meta"]["insightpilot/request_id"]
            for call in session.call_tool.await_args_list
        ] == [
            "a" * 32,
            "b" * 32,
        ]
    finally:
        await client.aclose()


@pytest.mark.parametrize("code", ["MCP_POLICY_REJECTED", "SQL_TIMEOUT"])
async def test_nonretryable_never_affects_breaker(settings: MCPSettings, code: str) -> None:
    session = AsyncMock(spec=ClientSession)
    session.call_tool.return_value = failure(code)

    @asynccontextmanager
    async def factory() -> AsyncIterator[ClientSession]:
        yield session

    client = McpClient(settings, session_factory=factory)
    try:
        with pytest.raises((McpPolicyRejected, SqlTimeoutError)):
            await client.call_tool(
                "execute_readonly_query",
                QueryArguments(sql="SELECT 1"),
                deadline=Deadline(time.monotonic() + 5),
            )
        assert session.call_tool.await_count == 1
        assert client.breaker.failures == 0
    finally:
        await client.aclose()


async def test_unavailable_retried_twice_counted_once(settings: MCPSettings) -> None:
    session = AsyncMock(spec=ClientSession)
    session.call_tool.return_value = failure("MCP_UNAVAILABLE")

    @asynccontextmanager
    async def factory() -> AsyncIterator[ClientSession]:
        yield session

    client = McpClient(settings, session_factory=factory)
    try:
        with pytest.raises(McpUnavailableError):
            await client.call_tool(
                "execute_readonly_query",
                QueryArguments(sql="SELECT 1"),
                deadline=Deadline(time.monotonic() + 5),
            )
        assert session.call_tool.await_count == TWO
        assert client.breaker.failures == 1
    finally:
        await client.aclose()


async def test_expired_deadline_no_connection(settings: MCPSettings) -> None:
    client = McpClient(settings)
    with pytest.raises(DeadlineExceededError):
        await client.call_tool(
            "execute_readonly_query", QueryArguments(sql="SELECT 1"), deadline=Deadline(0)
        )
    assert client._owner is None


async def test_disconnect_rebuilds_session(settings: MCPSettings) -> None:
    first, second = AsyncMock(spec=ClientSession), AsyncMock(spec=ClientSession)
    first.call_tool.side_effect = ConnectionError()
    second.call_tool.return_value = success()
    sessions = iter((first, second))
    closed = []

    @asynccontextmanager
    async def factory() -> AsyncIterator[ClientSession]:
        session = next(sessions)
        try:
            yield session
        finally:
            closed.append(session)

    client = McpClient(settings, session_factory=factory)
    try:
        result = await client.call_tool(
            "execute_readonly_query",
            QueryArguments(sql="SELECT 1"),
            deadline=Deadline(time.monotonic() + 5),
        )
        assert result.rows == [[1]]
        assert first in closed
        assert second.call_tool.await_count == 1
    finally:
        await client.aclose()
    assert second in closed


async def test_cancellation_closes_owner(settings: MCPSettings) -> None:
    session = AsyncMock(spec=ClientSession)
    started, closed = asyncio.Event(), asyncio.Event()

    async def hang(*args: object, **kwargs: object) -> None:
        started.set()
        await asyncio.Event().wait()

    session.call_tool.side_effect = hang

    @asynccontextmanager
    async def factory() -> AsyncIterator[ClientSession]:
        try:
            yield session
        finally:
            closed.set()

    client = McpClient(settings, session_factory=factory)
    async with asyncio.TaskGroup() as group:
        task = group.create_task(
            client.call_tool(
                "execute_readonly_query",
                QueryArguments(sql="SELECT 1"),
                deadline=Deadline(time.monotonic() + 5),
            )
        )
        await started.wait()
        task.cancel()
    await client.aclose()
    assert closed.is_set()
    assert client.breaker.failures == 0


async def test_deadline_includes_backoff(settings: MCPSettings) -> None:
    session = AsyncMock(spec=ClientSession)
    session.call_tool.return_value = failure("MCP_UNAVAILABLE")

    @asynccontextmanager
    async def factory() -> AsyncIterator[ClientSession]:
        yield session

    client = McpClient(settings, session_factory=factory)
    try:
        with pytest.raises(DeadlineExceededError):
            await client.call_tool(
                "execute_readonly_query",
                QueryArguments(sql="SELECT 1"),
                deadline=Deadline(time.monotonic() + 0.05),
            )
        assert session.call_tool.await_count == 1
        assert client.breaker.failures == 0
    finally:
        await client.aclose()


@pytest.mark.parametrize("status", [401, 403, 422, 429, 500, 502, 503, 504])
def test_http_failures_use_status_not_prose(status: int) -> None:

    response = httpx2.Response(status, request=httpx2.Request("POST", "http://mcp/mcp"))
    error = transport_failure(
        httpx2.HTTPStatusError(
            "misleading policy rejected text", request=response.request, response=response
        )
    )
    assert error.retryable is (status >= HTTPStatus.INTERNAL_SERVER_ERROR)


def test_sdk_closed_connection_is_retryable() -> None:

    assert transport_failure(MCPError(code=CONNECTION_CLOSED, message="arbitrary")).retryable
    assert not transport_failure(RuntimeError("connection closed")).retryable


@pytest.mark.parametrize("case", ["success", "drift", "metadata", "timeout", "invalid", "pii"])
async def test_schema_calls_share_session_and_typed_nonretryable_errors(
    settings: MCPSettings, case: str
) -> None:
    response = business_schema()
    raw = CallToolResult(content=[], structured_content=response.model_dump(mode="json"))
    errors = {
        "drift": SchemaDriftError,
        "metadata": SchemaMetadataError,
        "timeout": SqlTimeoutError,
        "invalid": McpResultError,
        "pii": McpResultError,
    }
    if case in {"drift", "metadata"}:
        raw = CallToolResult(
            content=[],
            is_error=True,
            structured_content={
                "schema_version": 1,
                "code": "SCHEMA_DRIFT" if case == "drift" else "SCHEMA_METADATA_INVALID",
                "message": "ignored error prose",
            },
        )
    elif case == "timeout":
        raw = failure("SQL_TIMEOUT")
    elif case == "invalid":
        raw.structured_content = {"schema_version": 999}
    elif case == "pii":
        column = next(
            c for t in raw.structured_content["tables"] for c in t["columns"] if c["is_pii"]
        )
        column["sample_values"] = ["private"]
    session = AsyncMock(spec=ClientSession)
    session.call_tool.return_value = raw

    @asynccontextmanager
    async def factory() -> AsyncIterator[ClientSession]:
        yield session

    client = McpClient(settings, session_factory=factory)
    try:
        args = GetSchemaArgs(include_samples=True)
        if case == "success":
            assert (
                await client.get_schema(args, deadline=Deadline(time.monotonic() + 5)) == response
            )
        else:
            with pytest.raises(errors[case]):
                await client.get_schema(args, deadline=Deadline(time.monotonic() + 5))
        session.call_tool.assert_awaited_once_with(
            "get_schema", arguments=args.model_dump(mode="json")
        )
        assert client.breaker.failures == 0
    finally:
        await client.aclose()
