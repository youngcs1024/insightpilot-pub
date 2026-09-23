"""Real MCP HTTP and PostgreSQL evidence for the Step 5.3 audit boundary."""

import asyncio
import hashlib
import json
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from uuid import uuid4

import asyncpg
import httpx2
import pytest
from asgi_correlation_id import correlation_id

from app.clients.mcp_client import McpClient
from app.core.config_models import MCPSettings
from app.core.deadline import Deadline
from app.core.errors import McpPolicyRejected, SqlExecutionError, SqlTimeoutError
from app.schemas.schema_tools import GetSchemaArgs
from scripts.deployment import Command, execute
from tests.database_support import DatabaseStack
from tests.integration.mcp_support import client, mcp_endpoint, query
from tests.metric_tool_support import metric_args

pytestmark = pytest.mark.integration
__all__ = ["client", "mcp_endpoint"]


@contextmanager
def request_id() -> Iterator[str]:
    """Assign a unique existing API-style correlation ID to one logical call."""
    value = uuid4().hex
    token = correlation_id.set(value)
    try:
        yield value
    finally:
        correlation_id.reset(token)


@asynccontextmanager
async def operator(stack: DatabaseStack) -> AsyncIterator[asyncpg.Connection]:
    """Use the test operator solely to inspect the private audit schema."""
    conn = await asyncpg.connect(
        host="127.0.0.1",
        port=stack.settings.db_host_port,
        database="insightpilot_business",
        user="postgres",
        password=stack.settings.postgres_superuser_password.get_secret_value(),
        timeout=5,
        command_timeout=5,
    )
    try:
        yield conn
    finally:
        await conn.close(timeout=5)


async def audit_row(stack: DatabaseStack, value: str) -> asyncpg.Record:
    """Read one request's latest row without depending on unrelated test traffic."""
    async with operator(stack) as conn:
        row = await conn.fetchrow(
            "SELECT * FROM mcp.audit_log WHERE correlation_id=$1 ORDER BY id DESC LIMIT 1", value
        )
    assert row is not None
    return row


async def test_successful_call_audited(client: McpClient, database_stack: DatabaseStack) -> None:
    with request_id() as value:
        result = await query(client, "SELECT 11 AS n")
    row = await audit_row(database_stack, value)
    assert result.rows == [[11]]
    assert row["caller"] == "insightpilot-api"
    assert row["tool"] == "execute_readonly_query"
    assert row["outcome"] == "ok"
    assert row["rows_returned"] == 1
    assert row["duration_ms"] >= 0


async def test_rejected_call_audited_with_reasons(
    client: McpClient, database_stack: DatabaseStack
) -> None:
    with request_id() as value, pytest.raises(McpPolicyRejected):
        await query(client, "INSERT INTO biz.regions(region_id) VALUES(99)")
    row = await audit_row(database_stack, value)
    assert row["outcome"] == "policy_rejected"
    assert "write_operation" in json.loads(row["reject_reasons"])
    assert row["rows_returned"] is None


async def test_timeout_audited(client: McpClient, database_stack: DatabaseStack) -> None:
    sql = (
        "SELECT sum(a.i * b.i) FROM generate_series(1,100000) a(i) "
        "CROSS JOIN generate_series(1,100000) b(i)"
    )
    with request_id() as value, pytest.raises(SqlTimeoutError):
        await query(client, sql)
    row = await audit_row(database_stack, value)
    assert row["outcome"] == "timeout"
    assert row["sql_text"] == sql


async def test_sql_text_stored_for_rejections(
    client: McpClient, database_stack: DatabaseStack
) -> None:
    sql = "SELECT * FROM pg_catalog.pg_user WHERE usename = 'audit-sentinel'"
    with request_id() as value, pytest.raises(McpPolicyRejected):
        await query(client, sql)
    row = await audit_row(database_stack, value)
    assert row["sql_text"] == sql
    assert row["outcome"] == "policy_rejected"


async def test_arguments_hashed_not_stored(
    client: McpClient, database_stack: DatabaseStack
) -> None:
    args = GetSchemaArgs(tables=["secret_audit_sentinel"])
    with request_id() as value:
        response = await client.get_schema(args, deadline=Deadline(time.monotonic() + 20))
    row = await audit_row(database_stack, value)
    canonical = json.dumps(args.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    assert response.rejected == ["secret_audit_sentinel"]
    assert row["tool"] == "get_schema"
    assert row["outcome"] == "policy_rejected"
    assert row["arguments_sha256"] == hashlib.sha256(canonical.encode()).hexdigest()
    assert row["reject_reasons"] == '["table_not_allowed"]'
    assert "secret_audit_sentinel" not in json.dumps(dict(row), default=str)
    assert row["sql_text"] is None


async def test_correlation_id_propagated(
    client: McpClient, database_stack: DatabaseStack
) -> None:
    with request_id() as value:
        await query(client, "SELECT 12")
    row = await audit_row(database_stack, value)
    assert row["correlation_id"] == value


async def test_audit_role_cannot_select(
    database_stack: DatabaseStack, mcp_endpoint: MCPSettings
) -> None:
    conn = await asyncpg.connect(
        host="127.0.0.1",
        port=database_stack.settings.db_host_port,
        database="insightpilot_business",
        user="mcp_audit",
        password=database_stack.settings.bootstrap.audit_password.get_secret_value(),
        timeout=5,
        command_timeout=5,
    )
    try:
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await conn.fetch("SELECT * FROM mcp.audit_log LIMIT 1")
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await conn.execute("UPDATE mcp.audit_log SET tool='tampered'")
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await conn.execute("DELETE FROM mcp.audit_log")
        with pytest.raises(asyncpg.InsufficientPrivilegeError):
            await conn.fetch("SELECT * FROM biz.regions LIMIT 1")
    finally:
        await conn.close(timeout=5)


async def test_audit_failure_does_not_fail_tool_call(
    client: McpClient, mcp_endpoint: MCPSettings, database_stack: DatabaseStack
) -> None:
    health = str(mcp_endpoint.base_url).removesuffix("/mcp") + "/health"
    async with httpx2.AsyncClient(timeout=5, trust_env=False) as http:
        before = (await http.get(health)).json()["audit_write_failures_total"]
        async with operator(database_stack) as conn:
            await conn.execute("REVOKE INSERT ON mcp.audit_log FROM mcp_audit")
        try:
            with request_id():
                assert (await query(client, "SELECT 13")).rows == [[13]]
            after = (await http.get(health)).json()["audit_write_failures_total"]
            assert after == before + 1
        finally:
            async with operator(database_stack) as conn:
                await conn.execute("GRANT INSERT ON mcp.audit_log TO mcp_audit")


async def test_execution_error_audited(client: McpClient, database_stack: DatabaseStack) -> None:
    with request_id() as value, pytest.raises(SqlExecutionError):
        await query(client, "SELECT missing_column")
    assert (await audit_row(database_stack, value))["outcome"] == "execution_error"


async def test_invalid_arguments_audited(
    client: McpClient, database_stack: DatabaseStack
) -> None:
    with request_id() as value:
        session = await client._get_session()
        result = await session.call_tool(
            "execute_readonly_query",
            arguments={"sql": "SELECT 1", "max_rows": 0},
            meta={"insightpilot/request_id": value},
        )
    assert result.is_error
    row = await audit_row(database_stack, value)
    assert row["outcome"] == "policy_rejected"
    assert json.loads(row["reject_reasons"]) == ["invalid_arguments"]


async def test_metric_tool_audited(client: McpClient, database_stack: DatabaseStack) -> None:
    with request_id() as value:
        await client.resolve_metric(metric_args(), deadline=Deadline(time.monotonic() + 20))
    row = await audit_row(database_stack, value)
    assert row["tool"] == "resolve_metric"
    assert row["outcome"] == "ok"
    assert row["sql_text"] is None


async def test_concurrent_correlation_ids_do_not_cross(
    client: McpClient, database_stack: DatabaseStack
) -> None:
    async def call(sql: str) -> str:
        with request_id() as value:
            await query(client, sql)
        return value

    first, second = await asyncio.gather(call("SELECT 21"), call("SELECT 22"))
    assert first != second
    rows = await asyncio.gather(audit_row(database_stack, first), audit_row(database_stack, second))
    assert [row["sql_text"] for row in rows] == ["SELECT 21", "SELECT 22"]


async def test_rebootstrap_preserves_audit_rows(
    client: McpClient, database_stack: DatabaseStack
) -> None:
    with request_id() as value:
        await query(client, "SELECT 31")
    before = (await audit_row(database_stack, value))["id"]
    call = database_stack.call.model_copy(update={"command": Command.BOOTSTRAP, "arguments": []})
    await asyncio.to_thread(execute, database_stack.docker, database_stack.settings, call)
    after = (await audit_row(database_stack, value))["id"]
    assert after == before
