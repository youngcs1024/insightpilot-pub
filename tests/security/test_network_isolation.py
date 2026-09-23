"""Step 5.7 MCP network, authentication and resource-boundary acceptance."""

# ruff: noqa: PLR2004 -- exact security limits and HTTP statuses are the contract.

import json
import os
import secrets
import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import asyncpg
import httpx2
import pytest
import yaml
from asgi_correlation_id import correlation_id
from mcp.types import CallToolResult
from pydantic import BaseModel, SecretStr

from app.clients.mcp_client import McpClient
from app.core.config_models import MCPSettings
from app.core.deadline import Deadline
from app.core.errors import McpRateLimitError
from app.schemas.mcp import (
    RESULT_CEILING,
    McpErrorCode,
    McpErrorPayload,
    PolicyReason,
    QueryArguments,
    QueryWarning,
)
from mcp_server.config import McpServerSettings
from mcp_server.server import MCP_REQUEST_BODY_LIMIT, create_server
from scripts.deployment import Command, execute
from tests.database_support import DatabaseStack
from tests.integration.mcp_support import mcp_endpoint, running_mcp_endpoint

__all__ = ["mcp_endpoint"]
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def compose_stack(database_stack: DatabaseStack) -> DatabaseStack:
    """Bring up the real API and MCP on this test run's isolated Compose project."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    settings = database_stack.settings.model_copy(
        update={
            "api_host_port": port,
            "mcp_auth_token": SecretStr(secrets.token_urlsafe(32)),
            "jwt_secret": SecretStr(secrets.token_urlsafe(48)),
            "llm_api_key": SecretStr("unused-security-test-key"),
        }
    )
    call = database_stack.call.model_copy(
        update={"profiles": ["core"], "command": Command.UP, "arguments": ["-d", "--wait"]}
    )
    execute(database_stack.docker, settings, call)
    return database_stack.model_copy(update={"settings": settings, "call": call})


@pytest.fixture
def rate_endpoint(database_stack: DatabaseStack) -> Iterator[MCPSettings]:
    """Do not consume another module's MCP quota in the sixty-call proof."""
    with running_mcp_endpoint(database_stack) as endpoint:
        yield endpoint


def test_mcp_service_has_no_published_ports() -> None:
    for filename in ("docker-compose.yml", "docker-compose.e2e.yml"):
        compose = yaml.safe_load((ROOT / filename).read_text())
        mcp = compose["services"]["mcp"]
        assert "ports" not in mcp
        assert mcp["networks"] == ["backend"]
        assert compose["networks"]["backend"]["internal"] is True
    dev = yaml.safe_load((ROOT / "docker-compose.dev.yml").read_text())
    assert "mcp" not in dev.get("services", {})


@pytest.mark.integration
def test_mcp_unreachable_from_host(compose_stack: DatabaseStack) -> None:
    with socket.socket() as probe:
        probe.settimeout(1)
        assert probe.connect_ex(("127.0.0.1", 8001)) != 0
    code = """
import urllib.error, urllib.request
try:
    urllib.request.urlopen('http://mcp:8001/mcp', timeout=3)
except urllib.error.HTTPError as exc:
    assert exc.code == 401
    print('internal_mcp_401')
"""
    output = execute(
        compose_stack.docker,
        compose_stack.settings,
        compose_stack.call.model_copy(
            update={"command": Command.EXEC, "arguments": ["-T", "api", "python", "-c", code]}
        ),
    )
    assert "internal_mcp_401" in output


@pytest.mark.parametrize("token", [None, "x" * 31])
def test_short_token_rejected_at_startup(token: str | None, tmp_path: Path) -> None:
    env = {key: value for key, value in os.environ.items() if not key.startswith("IP_")}
    env.update(
        {
            "IP_ENVIRONMENT": "test",
            "IP_BUSINESS__PASSWORD": "synthetic-business-password",
            "IP_AUDIT__PASSWORD": "synthetic-audit-password",
        }
    )
    if token is not None:
        env["IP_MCP__AUTH_TOKEN"] = token
    code = (
        "import sys; from pathlib import Path; "
        "from app.core.settings_base import ProcessSettings; "
        "ProcessSettings.project_root = Path(sys.argv[1]); "
        "from mcp_server.server import main; main()"
    )
    result = subprocess.run(  # noqa: S603 -- fixed entrypoint and isolated synthetic env.
        [sys.executable, "-c", code, str(tmp_path)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode != 0
    assert ("auth_token" if token is not None else "mcp\n  Field required") in result.stderr
    assert token is None or token not in result.stderr


def test_minimum_length_token_accepted() -> None:
    settings = McpServerSettings(
        _env_file=None,
        business={"password": "synthetic-business-password"},
        audit={"password": "synthetic-audit-password"},
        mcp={"auth_token": "x" * 32},
    )
    assert len(settings.mcp.auth_token.get_secret_value()) == 32
    assert create_server(settings) is not None


@pytest.mark.integration
async def test_wrong_token_rejected(mcp_endpoint: MCPSettings) -> None:
    async with httpx2.AsyncClient(timeout=3, trust_env=False) as http:
        for headers in ({}, {"Authorization": "Bearer " + "w" * 32}):
            response = await http.get(str(mcp_endpoint.base_url), headers=headers)
            assert response.status_code == 401


async def audit_outcome(stack: DatabaseStack, request_id: str) -> str:
    """Read a single row with the isolated test operator, never the API credential."""
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
        result = await conn.fetchval(
            "SELECT outcome FROM mcp.audit_log WHERE correlation_id=$1 ORDER BY id DESC LIMIT 1",
            request_id,
        )
        assert isinstance(result, str)
        return result
    finally:
        await conn.close(timeout=5)


class CountingMcpClient(McpClient):
    """Observe transport calls while keeping the real SDK and server in the path."""

    def __init__(self, settings: MCPSettings) -> None:
        super().__init__(settings)
        self.raw_calls = 0

    async def _raw_call(
        self, name: str, args: BaseModel, *, omit_unset: bool = False
    ) -> CallToolResult:
        self.raw_calls += 1
        return await super()._raw_call(name, args, omit_unset=omit_unset)


@pytest.mark.integration
async def test_rate_limit_enforced(
    rate_endpoint: MCPSettings, database_stack: DatabaseStack
) -> None:
    client = CountingMcpClient(rate_endpoint)
    try:
        for _ in range(60):
            result = await client.call_tool(
                "execute_readonly_query",
                QueryArguments(sql="SELECT 1"),
                deadline=Deadline(time.monotonic() + 20),
            )
            assert result.rows == [[1]]
        assert await client.list_tools(deadline=Deadline(time.monotonic() + 20))
        health = str(rate_endpoint.base_url).removesuffix("/mcp") + "/health"
        async with httpx2.AsyncClient(timeout=3, trust_env=False) as http:
            assert (await http.get(health)).status_code == 200
        request_id = uuid4().hex
        token = correlation_id.set(request_id)
        try:
            with pytest.raises(McpRateLimitError):
                await client.call_tool(
                    "execute_readonly_query",
                    QueryArguments(sql="SELECT 1"),
                    deadline=Deadline(time.monotonic() + 20),
                )
        finally:
            correlation_id.reset(token)
        assert client.raw_calls == 61
        assert client.breaker.failures == 0
        assert client.breaker.opened_at is None
        assert await audit_outcome(database_stack, request_id) == "rate_limited"
    finally:
        await client.aclose()


@pytest.mark.integration
async def test_max_rows_clamped_server_side(mcp_endpoint: MCPSettings) -> None:
    client = McpClient(mcp_endpoint)
    try:
        result = await client.call_tool(
            "execute_readonly_query",
            QueryArguments(sql="SELECT * FROM generate_series(1,5001)", max_rows=10_000_000),
            deadline=Deadline(time.monotonic() + 20),
        )
        assert result.row_count == RESULT_CEILING
        assert result.result_truncated
        assert result.warnings == [QueryWarning.ROW_CAP_CLAMPED]
        exact = await client.call_tool(
            "execute_readonly_query",
            QueryArguments(sql="SELECT 1", max_rows=RESULT_CEILING),
            deadline=Deadline(time.monotonic() + 20),
        )
        assert exact.warnings == []
    finally:
        await client.aclose()


@pytest.mark.integration
@pytest.mark.parametrize("streamed", [False, True])
async def test_oversized_request_rejected(mcp_endpoint: MCPSettings, streamed: bool) -> None:
    body = b"x" * (MCP_REQUEST_BODY_LIMIT + 1)

    async def chunks() -> AsyncIterator[bytes]:
        yield body[: MCP_REQUEST_BODY_LIMIT // 2]
        yield body[MCP_REQUEST_BODY_LIMIT // 2 :]

    headers = {"Authorization": "Bearer " + mcp_endpoint.auth_token.get_secret_value()}
    async with httpx2.AsyncClient(timeout=5, trust_env=False) as http:
        response = await http.post(
            str(mcp_endpoint.base_url),
            headers=headers,
            content=chunks() if streamed else body,
        )
    assert response.status_code == 413
    assert "Traceback" not in response.text


async def test_tool_error_contains_no_stack_trace() -> None:
    server = create_server(
        McpServerSettings(
            _env_file=None,
            business={"password": "synthetic-business-password"},
            audit={"password": "synthetic-audit-password"},
            mcp={"auth_token": "x" * 32},
        )
    )
    server.audit.record = AsyncMock()
    invalid = await server.call_tool(
        "execute_readonly_query", {"sql": "SELECT 'PRIVATE_SQL_SENTINEL'", "max_rows": 0}
    )
    assert isinstance(invalid, CallToolResult)
    invalid_error = McpErrorPayload.model_validate(invalid.structured_content)
    assert invalid_error.code is McpErrorCode.POLICY_REJECTED
    assert invalid_error.reasons == [PolicyReason.INVALID_ARGUMENTS]

    async def crash() -> str:
        raise RuntimeError("PRIVATE_EXCEPTION_SENTINEL")

    server.add_tool(crash)
    unexpected = await server.call_tool("crash", {})
    assert isinstance(unexpected, CallToolResult)
    assert McpErrorPayload.model_validate(unexpected.structured_content).code is (
        McpErrorCode.INVALID_RESULT
    )
    for result in (invalid, unexpected):
        rendered = json.dumps(result.model_dump(mode="json"))
        assert "Traceback" not in rendered
        assert "PRIVATE_SQL_SENTINEL" not in rendered
        assert "PRIVATE_EXCEPTION_SENTINEL" not in rendered
        assert "x" * 32 not in rendered


async def test_policy_rejections_consume_tool_quota() -> None:
    server = create_server(
        McpServerSettings(
            _env_file=None,
            business={"password": "synthetic-business-password"},
            audit={"password": "synthetic-audit-password"},
            mcp={"auth_token": "x" * 32},
        )
    )
    server.audit.record = AsyncMock()
    for _ in range(60):
        result = await server.call_tool(
            "execute_readonly_query", {"sql": "SELECT 1", "max_rows": 0}
        )
        assert isinstance(result, CallToolResult)
        assert McpErrorPayload.model_validate(result.structured_content).code is (
            McpErrorCode.POLICY_REJECTED
        )
    limited = await server.call_tool("execute_readonly_query", {"sql": "SELECT 1", "max_rows": 0})
    assert isinstance(limited, CallToolResult)
    assert McpErrorPayload.model_validate(limited.structured_content).code is (
        McpErrorCode.RATE_LIMITED
    )
    assert server.audit.record.await_count == 61
