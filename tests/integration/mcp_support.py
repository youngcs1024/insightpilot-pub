"""Independent MCP process and business database fixtures, imported explicitly."""

import os
import secrets
import socket
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from subprocess import DEVNULL, Popen, TimeoutExpired

import pytest
from pydantic import SecretStr

from alembic import command
from app.clients.mcp_client import McpClient
from app.core.config_models import MCPSettings
from app.core.deadline import Deadline
from app.schemas.mcp import QueryArguments, QueryResultPayload
from mcp_server.config import BusinessSettings
from mcp_server.db import BusinessDatabase
from scripts.migrate_all import migration_config
from scripts.migration_settings import MigrationTarget
from tests.database_support import DatabaseStack

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def business_tables(database_stack: DatabaseStack) -> None:
    """Use the real business migration rather than a policy-exempt probe relation."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("IP_ENVIRONMENT", "test")
        patch.setenv("IP_MIGRATION__HOST", "127.0.0.1")
        patch.setenv("IP_MIGRATION__PORT", str(database_stack.settings.db_host_port))
        patch.setenv("IP_MIGRATION__USER", "postgres")
        patch.setenv(
            "IP_MIGRATION__PASSWORD",
            database_stack.settings.postgres_superuser_password.get_secret_value(),
        )
        command.upgrade(migration_config(MigrationTarget.BUSINESS), "head")


@pytest.fixture
async def business(
    database_stack: DatabaseStack, business_tables: None
) -> AsyncIterator[BusinessDatabase]:
    db = BusinessDatabase(
        BusinessSettings(
            host="127.0.0.1",
            port=database_stack.settings.db_host_port,
            password=database_stack.settings.bootstrap.mcp_password,
        )
    )
    await db.start()
    try:
        yield db
    finally:
        await db.aclose()


@pytest.fixture(scope="module")
def mcp_endpoint(database_stack: DatabaseStack) -> Iterator[MCPSettings]:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    token = secrets.token_urlsafe(32)
    env = {key: value for key, value in os.environ.items() if not key.startswith(("IP_", "PYTHON"))}
    env.update(
        {
            "IP_BUSINESS__HOST": "127.0.0.1",
            "IP_BUSINESS__PORT": str(database_stack.settings.db_host_port),
            "IP_BUSINESS__PASSWORD": database_stack.settings.bootstrap.mcp_password.get_secret_value(),
            "IP_MCP__HOST": "127.0.0.1",
            "IP_MCP__PORT": str(port),
            "IP_MCP__AUTH_TOKEN": token,
        }
    )
    process = Popen(  # noqa: S603 -- fixed module with credential-scoped test env.
        [str(ROOT / "mcp_server/.venv/bin/python"), "-m", "mcp_server.server"],
        cwd=ROOT,
        env=env,
        stdout=DEVNULL,
        stderr=DEVNULL,
    )
    try:  # noqa: PLR1702 -- bounded subprocess fixture with guaranteed cleanup.
        for _ in range(100):
            assert process.poll() is None, "MCP subprocess failed during startup"
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except OSError:
                time.sleep(0.05)
        else:
            pytest.fail("MCP startup timeout")
        yield MCPSettings(base_url=f"http://127.0.0.1:{port}/mcp", auth_token=SecretStr(token))
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


@pytest.fixture
async def client(mcp_endpoint: MCPSettings) -> AsyncIterator[McpClient]:
    client = McpClient(mcp_endpoint)
    await client.connect()
    try:
        yield client
    finally:
        await client.aclose()


async def query(client: McpClient, sql: str, cap: int = 1000) -> QueryResultPayload:
    return await client.call_tool(
        "execute_readonly_query",
        QueryArguments(sql=sql, max_rows=cap),
        deadline=Deadline(time.monotonic() + 20),
    )
