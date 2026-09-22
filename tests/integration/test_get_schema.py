"""Real MCP HTTP schema boundary, database drift and reproducible application export."""

# ruff: noqa: PLR2004 -- fixed acceptance counts and process/HTTP statuses.

import json
import subprocess
import time
from pathlib import Path

import httpx
import pytest

from app.clients.mcp_client import McpClient
from app.core.config_models import MCPSettings, Settings
from app.core.deadline import Deadline
from app.core.errors import SchemaDriftError
from app.schemas.schema_catalog import BUSINESS_TABLES
from app.schemas.schema_tools import GetSchemaArgs
from tests.database_support import DatabaseStack
from tests.integration.catalog_support import catalog_migrated, cli_environment, client, operator_change
from tests.integration.mcp_support import mcp_endpoint as existing_endpoint
from tests.schema_tool_support import ARTIFACT, ROOT
from tests.seed_support import seed_database_stack

pytestmark = pytest.mark.integration
__all__ = ["catalog_migrated", "client"]
database_stack = seed_database_stack
server_endpoint = existing_endpoint


def budget() -> Deadline:
    return Deadline(time.monotonic() + 20)


async def test_returns_all_allowlisted_tables(client: McpClient) -> None:
    result = await client.get_schema(GetSchemaArgs(), deadline=budget())
    assert [t.table_name for t in result.tables] == list(BUSINESS_TABLES)
    assert sum(len(t.columns) for t in result.tables) == 52
    assert result.schema_version == 1
    assert len(result.metadata_revision) == 64
    assert all(not c.sample_values for t in result.tables for c in t.columns)


async def test_non_allowlisted_table_in_rejected_list(client: McpClient) -> None:
    response = await client.get_schema(GetSchemaArgs(tables=["biz.orders", "pg_catalog.pg_user"]), deadline=budget())
    assert [t.table_name for t in response.tables] == ["biz.orders"]
    assert response.rejected == ["pg_catalog.pg_user"]
    assert "pg_catalog.pg_user" not in response.rendered


async def test_pii_column_has_no_sample_values(client: McpClient) -> None:
    result = await client.get_schema(GetSchemaArgs(include_samples=True), deadline=budget())
    pii = [c for t in result.tables for c in t.columns if c.is_pii]
    assert {c.column_name for c in pii} == {"phone", "email"}
    assert all(not c.sample_values and not c.allowed_values for c in pii)


async def test_include_samples_capped_at_three(client: McpClient) -> None:
    result = await client.get_schema(GetSchemaArgs(include_samples=True), deadline=budget())
    assert any(c.sample_values for t in result.tables for c in t.columns)
    assert all(len(c.sample_values) <= 3 for t in result.tables for c in t.columns)
    assert all(len(value) <= 50 for t in result.tables for c in t.columns for value in c.sample_values)


async def test_rendered_text_matches_expected_shape(client: McpClient) -> None:
    result = await client.get_schema(GetSchemaArgs(), deadline=budget())
    assert result.rendered.count("Table: ") == 8
    for marker in ("[PK]", "[FK →", "[NOT NULL]", "cancelled=已取消", "Asia/Shanghai", "COALESCE", "GMV="):
        assert marker in result.rendered


async def test_stale_artifact_reported_in_ready(
    client: McpClient, server_endpoint: MCPSettings, database_stack: DatabaseStack
) -> None:
    await client.get_schema(GetSchemaArgs(refresh=True), deadline=budget())
    ready = str(server_endpoint.base_url).removesuffix("/mcp") + "/ready"
    async with httpx.AsyncClient(timeout=10) as http:
        assert (await http.get(ready)).status_code == 200
        async with operator_change(
            database_stack, "insightpilot_business",
            ["ALTER TABLE biz.customers ALTER COLUMN registered_at DROP NOT NULL"],
            ["ALTER TABLE biz.customers ALTER COLUMN registered_at SET NOT NULL"],
        ):
            response = await http.get(ready)
            assert response.status_code == 503
            with pytest.raises(SchemaDriftError) as caught:
                await client.get_schema(GetSchemaArgs(), deadline=budget())
            assert caught.value.report.differences
        assert (await http.get(ready)).status_code == 200
    assert (await client.get_schema(GetSchemaArgs(), deadline=budget())).tables


def test_artifact_matches_metadata_table(
    catalog_migrated: None, database_stack: DatabaseStack,
    server_endpoint: MCPSettings, settings: Settings, tmp_path: Path,
) -> None:
    output = tmp_path / "exported.json"
    env = cli_environment(settings, database_stack, server_endpoint)
    command = [str(ROOT / ".venv/bin/python"), "-m", "scripts.export_schema_artifact", "--output", str(output)]
    exported = subprocess.run(command, env=env, capture_output=True, text=True, timeout=45, check=False)  # noqa: S603 -- fixed CLI and fixture-owned output.
    assert exported.returncode == 0, exported.stdout + exported.stderr
    assert output.read_bytes() == ARTIFACT.read_bytes()
    checked = subprocess.run([*command, "--check"], env=env, capture_output=True, text=True, timeout=45, check=False)  # noqa: S603 -- fixed CLI.
    assert checked.returncode == 0, checked.stdout + checked.stderr
    source = json.loads(output.read_text())
    source["tables"][0]["description"] = "stale exported content"
    output.write_text(json.dumps(source))
    rejected = subprocess.run([*command, "--check"], env=env, capture_output=True, text=True, timeout=45, check=False)  # noqa: S603 -- fixed CLI.
    assert rejected.returncode == 1
    assert "SCHEMA_METADATA_INVALID" in rejected.stderr
    assert json.loads(output.read_text())["tables"][0]["description"] == "stale exported content"


def test_dev_cli_reports_partial_rejection(
    catalog_migrated: None, database_stack: DatabaseStack,
    server_endpoint: MCPSettings, settings: Settings,
) -> None:
    result = subprocess.run(  # noqa: S603 -- fixed diagnostic CLI, isolated MCP endpoint.
        [str(ROOT / ".venv/bin/python"), "-m", "scripts.dev_mcp", "get_schema", "--tables", "biz.orders,pg_catalog.pg_user"],
        env=cli_environment(settings, database_stack, server_endpoint),
        capture_output=True, text=True, timeout=45, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["rejected"] == ["pg_catalog.pg_user"]
