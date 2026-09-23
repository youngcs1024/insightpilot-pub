"""Server policy, refresh and readiness contracts with scripted physical metadata."""

# ruff: noqa: PLR2004 -- fixed protocol bounds and HTTP statuses.

import asyncio
import json
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from mcp.server.mcpserver.exceptions import ToolError
from structlog.testing import capture_logs

from app.core.errors import (
    McpUnavailableError,
    SchemaDriftError,
    SchemaMetadataError,
    SqlTimeoutError,
)
from app.schemas.schema_catalog import BUSINESS_TABLES
from app.schemas.schema_tools import GetSchemaArgs
from mcp_server.config import McpServerSettings
from mcp_server.server import create_server, error_result
from mcp_server.tools.get_schema import SchemaTool
from tests.schema_tool_support import (
    ARTIFACT,
    ScriptedSchemaReader,
    authored_catalog,
    write_artifact,
)


async def test_returns_all_allowlisted_tables() -> None:
    response = await SchemaTool(ScriptedSchemaReader(), ARTIFACT).read(GetSchemaArgs())
    assert [t.table_name for t in response.tables] == list(BUSINESS_TABLES)
    assert all(not c.sample_values for t in response.tables for c in t.columns)
    assert "Examples:" not in response.rendered
    assert response.notes
    assert not response.rejected


async def test_partial_and_total_rejections_are_visible_without_logging_input() -> None:
    tool = SchemaTool(ScriptedSchemaReader(), ARTIFACT)
    with capture_logs() as logs:
        response = await tool.read(
            GetSchemaArgs(tables=["biz.orders", "pg_catalog.pg_user", "SECRET_SENTINEL"])
        )
    assert [t.table_name for t in response.tables] == ["biz.orders"]
    assert response.rejected == ["pg_catalog.pg_user", "SECRET_SENTINEL"]
    assert "SECRET_SENTINEL" not in json.dumps(logs)
    assert any(row["event"] == "schema_tables_rejected" for row in logs)
    empty = await tool.read(GetSchemaArgs(tables=["pg_catalog.pg_user"]))
    assert empty.tables == []
    assert empty.rendered == ""
    assert empty.notes == []
    assert empty.rejected == ["pg_catalog.pg_user"]


async def test_pii_column_has_no_sample_or_allowed_values(tmp_path: Path) -> None:
    catalog = authored_catalog()
    phone = next(c for t in catalog.tables for c in t.columns if c.column_name == "phone")
    phone.sample_values = ["PII_SAMPLE"]
    phone.allowed_values = {"PII_ALLOWED": "private"}
    path = tmp_path / "schema.json"
    write_artifact(path, catalog)
    with capture_logs() as logs:
        response = await SchemaTool(ScriptedSchemaReader(), path).read(
            GetSchemaArgs(include_samples=True)
        )
    for source in (response.model_dump_json(), response.rendered, json.dumps(logs)):
        assert "PII_SAMPLE" not in source
        assert "PII_ALLOWED" not in source
    column = next(c for t in response.tables for c in t.columns if c.column_name == "phone")
    assert column.description
    assert column.sql_type
    assert column.sample_values == []
    assert column.allowed_values == {}


async def test_include_samples_capped_at_three_and_text_at_fifty(tmp_path: Path) -> None:
    catalog = authored_catalog()
    catalog.tables[0].columns[0].sample_values = ["x" * 90, "two", "three", "four"]
    path = tmp_path / "schema.json"
    write_artifact(path, catalog)
    response = await SchemaTool(ScriptedSchemaReader(), path).read(
        GetSchemaArgs(include_samples=True)
    )
    assert response.tables[0].columns[0].sample_values == ["x" * 50, "two", "three"]
    assert "two | three" in response.rendered
    assert "four" not in response.rendered
    assert "x" * 51 not in response.rendered


async def test_rendered_text_matches_expected_shape_and_selection_order() -> None:
    tool = SchemaTool(ScriptedSchemaReader(), ARTIFACT)
    response = await tool.read(GetSchemaArgs(tables=["biz.orders", "biz.regions", "biz.orders"]))
    expected = await tool.read(GetSchemaArgs(tables=["biz.regions", "biz.orders"]))
    assert response == expected
    for marker in (
        "Table: biz.orders",
        "Columns:",
        "[PK]",
        "[FK →",
        "[NOT NULL]",
        "cancelled=已取消",
        "Notes:",
        "Asia/Shanghai",
        "GMV=",
    ):
        assert marker in response.rendered
    assert "Table: biz.customers" not in response.rendered


async def test_cache_coalesces_refresh_and_obeys_ttl_and_revision() -> None:
    reader = ScriptedSchemaReader()
    now = [0.0]
    tool = SchemaTool(reader, ARTIFACT, ttl_s=1, clock=lambda: now[0])
    responses = await asyncio.gather(*(tool.read(GetSchemaArgs()) for _ in range(5)))
    assert all(response == responses[0] for response in responses)
    assert reader.calls.count(None) == 1
    now[0] = 2
    await tool.read(GetSchemaArgs())
    assert reader.calls[-1] is None
    reader.source.revision = "new-business-revision"
    changed = await tool.read(GetSchemaArgs())
    assert changed.business_revision == "new-business-revision"
    await tool.read(GetSchemaArgs(refresh=True))
    assert reader.calls[-1] is None


async def test_forced_read_detects_ddl_without_revision_change() -> None:
    reader = ScriptedSchemaReader()
    tool = SchemaTool(reader, ARTIFACT)
    await tool.read(GetSchemaArgs())
    reader.source.tables[0].columns[0].nullable = True
    with pytest.raises(SchemaDriftError) as caught:
        await tool.read(GetSchemaArgs(refresh=True))
    assert caught.value.report.differences
    assert tool._live is None
    result = error_result(caught.value)
    assert result.is_error
    assert result.structured_content["code"] == "SCHEMA_DRIFT"
    assert all(
        d["expected"] is None and d["actual"] is None
        for d in result.structured_content["report"]["differences"]
    )


@pytest.mark.parametrize("failure", [McpUnavailableError(), SqlTimeoutError()])
async def test_refresh_failure_invalidates_cache(failure: Exception) -> None:
    reader = ScriptedSchemaReader()
    tool = SchemaTool(reader, ARTIFACT)
    await tool.read(GetSchemaArgs())
    reader.error = failure
    with pytest.raises(type(failure)):
        await tool.read(GetSchemaArgs())
    assert tool._live is None
    reader.error = None
    await tool.read(GetSchemaArgs())
    assert reader.calls[-1] is None


async def test_lock_and_read_share_one_typed_timeout() -> None:
    reader = ScriptedSchemaReader()
    reader.database.settings.operation_timeout_s = 0.01
    reader.delay = 1
    tool = SchemaTool(reader, ARTIFACT)
    with pytest.raises(SqlTimeoutError):
        await tool.read(GetSchemaArgs())
    assert tool._live is None


@pytest.mark.parametrize("corrupt", [False, True])
async def test_invalid_artifact_never_reads_database(tmp_path: Path, corrupt: bool) -> None:
    path = tmp_path / "schema.json"
    if corrupt:
        path.write_text("invalid")
    reader = ScriptedSchemaReader()
    tool = SchemaTool(reader, path)
    with pytest.raises(SchemaMetadataError):
        await tool.read(GetSchemaArgs())
    assert not reader.calls


async def test_descriptor_exposes_new_tool_and_retires_internal_tool() -> None:
    server = create_server(
        McpServerSettings(
            _env_file=None, business={"password": "test-only"}, audit={"password": "test-only"}, mcp={"auth_token": "test-only"}
        )
    )
    tools = {tool.name: tool for tool in await server.list_tools()}
    assert set(tools) == {"execute_readonly_query", "get_schema", "resolve_metric"}
    descriptor = tools["get_schema"].model_dump(mode="json", by_alias=True)
    props = descriptor["inputSchema"]["properties"]
    assert props["include_samples"]["default"] is False
    assert props["refresh"]["default"] is False
    assert props["tables"]["default"] is None
    assert descriptor["outputSchema"]["properties"]["rejected"]
    with pytest.raises(ToolError):
        await server.call_tool("get_schema", {"tables": []})


async def test_stale_artifact_reported_in_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    reader = ScriptedSchemaReader()
    monkeypatch.setattr("mcp_server.server.BusinessSchemaReader", Mock(return_value=reader))
    server = create_server(
        McpServerSettings(
            _env_file=None, business={"password": "test-only"}, audit={"password": "test-only"}, mcp={"auth_token": "test-only"}
        )
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=server.streamable_http_app()), base_url="http://test"
    ) as client:
        assert (await client.get("/ready")).status_code == 200
        reader.source.tables[0].columns[0].sql_type = "text"
        response = await client.get("/ready")
        assert response.status_code == 503
        assert response.json() == {"ready": False}
        assert (await client.get("/health")).status_code == 200
