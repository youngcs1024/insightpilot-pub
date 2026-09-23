"""Build every descriptor from the locked real MCP process before invoking tools."""

# ruff: noqa: PLR2004 -- fixed eight-table business schema acceptance.

import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from app.application import create_app
from app.clients.mcp_client import McpClient
from app.core.config_models import MCPSettings, Settings
from app.core.deadline import Deadline, bind_deadline, reset_deadline
from app.core.observability import Observability
from app.db.session import Database
from app.services.graph import GraphService
from app.services.health import HealthService
from app.services.llm.service import LlmService
from tests.fakes.health_probe import FakeProbe
from tests.integration import mcp_support
from tests.seed_support import seed_database_stack

pytestmark = pytest.mark.integration
database_stack = seed_database_stack
mcp_endpoint = mcp_support.mcp_endpoint
client = mcp_support.client


def budget() -> Deadline:
    return Deadline(time.monotonic() + 20)


async def test_all_live_descriptors_load_and_schema_arguments(client: McpClient) -> None:
    descriptors = await client.list_tools(deadline=budget())
    tools = await client.refresh_tools(deadline=budget())
    assert {tool.name for tool in tools} == {item.name for item in descriptors}
    assert all(tool.args_schema is not None for tool in tools)
    schema_tool = next(tool for tool in tools if tool.name == "get_schema")

    token = bind_deadline(budget())
    try:
        omitted = json.loads(await schema_tool.ainvoke({}))
        explicit_null = json.loads(await schema_tool.ainvoke({"tables": None}))
        valid = json.loads(await schema_tool.ainvoke({"tables": ["biz.orders"]}))
        invalid = json.loads(await schema_tool.ainvoke({"tables": ["pg_catalog.pg_user"]}))
        with pytest.raises(ValidationError):
            await schema_tool.ainvoke({"tables": []})
    finally:
        reset_deadline(token)

    assert len(omitted["tables"]) == 8
    assert len(explicit_null["tables"]) == 8
    assert [item["table_name"] for item in valid["tables"]] == ["biz.orders"]
    assert invalid["tables"] == []
    assert invalid["rejected"] == ["pg_catalog.pg_user"]


async def test_api_lifespan_loads_every_live_descriptor(
    settings: Settings, mcp_endpoint: MCPSettings
) -> None:
    settings.mcp = mcp_endpoint
    mcp = McpClient(mcp_endpoint)
    service = HealthService(FakeProbe(), FakeProbe(), settings.health)
    app = create_app(
        settings,
        health_service=service,
        mcp_client=mcp,
        database=MagicMock(spec=Database),
        llm_service=AsyncMock(spec=LlmService),
        graph_service=AsyncMock(spec=GraphService),
        observability=AsyncMock(spec=Observability),
    )
    app.state.metrics.validate_startup = AsyncMock()
    app.state.chat.reconcile = AsyncMock()
    async with app.router.lifespan_context(app):
        listed = await mcp.list_tools(deadline=budget())
        assert mcp.tools_loaded
        assert {tool.name for tool in mcp.tools} == {tool.name for tool in listed}
