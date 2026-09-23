"""MCP startup outages degrade readiness without preventing API liveness."""

from unittest.mock import AsyncMock

import httpx
import pytest

from app.application import create_app
from app.clients.mcp_client import McpClient
from app.core.config_models import Settings
from app.core.deadline import Deadline
from app.core.errors import McpToolSchemaError, McpUnavailableError
from app.services.health import HealthService
from tests.fakes.health_probe import FakeProbe

OK, UNAVAILABLE = 200, 503


async def test_mcp_startup_failure_degrades_and_cleanup_runs(settings: Settings) -> None:
    database, probe = FakeProbe(), FakeProbe()
    probe.failure = True
    client = AsyncMock(spec=McpClient)
    client.connect.side_effect = McpUnavailableError()
    app = create_app(
        settings,
        health_service=HealthService(database, probe, settings.health),
        mcp_client=client,
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app),
            base_url="http://test",
        ) as http,
    ):
        assert (await http.get("/health")).status_code == OK
        assert (await http.get("/ready")).status_code == UNAVAILABLE
        assert app.state.mcp is client
    client.connect.assert_awaited_once()
    client.aclose.assert_awaited_once()
    assert database.closed
    assert probe.closed


async def test_lifespan_loads_descriptors_once(settings: Settings) -> None:
    client = AsyncMock(spec=McpClient)
    app = create_app(
        settings,
        health_service=HealthService(FakeProbe(), FakeProbe(), settings.health),
        mcp_client=client,
    )
    async with app.router.lifespan_context(app):
        client.connect.assert_awaited_once()
        client.refresh_tools.assert_awaited_once()
        assert isinstance(client.refresh_tools.await_args.kwargs["deadline"], Deadline)


async def test_unsupported_descriptor_prevents_startup(settings: Settings) -> None:
    client = AsyncMock(spec=McpClient)
    client.refresh_tools.side_effect = McpToolSchemaError("new_tool", "inputSchema.oneOf")
    app = create_app(
        settings,
        health_service=HealthService(FakeProbe(), FakeProbe(), settings.health),
        mcp_client=client,
    )
    with pytest.raises(McpToolSchemaError, match="new_tool"):
        async with app.router.lifespan_context(app):
            pytest.fail("Invalid tool schema must prevent startup")
    client.aclose.assert_awaited_once()


async def test_discovery_outage_degrades_without_preventing_liveness(settings: Settings) -> None:
    client = AsyncMock(spec=McpClient)
    client.refresh_tools.side_effect = McpUnavailableError()
    probe = FakeProbe()
    probe.failure = True
    app = create_app(
        settings,
        health_service=HealthService(FakeProbe(), probe, settings.health),
        mcp_client=client,
    )
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as http,
    ):
        assert (await http.get("/health")).status_code == OK
        assert (await http.get("/ready")).status_code == UNAVAILABLE


async def test_readiness_requires_loaded_tool_cache(settings: Settings) -> None:
    loaded = False
    service = HealthService(
        FakeProbe(), FakeProbe(), settings.health, mcp_tools_ready=lambda: loaded
    )
    app = create_app(settings, health_service=service, mcp_client=AsyncMock(spec=McpClient))
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as http,
    ):
        response = await http.get("/ready")
        assert response.status_code == UNAVAILABLE
        assert response.json()["checks"]["mcp"] is False
        loaded = True
        assert (await http.get("/ready")).status_code == OK
