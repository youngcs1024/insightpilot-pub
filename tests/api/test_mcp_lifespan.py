"""MCP startup outages degrade readiness without preventing API liveness."""

from unittest.mock import AsyncMock

import httpx

from app.application import create_app
from app.clients.mcp_client import McpClient
from app.core.config_models import Settings
from app.core.errors import McpUnavailableError
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
