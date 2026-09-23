"""Exercise adapters at SDK boundaries, including cleanup and typed failures."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import OperationalError

from app.clients.mcp_client import McpClient
from app.core.config_models import Settings
from app.core.errors import HealthProbeError
from app.db.session import Database
from app.services.health import MCPProbe, PostgreSQLProbe


async def test_postgresql_uses_async_engine_and_disposes_after_failure(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    connection = AsyncMock()
    connection.execute.side_effect = OperationalError("SELECT 1", None, OSError("private"))
    engine = MagicMock()
    engine.connect.return_value.__aenter__ = AsyncMock(return_value=connection)
    engine.connect.return_value.__aexit__ = AsyncMock(return_value=None)
    engine.dispose = AsyncMock()
    factory = MagicMock(return_value=engine)
    monkeypatch.setattr("app.db.session.create_async_engine", factory)
    database = Database(settings.database)
    database.start()
    probe = PostgreSQLProbe(database)
    with pytest.raises(HealthProbeError):
        await probe.check()
    connection.execute.side_effect = None
    await probe.check()
    assert str(connection.execute.call_args.args[0]) == "SELECT 1"
    factory.assert_called_once_with(
        settings.database.app_url,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=5,
        pool_recycle=1800,
        pool_timeout=30,
        echo=False,
        hide_parameters=True,
        connect_args={"timeout": 5, "command_timeout": 10},
    )
    await probe.aclose()
    engine.dispose.assert_not_awaited()
    await database.aclose()
    engine.dispose.assert_awaited_once()


@pytest.mark.parametrize("mode", ["success", "failure", "cancel"])
async def test_mcp_probe_borrows_client_without_closing_it(mode: str) -> None:
    client = AsyncMock(spec=McpClient)
    if mode == "failure":
        client.health.side_effect = OSError("private readiness failure")
    if mode == "cancel":
        client.health.side_effect = asyncio.CancelledError
    probe = MCPProbe(client)
    if mode == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await probe.check()
    elif mode != "success":
        with pytest.raises(HealthProbeError):
            await probe.check()
    else:
        await probe.check()
    client.health.assert_awaited_once()
    await probe.aclose()
    client.aclose.assert_not_awaited()
