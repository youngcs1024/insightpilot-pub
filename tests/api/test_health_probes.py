"""Exercise adapters at SDK boundaries, including cleanup and typed failures."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.exc import OperationalError

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


@pytest.mark.parametrize("mode", ["success", "initialize_failure", "list_failure", "cancel"])
async def test_mcp_sdk_session_scope_and_no_business_calls(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    closed: list[str] = []
    session = AsyncMock()
    if mode == "initialize_failure":
        session.initialize.side_effect = OSError("private initialization failure")
    if mode == "list_failure":
        session.list_tools.side_effect = OSError("private tools failure")
    if mode == "cancel":
        session.list_tools.side_effect = asyncio.CancelledError

    @asynccontextmanager
    async def transport(*args: object, **kwargs: object) -> AsyncIterator[tuple[None, None]]:
        try:
            yield None, None
        finally:
            closed.append("transport")

    @asynccontextmanager
    async def session_context(*args: object, **kwargs: object) -> AsyncIterator[AsyncMock]:
        try:
            yield session
        finally:
            closed.append("session")

    http = AsyncMock()
    http.__aenter__.return_value = http
    factory = MagicMock(return_value=http)
    monkeypatch.setattr("app.services.health.httpx2.AsyncClient", factory)
    monkeypatch.setattr("app.services.health.streamable_http_client", transport)
    monkeypatch.setattr("app.services.health.ClientSession", session_context)
    probe = MCPProbe(settings.mcp, settings.health.mcp_timeout_s)
    if mode == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await probe.check()
    elif mode != "success":
        with pytest.raises(HealthProbeError):
            await probe.check()
    else:
        await probe.check()
        session.initialize.assert_awaited_once()
        session.list_tools.assert_awaited_once()
    factory.assert_called_once_with(
        headers={"Authorization": "Bearer " + settings.mcp.auth_token.get_secret_value()},
        timeout=settings.health.mcp_timeout_s,
        trust_env=False,
    )
    session.call_tool.assert_not_called()
    assert closed == ["session", "transport"]
    http.__aexit__.assert_awaited_once()
    await probe.aclose()
