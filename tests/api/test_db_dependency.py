"""Session DI and safe HTTP translation without external database calls."""

import asyncio
from typing import Annotated
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
import pytest
from fastapi import Depends
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError
from sqlalchemy.ext.asyncio import AsyncSession

from app.application import create_app
from app.core.config_models import Settings
from app.core.errors import UpstreamUnavailableError
from app.db.session import Database, get_session
from app.services.health import HealthService
from tests.fakes.health_probe import FakeProbe


class Result(BaseModel):
    """Synthetic endpoint result for exercising the real dependency."""

    value: int = 1


class UniqueViolationError(Exception):
    """Driver substitute with a structured SQLSTATE, independent of its message."""

    sqlstate = "23505"


@pytest.mark.parametrize("failure", ["pool", "unique", "connection", "success"])
async def test_dependency_translates_errors_and_closes(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    session = AsyncMock(spec=AsyncSession)
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=session)
    context.__aexit__ = AsyncMock(return_value=False)
    factory = MagicMock(return_value=context)
    monkeypatch.setattr("app.db.session.async_sessionmaker", MagicMock(return_value=factory))
    failures = {
        "pool": PoolTimeoutError("private pool diagnostic"),
        "unique": IntegrityError(
            "secret SQL", {"password": "private"}, UniqueViolationError("private")
        ),
        "connection": ConnectionRefusedError("private connection string"),
    }
    session.execute.side_effect = failures.get(failure)
    app = create_app(
        settings, health_service=HealthService(FakeProbe(), FakeProbe(), settings.health)
    )

    @app.get("/db-test")
    async def endpoint(
        db: Annotated[AsyncSession, Depends(get_session, scope="function")],
    ) -> Result:
        await db.execute(text("SELECT 1"))
        return Result()

    request_id = str(uuid4())
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        response = await client.get("/db-test", headers={"X-Request-ID": request_id})
    expected = {"pool": 503, "unique": 409, "connection": 503, "success": 200}
    assert response.status_code == expected[failure]
    assert response.headers["X-Request-ID"] == request_id.replace("-", "")
    if failure != "success":
        assert response.json()["request_id"] == request_id.replace("-", "")
        assert set(response.json()) == {"code", "message", "request_id"}
        assert "private" not in response.text
        assert "secret SQL" not in response.text
    factory.assert_called_once_with()
    context.__aexit__.assert_awaited_once()
    session.commit.assert_not_awaited()


async def test_app_instances_have_independent_pools_and_release_them(settings: Settings) -> None:
    first = create_app(
        settings, health_service=HealthService(FakeProbe(), FakeProbe(), settings.health)
    )
    second = create_app(
        settings, health_service=HealthService(FakeProbe(), FakeProbe(), settings.health)
    )
    one: Database = first.state.database
    two: Database = second.state.database
    assert one is not two
    async with first.router.lifespan_context(first), second.router.lifespan_context(second):
        assert one.engine is not two.engine
    for database in (one, two):
        with pytest.raises(UpstreamUnavailableError):
            _ = database.engine
        with pytest.raises(UpstreamUnavailableError):
            async with database.session():
                pytest.fail("Closed database provided a session")


async def test_partial_startup_and_shutdown_timeout_release_database(settings: Settings) -> None:
    settings.http.shutdown_timeout_s = 0.02
    db_probe, mcp_probe = FakeProbe(), FakeProbe()
    mcp_probe.hang = True
    mcp_probe.close_hang = True
    app = create_app(settings, health_service=HealthService(db_probe, mcp_probe, settings.health))
    database: Database = app.state.database
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.02), app.router.lifespan_context(app):
            pytest.fail("Startup should be cancelled")
    with pytest.raises(UpstreamUnavailableError):
        _ = database.engine


async def test_slow_pool_disposal_is_bounded_and_other_cleanup_runs(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.http.shutdown_timeout_s = 0.02
    engine = MagicMock()
    engine.dispose = AsyncMock(side_effect=lambda: None)

    async def hang() -> None:
        await asyncio.Event().wait()

    engine.dispose.side_effect = hang
    monkeypatch.setattr("app.db.session.create_async_engine", MagicMock(return_value=engine))
    db_probe, mcp_probe = FakeProbe(), FakeProbe()
    app = create_app(settings, health_service=HealthService(db_probe, mcp_probe, settings.health))
    async with asyncio.timeout(1):
        async with app.router.lifespan_context(app):
            assert app.state.ready
    engine.dispose.assert_awaited_once()
    assert db_probe.closed
    assert mcp_probe.closed
