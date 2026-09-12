"""API and lifecycle acceptance with deterministic probes, never external services."""

import asyncio
import json
import subprocess
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from time import monotonic
from uuid import UUID, uuid4

import httpx
import pytest
import structlog
from asgi_correlation_id import correlation_id
from fastapi import Request
from pydantic import BaseModel, ValidationError

from app.application import create_app
from app.core.config_models import HealthSettings, HTTPSettings, Settings
from app.services.health import HealthService
from tests.fakes.health_probe import FakeProbe

OK = 200
UUID_VERSION = 4
CONCURRENT_REQUESTS = 2
PROBE_CALLS = 2
UNAVAILABLE = 503
FORBIDDEN = 400


@asynccontextmanager
async def client_for(
    settings: Settings, db: FakeProbe, mcp: FakeProbe
) -> AsyncIterator[httpx.AsyncClient]:
    service = HealthService(db, mcp, settings.health)
    app = create_app(settings, health_service=service)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        yield client


async def test_health_always_ok(settings: Settings) -> None:
    db, mcp = FakeProbe(), FakeProbe()
    db.failure = mcp.failure = True
    async with client_for(settings, db, mcp) as client:
        response = await client.get("/health")
        assert response.status_code == OK
        assert response.json()["status"] == "ok"
        assert response.json()["version"] == "0.1.0"
        assert response.json()["request_id"] == response.headers["X-Request-ID"]
        assert db.calls == mcp.calls == 1  # Only startup probes; liveness performs no I/O.
    assert db.closed
    assert mcp.closed


async def test_ready_returns_503_when_db_down(settings: Settings) -> None:
    db, mcp = FakeProbe(), FakeProbe()
    db.failure = True
    async with client_for(settings, db, mcp) as client:
        response = await client.get("/ready")
        assert response.status_code == UNAVAILABLE
        assert response.json()["checks"] == {"postgresql": False, "mcp": True}
        assert response.json()["status"] == "degraded"
        assert "secret" not in response.text
        db.failure = False
        response = await client.get("/ready")
        assert response.status_code == OK
        assert response.json()["status"] == "ready"


async def test_mcp_outage_and_recovery(settings: Settings) -> None:
    db, mcp = FakeProbe(), FakeProbe()
    async with client_for(settings, db, mcp) as client:
        assert (await client.get("/ready")).status_code == OK
        mcp.failure = True
        response = await client.get("/ready")
        assert response.status_code == UNAVAILABLE
        assert response.json()["checks"] == {"postgresql": True, "mcp": False}
        mcp.failure = False
        assert (await client.get("/ready")).status_code == OK


@pytest.mark.parametrize("dependency", ["postgresql", "mcp"])
async def test_probe_timeout_is_cancelled(settings: Settings, dependency: str) -> None:
    settings.health.postgresql_timeout_s = settings.health.mcp_timeout_s = 0.02
    db, mcp = FakeProbe(), FakeProbe()
    probe = db if dependency == "postgresql" else mcp
    async with client_for(settings, db, mcp) as client:
        probe.hang = True
        async with asyncio.timeout(1):
            response = await client.get("/ready")
        assert response.status_code == UNAVAILABLE
        assert response.json()["checks"][dependency] is False
        assert probe.cancelled
        assert probe.calls == PROBE_CALLS  # Startup plus one bounded request; no retries.


async def test_checks_run_concurrently(settings: Settings) -> None:
    db, mcp = FakeProbe(), FakeProbe()
    db.peer, mcp.peer = mcp, db
    async with client_for(settings, db, mcp) as client:
        db.entered.clear()
        mcp.entered.clear()
        async with asyncio.timeout(1):
            assert (await client.get("/ready")).status_code == OK


async def test_correlation_id_echoed_in_response_header(settings: Settings) -> None:
    request_id = uuid4().hex
    async with client_for(settings, FakeProbe(), FakeProbe()) as client:
        response = await client.get("/health", headers={"X-Request-ID": request_id})
        assert response.headers["X-Request-ID"] == request_id
        assert response.json()["request_id"] == request_id
        assert correlation_id.get() is None


async def test_correlation_id_generated_when_absent(settings: Settings) -> None:
    async with client_for(settings, FakeProbe(), FakeProbe()) as client:
        response = await client.get("/health")
        returned = response.headers["X-Request-ID"]
        assert UUID(returned).version == UUID_VERSION
        assert response.json()["request_id"] == returned
        assert correlation_id.get() is None


async def test_invalid_request_id_replaced(settings: Settings) -> None:
    async with client_for(settings, FakeProbe(), FakeProbe()) as client:
        response = await client.get("/health", headers={"X-Request-ID": "invalid"})
        assert UUID(response.headers["X-Request-ID"]).version == UUID_VERSION


async def test_cors_preflight_has_request_id(settings: Settings) -> None:
    settings.http.cors_origins = ["http://localhost:18081"]
    settings.http.cors_allow_credentials = True
    async with client_for(settings, FakeProbe(), FakeProbe()) as client:
        for origin, expected in [
            ("http://localhost:18081", 200),
            ("https://foreign.invalid", FORBIDDEN),
        ]:
            response = await client.options(
                "/ready", headers={"Origin": origin, "Access-Control-Request-Method": "GET"}
            )
            assert response.status_code == expected
            assert UUID(response.headers["X-Request-ID"]).version == UUID_VERSION
        response = await client.get("/health", headers={"Origin": "http://localhost:18081"})
        assert response.headers["access-control-allow-origin"] == "http://localhost:18081"
        assert (
            response.headers["access-control-expose-headers"]
            == "X-Request-ID, Idempotency-Replayed, Retry-After"
        )


@pytest.mark.parametrize(
    "origin", ["*", "https://*.example.com", "http://user:pw@localhost", "http://localhost/path"]
)
def test_cors_rejects_non_origins(origin: str) -> None:
    with pytest.raises(ValidationError):
        HTTPSettings(cors_origins=[origin], cors_allow_credentials=True)


@pytest.mark.parametrize(
    ("model", "field", "value"),
    [
        (HTTPSettings, "request_timeout_s", 0),
        (HTTPSettings, "request_timeout_s", 601),
        (HTTPSettings, "shutdown_timeout_s", 0),
        (HTTPSettings, "shutdown_timeout_s", 61),
        (HealthSettings, "postgresql_timeout_s", 0),
        (HealthSettings, "mcp_timeout_s", 11),
    ],
)
def test_timeout_configuration_is_bounded(model: type[BaseModel], field: str, value: int) -> None:
    with pytest.raises(ValidationError):
        model.model_validate({field: value})


class ContextResponse(BaseModel):
    """Test-only endpoint contract."""

    deadline: float
    request_id: str | None


async def test_deadline_and_concurrent_log_isolation(
    settings: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    app = create_app(
        settings, health_service=HealthService(FakeProbe(), FakeProbe(), settings.health)
    )
    both_entered = asyncio.Event()
    arrived: list[str | None] = []

    @app.get("/context")
    async def context(request: Request) -> ContextResponse:
        arrived.append(correlation_id.get())
        if len(arrived) == CONCURRENT_REQUESTS:
            both_entered.set()
        await both_entered.wait()
        structlog.get_logger().info("context_test", marker=correlation_id.get())
        return ContextResponse(deadline=request.state.deadline.at, request_id=correlation_id.get())

    ids = [uuid4().hex, uuid4().hex]
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        start = monotonic()
        async with asyncio.timeout(1), asyncio.TaskGroup() as group:
            requests = [
                group.create_task(client.get("/context", headers={"X-Request-ID": v})) for v in ids
            ]
        end = monotonic()
        for task, expected_id in zip(requests, ids, strict=True):
            body = task.result().json()
            assert body["request_id"] == expected_id
            assert start + 90 <= body["deadline"] <= end + 90
    logs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    events = [entry for entry in logs if entry["event"] == "context_test"]
    assert {entry["request_id"] for entry in events} == set(ids)
    assert all(entry["marker"] == entry["request_id"] for entry in events)
    assert structlog.contextvars.get_contextvars() == {}


@pytest.mark.parametrize("close_mode", ["hang", "failure"])
async def test_bounded_shutdown_attempts_both_cleanups(settings: Settings, close_mode: str) -> None:
    settings.http.shutdown_timeout_s = 0.02
    db, mcp = FakeProbe(), FakeProbe()
    mcp.close_hang = close_mode == "hang"
    mcp.close_failure = close_mode == "failure"
    service = HealthService(db, mcp, settings.health)
    app = create_app(settings, health_service=service)
    async with asyncio.timeout(1):
        async with app.router.lifespan_context(app):
            assert app.state.ready
        assert not app.state.ready
        assert not service.active
        assert not (await service.check()).ready
        assert db.closed
        assert mcp.closed


async def test_cancelled_partial_startup_cleans_resources(settings: Settings) -> None:
    db, mcp = FakeProbe(), FakeProbe()
    mcp.hang = True
    service = HealthService(db, mcp, settings.health)
    app = create_app(settings, health_service=service)
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.02), app.router.lifespan_context(app):
            pytest.fail("Startup was expected to be cancelled")
    assert db.closed
    assert mcp.closed
    assert not service.active
    assert not app.state.ready


async def test_cancelled_request_clears_log_and_correlation_context(settings: Settings) -> None:
    app = create_app(
        settings, health_service=HealthService(FakeProbe(), FakeProbe(), settings.health)
    )
    entered = asyncio.Event()

    @app.get("/wait")
    async def wait_for_cancel() -> ContextResponse:
        entered.set()
        await asyncio.Event().wait()
        pytest.fail("Request should have been cancelled")

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://test") as client,
    ):
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(0.02):
                await client.get("/wait")
        assert entered.is_set()
        assert correlation_id.get() is None
        assert structlog.contextvars.get_contextvars() == {}


def test_main_import_fails_on_missing_settings(tmp_path: Path) -> None:
    code = (
        "from pathlib import Path; import sys; "
        "from app.core.settings_base import ProcessSettings; "
        "ProcessSettings.project_root=Path(sys.argv[1]); import app.main"
    )
    result = subprocess.run(  # noqa: S603 -- fixed interpreter/code with an empty configuration root.
        [sys.executable, "-c", code, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode != 0
    assert "database.app_password" in result.stderr
