"""API model warmup is bounded, recoverable and owned by the lifespan."""
# ruff: noqa: PLR2004 -- explicit readiness statuses and startup budget expectations.

import asyncio
import time
from unittest.mock import AsyncMock

import httpx
import pytest
import respx
from structlog.testing import capture_logs

from app.application import create_app
from app.clients.model_runtime import ModelRuntimeClient, ModelRuntimeProbe
from app.core.config_models import Settings
from app.core.errors import CheckpointError
from app.schemas.model_runtime import ReadyResult
from app.services.graph import GraphService
from app.services.health import HealthService
from model_runtime.errors import ModelError
from tests.fakes.health_probe import FakeProbe
from tests.fakes.model_runtime import FakeModels
from tests.model_client_support import BASE_URL, TOKEN, failure
from tests.model_client_support import settings as model_settings


async def test_model_warmup_readiness_and_lifespan_ownership(
    settings: Settings,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings.retrieval.enabled = True
    settings.model_runtime = model_settings()
    monkeypatch.setattr("app.application.setup_logging", lambda settings: None)
    client = ModelRuntimeClient(settings.model_runtime)
    probe = ModelRuntimeProbe(client)
    service = HealthService(FakeProbe(), FakeProbe(), settings.health, model=probe)
    monkeypatch.setattr("app.clients.model_runtime.WARMUP_POLL_S", 0.001)
    ready = ReadyResult(request_id="ready", metadata=FakeModels().metadata())
    route = respx_mock.get(BASE_URL + "/ready").mock(
        side_effect=[
            failure(ModelError()),
            httpx.Response(200, json=ready.model_dump(mode="json")),
            httpx.Response(200, json=ready.model_dump(mode="json")),
        ]
    )
    app = create_app(settings, health_service=service, model_runtime_client=client)
    with capture_logs() as logs:
        async with app.router.lifespan_context(app):
            assert app.state.ready
            assert app.state.model_runtime is probe.client is client
            assert route.call_count == 3  # Two warmup polls, one independent readiness check.
            assert not client._http.is_closed
    assert client._http.is_closed
    assert any(item["event"] == "model_warmup_ready" for item in logs)
    assert TOKEN not in str(logs)


async def test_model_outage_keeps_api_live_and_recovers(
    settings: Settings,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings.retrieval.enabled = True
    settings.model_runtime = model_settings()
    monkeypatch.setattr("app.application.setup_logging", lambda settings: None)
    client = ModelRuntimeClient(settings.model_runtime)
    service = HealthService(
        FakeProbe(), FakeProbe(), settings.health, model=ModelRuntimeProbe(client)
    )
    monkeypatch.setattr("app.application.STARTUP_BUDGET_S", 0.05)
    monkeypatch.setattr("app.clients.model_runtime.WARMUP_POLL_S", 0.001)
    route = respx_mock.get(BASE_URL + "/ready").mock(return_value=failure(ModelError()))
    app = create_app(settings, health_service=service, model_runtime_client=client)
    with capture_logs() as logs:
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://api") as http,
        ):
            assert not app.state.ready
            assert (await http.get("/health")).status_code == 200
            response = await http.get("/ready")
            assert response.status_code == 503
            assert response.json()["checks"]["model_runtime"] is False
            ready = ReadyResult(request_id="ready", metadata=FakeModels().metadata())
            route.mock(return_value=httpx.Response(200, json=ready.model_dump(mode="json")))
            assert (await http.get("/ready")).status_code == 200
            assert app.state.ready
    assert any(item["event"] == "model_startup_unavailable" for item in logs)
    assert TOKEN not in str(logs)
    assert client._http.is_closed


async def test_disabled_retrieval_never_constructs_or_warms_model(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings.model_runtime = model_settings()
    factory = AsyncMock()
    monkeypatch.setattr("app.application.ModelRuntimeClient", factory)
    app = create_app(
        settings, health_service=HealthService(FakeProbe(), FakeProbe(), settings.health)
    )
    async with app.router.lifespan_context(app):
        assert app.state.model_runtime is None
        assert app.state.ready
    factory.assert_not_called()


async def test_prior_startup_work_consumes_model_budget(settings: Settings) -> None:
    settings.retrieval.enabled = True
    settings.model_runtime = model_settings()
    graph = AsyncMock(spec=GraphService)
    observed = []

    async def start() -> None:
        observed.append(time.monotonic())

    graph.start.side_effect = start
    client = AsyncMock(spec=ModelRuntimeClient)
    app = create_app(
        settings,
        model_runtime_client=client,
        graph_service=graph,
        health_service=HealthService(FakeProbe(), FakeProbe(), settings.health),
    )
    async with app.router.lifespan_context(app):
        budget = client.warmup.call_args.kwargs["deadline"]
        assert budget.at <= observed[0] + 180
        client.warmup.assert_awaited_once()
    client.aclose.assert_awaited_once()


async def test_partial_startup_failure_closes_model_client(settings: Settings) -> None:
    settings.retrieval.enabled = True
    settings.model_runtime = model_settings()
    graph = AsyncMock(spec=GraphService)
    graph.start.side_effect = CheckpointError()
    client = AsyncMock(spec=ModelRuntimeClient)
    app = create_app(settings, graph_service=graph, model_runtime_client=client)
    with pytest.raises(CheckpointError):
        async with app.router.lifespan_context(app):
            pytest.fail("Startup failure cannot yield")
    client.warmup.assert_not_awaited()
    client.aclose.assert_awaited_once()


async def test_cancelled_startup_closes_model_client(settings: Settings) -> None:
    settings.retrieval.enabled = True
    settings.model_runtime = model_settings()
    entered = asyncio.Event()
    client = AsyncMock(spec=ModelRuntimeClient)

    async def warmup(**kwargs: object) -> None:
        entered.set()
        await asyncio.Event().wait()

    client.warmup.side_effect = warmup
    app = create_app(settings, model_runtime_client=client)

    async def start() -> None:
        async with app.router.lifespan_context(app):
            pytest.fail("Cancelled startup cannot yield")

    async with asyncio.timeout(3), asyncio.TaskGroup() as group:
        work = group.create_task(start())
        await entered.wait()
        work.cancel()
        with pytest.raises(asyncio.CancelledError):
            await work
    client.aclose.assert_awaited_once()
    assert not app.state.ready
