"""Deadlines, cancellation and startup remain bounded without live model services."""
# ruff: noqa: PLR2004 -- explicit timeout, protocol and admission expectations.

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import httpx
import pytest
import respx

from app.clients.model_resilience import ModelCircuitBreaker
from app.clients.model_runtime import ModelRuntimeClient, ModelRuntimeProbe
from app.core.errors import RetryNestingError
from app.core.retry import run_operation
from app.schemas.model_runtime import EmbedMode, ReadyResult, RerankResult
from model_runtime.errors import ModelAuthError, ModelContractError, ModelDeadlineError, ModelError
from tests.fakes.model_runtime import FakeModels
from tests.model_client_support import BASE_URL, deadline, embedding_response, failure, settings


@pytest.fixture
async def client() -> AsyncIterator[ModelRuntimeClient]:
    value = ModelRuntimeClient(settings())
    try:
        yield value
    finally:
        await value.aclose()


@pytest.mark.parametrize("operation", ["embed", "rerank"])
async def test_hard_timeout_caps(
    client: ModelRuntimeClient,
    respx_mock: respx.MockRouter,
    operation: str,
) -> None:
    client._settings.embed_timeout_s = client._settings.rerank_timeout_s = 120
    observed = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request.extensions["timeout"]["read"])
        if operation == "embed":
            return embedding_response(request)
        return httpx.Response(
            200,
            json=RerankResult(
                request_id="r",
                ms=1,
                queue_ms=0,
                inference_ms=1,
                metadata=FakeModels().metadata(),
                scores=[0.8],
            ).model_dump(mode="json"),
        )

    respx_mock.post(BASE_URL + "/v1/" + operation).mock(side_effect=handler)
    if operation == "embed":
        await client.embed(["x"], EmbedMode.QUERY, deadline=deadline(100))
    else:
        await client.rerank("x", ["x"], deadline=deadline(100))
    assert observed == [20 if operation == "embed" else 30]


async def test_timeout_is_not_retried(
    client: ModelRuntimeClient, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(BASE_URL + "/v1/embed").mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(ModelDeadlineError):
        await client.embed(["x"], EmbedMode.QUERY, deadline=deadline())
    assert route.call_count == 1
    assert client.breaker.failures == 0


async def test_deadline_expires_during_backoff(
    client: ModelRuntimeClient, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(BASE_URL + "/v1/embed").mock(return_value=failure(ModelError()))
    with pytest.raises(ModelDeadlineError):
        await client.embed(["x"], EmbedMode.QUERY, deadline=deadline(0.05))
    assert route.call_count == 1
    assert client.breaker.failures == 0
    assert not client.breaker.probing


async def test_expired_deadline_skips_remaining_batches(
    client: ModelRuntimeClient, respx_mock: respx.MockRouter
) -> None:
    budget = deadline()

    def handler(request: httpx.Request) -> httpx.Response:
        object.__setattr__(budget, "at", 0)
        return embedding_response(request)

    route = respx_mock.post(BASE_URL + "/v1/embed").mock(side_effect=handler)
    with pytest.raises(ModelDeadlineError):
        await client.embed(["x"] * 17, EmbedMode.DOCUMENT, deadline=budget)
    assert route.call_count == 1


async def test_queue_wait_obeys_deadline(
    client: ModelRuntimeClient, respx_mock: respx.MockRouter
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        entered.set()
        await release.wait()
        return embedding_response(request)

    route = respx_mock.post(BASE_URL + "/v1/embed").mock(side_effect=handler)
    async with asyncio.timeout(3), asyncio.TaskGroup() as group:
        work = group.create_task(client.embed(["first"], EmbedMode.QUERY, deadline=deadline()))
        await entered.wait()
        try:
            with pytest.raises(ModelDeadlineError):
                await client.embed(["queued"], EmbedMode.QUERY, deadline=deadline(0.02))
            assert not work.done()
        finally:
            release.set()
    assert route.call_count == 1
    assert client.breaker.failures == 0


@pytest.mark.parametrize("stage", ["queued", "inflight", "backoff"])
async def test_cancellation_releases_admission(
    client: ModelRuntimeClient,
    respx_mock: respx.MockRouter,
    stage: str,
) -> None:
    entered, release, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        entered.set()
        if stage == "backoff":
            return failure(ModelError())
        try:
            await release.wait()
        finally:
            stopped.set()
        return embedding_response(request)

    route = respx_mock.post(BASE_URL + "/v1/embed").mock(side_effect=handler)
    async with asyncio.timeout(3), asyncio.TaskGroup() as group:
        first = group.create_task(client.embed(["x"], EmbedMode.QUERY, deadline=deadline()))
        await entered.wait()
        if stage == "queued":
            queued = asyncio.Event()

            async def waiting() -> None:
                queued.set()
                await client.embed(["y"], EmbedMode.QUERY, deadline=deadline())

            target = group.create_task(waiting())
            await queued.wait()
        else:
            target = first
        target.cancel()
        with pytest.raises(asyncio.CancelledError):
            await target
        release.set()
    if stage == "inflight":
        assert stopped.is_set()
    assert client.breaker.failures == 0
    route.mock(side_effect=embedding_response)
    assert (await client.embed(["x"], EmbedMode.QUERY, deadline=deadline())).dense


async def test_cancelled_half_open_probe_can_be_replaced(
    client: ModelRuntimeClient, respx_mock: respx.MockRouter
) -> None:
    now = [0.0]
    client.breaker = ModelCircuitBreaker(clock=lambda: now[0])
    for _ in range(5):
        client.breaker.failed()
    now[0] = 30
    entered = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("Unreleased operation returned")

    route = respx_mock.post(BASE_URL + "/v1/embed").mock(side_effect=handler)
    async with asyncio.timeout(3), asyncio.TaskGroup() as group:
        work = group.create_task(client.embed(["x"], EmbedMode.QUERY, deadline=deadline()))
        await entered.wait()
        work.cancel()
        with pytest.raises(asyncio.CancelledError):
            await work
    assert not client.breaker.probing
    assert client.breaker.opened_at == 0
    route.mock(side_effect=embedding_response)
    await client.embed(["x"], EmbedMode.QUERY, deadline=deadline())
    assert client.breaker.opened_at is None


async def test_retry_owner_cannot_be_nested(
    client: ModelRuntimeClient, respx_mock: respx.MockRouter
) -> None:
    with pytest.raises(RetryNestingError):
        await run_operation(
            lambda: client.embed(["x"], EmbedMode.QUERY, deadline=deadline()),
            deadline=deadline(),
            timeout_s=1,
            name="outer",
        )
    assert not respx_mock.calls
    assert client.breaker.failures == 0


async def test_warmup_polls_authenticated_ready_without_inference(
    client: ModelRuntimeClient,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.clients.model_runtime.WARMUP_POLL_S", 0.001)
    ready = ReadyResult(request_id="ready", metadata=FakeModels().metadata())
    route = respx_mock.get(BASE_URL + "/ready").mock(
        side_effect=[failure(ModelError()), httpx.Response(200, json=ready.model_dump(mode="json"))]
    )
    assert await client.warmup(deadline=deadline()) == ready
    assert route.call_count == 2
    assert all(call.request.headers["Authorization"].startswith("Bearer ") for call in route.calls)
    assert client.breaker.failures == 0


@pytest.mark.parametrize("error", [ModelAuthError(), ModelContractError()])
async def test_warmup_stops_on_permanent_error(
    client: ModelRuntimeClient,
    respx_mock: respx.MockRouter,
    error: ModelError,
) -> None:
    route = respx_mock.get(BASE_URL + "/ready").mock(return_value=failure(error))
    with pytest.raises(type(error)):
        await client.warmup(deadline=deadline())
    assert route.call_count == 1


async def test_warmup_total_budget_and_no_inference_breaker(
    client: ModelRuntimeClient,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.clients.model_runtime.WARMUP_POLL_S", 0.001)
    route = respx_mock.get(BASE_URL + "/ready").mock(return_value=failure(ModelError()))
    with pytest.raises(ModelDeadlineError):
        await client.warmup(deadline=deadline(0.05))
    assert route.call_count >= 1
    assert client.breaker.failures == 0
    assert not client.breaker.probing


async def test_warmup_caps_budget_and_preserves_smaller_parent(client: ModelRuntimeClient) -> None:
    ready = AsyncMock(
        return_value=ReadyResult(request_id="ready", metadata=FakeModels().metadata())
    )
    client.ready = ready
    for seconds in (500, 0.5):
        budget = deadline(seconds)
        await client.warmup(deadline=budget)
        observed = ready.call_args.kwargs["deadline"]
        assert observed.at <= budget.at
        assert 0 < observed.remaining() <= min(180, seconds)


async def test_cancelled_warmup_does_not_retry(client: ModelRuntimeClient) -> None:
    entered, stopped = asyncio.Event(), asyncio.Event()

    async def ready(**kwargs: object) -> ReadyResult:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
        raise AssertionError("Unreleased operation returned")

    client.ready = ready
    async with asyncio.timeout(3), asyncio.TaskGroup() as group:
        work = group.create_task(client.warmup(deadline=deadline()))
        await entered.wait()
        work.cancel()
        with pytest.raises(asyncio.CancelledError):
            await work
    assert stopped.is_set()


async def test_resource_ownership_and_closed_admission(respx_mock: respx.MockRouter) -> None:
    async with httpx.AsyncClient() as http:
        client = ModelRuntimeClient(settings(), http)
        probe = ModelRuntimeProbe(client)
        await probe.aclose()
        assert not http.is_closed
        await client.aclose()
        assert not http.is_closed
        with pytest.raises(ModelError):
            await client.embed(["x"], EmbedMode.QUERY, deadline=deadline())
    owned = ModelRuntimeClient(settings())
    await owned.aclose()
    await owned.aclose()
    assert owned._http.is_closed
    assert not respx_mock.calls
