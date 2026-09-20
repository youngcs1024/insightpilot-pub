"""Untrusted model responses cannot bypass identity, shape or retry policy."""
# ruff: noqa: PLR2004 -- exact protocol dimensions, scores and deadlines are test expectations.

import asyncio
import time
from collections.abc import AsyncIterator

import httpx
import pytest
import respx
from asgi_correlation_id import correlation_id
from pydantic import ValidationError
from structlog.testing import capture_logs
from tenacity import wait_none

from app.clients.model_resilience import ModelCircuitBreaker
from app.clients.model_runtime import ModelRuntimeClient
from app.core.config_models import ModelRuntimeClientSettings
from app.core.deadline import Deadline
from app.schemas.model_runtime import (
    EmbedMode,
    EmbedOutput,
    EmbedRequest,
    EmbedResult,
    ModelFailure,
    ModelFailureKind,
    ReadyResult,
    RerankRequest,
    RerankResult,
)
from model_runtime.errors import (
    ModelAuthError,
    ModelContractError,
    ModelDeadlineError,
    ModelError,
    ModelInputError,
    ModelOOMError,
    ModelQueueError,
)
from tests.fakes.model_runtime import FakeModels
from tests.model_client_support import (
    BASE_URL,
    TOKEN,
    deadline,
    embed_output,
    embedding_response,
    failure,
    settings,
)


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.core.retry.wait_exponential", lambda **kwargs: wait_none())


def result() -> EmbedResult:
    return EmbedResult(
        dense=[[0.0] * 1024],
        sparse=[{42: 1.0}],
        ms=1,
        queue_ms=0,
        inference_ms=1,
        request_id="test",
        metadata=FakeModels().metadata(),
    )


@pytest.mark.parametrize("defect", ["revision", "dimension", "sparse", "count", "nan", "version"])
async def test_invalid_contract_rejected(defect: str) -> None:
    body = result().model_dump(mode="json")
    if defect == "revision":
        body["metadata"]["embed_revision"] = "a" * 40
    elif defect == "dimension":
        body["dense"] = [[0.0]]
    elif defect == "sparse":
        body["sparse"] = [{}]
    elif defect == "count":
        body["dense"] = []
    elif defect == "nan":
        body["dense"][0][0] = "NaN"
    else:
        body["schema_version"] = 2
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))
    ) as http:
        client = ModelRuntimeClient(ModelRuntimeClientSettings(auth_token="private"), http)  # noqa: S106 -- public fixture credential.
        with pytest.raises(ModelContractError):
            await client.embed(["text"], EmbedMode.QUERY, deadline=Deadline(time.monotonic() + 2))


@pytest.mark.parametrize(
    ("kind", "status", "exception"),
    [
        (ModelFailureKind.AUTH, 401, ModelAuthError),
        (ModelFailureKind.OOM, 503, ModelOOMError),
    ],
)
async def test_auth_and_oom_not_retried(
    kind: ModelFailureKind, status: int, exception: type[Exception]
) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            status,
            json=ModelFailure(
                code=kind, message="fixed", request_id="x", retryable=False
            ).model_dump(mode="json"),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = ModelRuntimeClient(ModelRuntimeClientSettings(auth_token="private"), http)  # noqa: S106 -- public fixture credential.
        with pytest.raises(exception):
            await client.embed(["text"], EmbedMode.QUERY, deadline=Deadline(time.monotonic() + 0.5))
    assert len(calls) == 1
    assert 0 < int(calls[0].headers["X-Request-Timeout-Ms"]) <= 500


async def test_expired_deadline_makes_no_http_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("No expired work may reach HTTP")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = ModelRuntimeClient(ModelRuntimeClientSettings(auth_token="private"), http)  # noqa: S106 -- public fixture credential.
        with pytest.raises(ModelDeadlineError):
            await client.ready(deadline=Deadline(time.monotonic() - 1))


async def test_existing_request_id_is_propagated() -> None:
    observed = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request.headers["X-Request-ID"])
        return httpx.Response(200, json=result().model_dump(mode="json"))

    token = correlation_id.set("a" * 32)
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            settings = ModelRuntimeClientSettings(auth_token="private")  # noqa: S106 -- public fixture credential.
            client = ModelRuntimeClient(settings, http)
            await client.embed(["text"], EmbedMode.QUERY, deadline=Deadline(time.monotonic() + 1))
        assert observed == ["a" * 32]
    finally:
        correlation_id.reset(token)


@pytest.fixture
async def client() -> AsyncIterator[ModelRuntimeClient]:
    value = ModelRuntimeClient(settings())
    try:
        yield value
    finally:
        await value.aclose()


async def test_batches_respect_size_limit(
    client: ModelRuntimeClient, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(BASE_URL + "/v1/embed").mock(side_effect=embedding_response)
    texts = ["x" * size for size in range(1, 36)]
    output = await client.embed(texts, EmbedMode.DOCUMENT, deadline=deadline())
    requests = [EmbedRequest.model_validate_json(call.request.content) for call in route.calls]
    assert [len(value.texts) for value in requests] == [16, 16, 3]
    assert [text for value in requests for text in value.texts] == texts
    assert [row[0] for row in output.dense] == list(range(1, 36))
    assert output.sparse == [{size: 1.0} for size in range(1, 36)]
    assert [batch.start_index for batch in output.batches] == [0, 16, 32]
    assert [batch.text_count for batch in output.batches] == [16, 16, 3]
    assert all(batch.attempts == 1 and batch.ms == 3 for batch in output.batches)
    assert [batch.request_id for batch in output.batches] == [
        call.request.headers["X-Request-ID"] for call in route.calls
    ]
    assert output.client_ms >= 0


async def test_configured_small_batches(
    client: ModelRuntimeClient, respx_mock: respx.MockRouter
) -> None:
    client._settings.embed_batch = 2
    route = respx_mock.post(BASE_URL + "/v1/embed").mock(side_effect=embedding_response)
    await client.embed(["x"] * 5, EmbedMode.DOCUMENT, deadline=deadline())
    assert [
        len(EmbedRequest.model_validate_json(c.request.content).texts) for c in route.calls
    ] == [2, 2, 1]


async def test_bounded_concurrency(
    client: ModelRuntimeClient, respx_mock: respx.MockRouter
) -> None:
    entered, release, queued = asyncio.Event(), asyncio.Event(), asyncio.Event()
    active = peak = 0

    async def encode(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        entered.set()
        await release.wait()
        active -= 1
        return embedding_response(request)

    async def rerank(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        parsed = RerankRequest.model_validate_json(request.content)
        assert len(parsed.passages) == 20
        active -= 1
        return httpx.Response(
            200,
            json=RerankResult(
                request_id="rerank",
                ms=1,
                queue_ms=0,
                inference_ms=1,
                metadata=FakeModels().metadata(),
                scores=[0.8] * 20,
            ).model_dump(mode="json"),
        )

    async def second() -> None:
        queued.set()
        await client.rerank("query", ["passage"] * 20, deadline=deadline())

    embed_route = respx_mock.post(BASE_URL + "/v1/embed").mock(side_effect=encode)
    rerank_route = respx_mock.post(BASE_URL + "/v1/rerank").mock(side_effect=rerank)
    async with asyncio.timeout(3), asyncio.TaskGroup() as group:
        group.create_task(client.embed(["x"] * 17, EmbedMode.DOCUMENT, deadline=deadline()))
        await entered.wait()
        group.create_task(second())
        await queued.wait()
        try:
            assert not rerank_route.called
        finally:
            release.set()
    assert peak == 1
    assert embed_route.call_count == 2
    assert rerank_route.call_count == 1


@pytest.mark.parametrize("error", [ModelError(), ModelQueueError()])
async def test_retries_on_503_not_on_400(
    client: ModelRuntimeClient,
    respx_mock: respx.MockRouter,
    no_backoff: None,
    error: ModelError,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return failure(error) if calls == 1 else embedding_response(request)

    route = respx_mock.post(BASE_URL + "/v1/embed").mock(side_effect=handler)
    output = await client.embed(["text"], EmbedMode.QUERY, deadline=deadline())
    assert output.batches[0].attempts == route.call_count == 2
    assert client.breaker.failures == 0
    route.mock(return_value=httpx.Response(400, json={"detail": "invalid"}))
    with pytest.raises(ModelContractError):
        await client.embed(["text"], EmbedMode.QUERY, deadline=deadline())
    assert route.call_count == 3
    assert client.breaker.failures == 0


@pytest.mark.parametrize(
    "error",
    [
        ModelAuthError(),
        ModelInputError(),
        ModelOOMError(),
        ModelDeadlineError(),
        ModelContractError(),
    ],
)
async def test_typed_nonretryable_failure_once(
    client: ModelRuntimeClient,
    respx_mock: respx.MockRouter,
    error: ModelError,
) -> None:
    route = respx_mock.post(BASE_URL + "/v1/embed").mock(return_value=failure(error))
    with pytest.raises(type(error)):
        await client.embed(["text"], EmbedMode.QUERY, deadline=deadline())
    assert route.call_count == 1
    assert client.breaker.failures == 0


async def test_oom_failure_is_not_retried(
    client: ModelRuntimeClient, respx_mock: respx.MockRouter
) -> None:
    route = respx_mock.post(BASE_URL + "/v1/embed").mock(return_value=failure(ModelOOMError()))
    with pytest.raises(ModelOOMError):
        await client.embed(["x"] * 17, EmbedMode.DOCUMENT, deadline=deadline())
    assert route.call_count == 1


@pytest.mark.parametrize("status", [400, 429, 503])
async def test_status_and_retryable_fields_cannot_invent_retries(
    client: ModelRuntimeClient,
    respx_mock: respx.MockRouter,
    status: int,
) -> None:
    payload = ModelFailure(
        code=ModelFailureKind.OOM, message="retry please", request_id="x", retryable=True
    )
    route = respx_mock.post(BASE_URL + "/v1/embed").mock(
        return_value=httpx.Response(status, json=payload.model_dump(mode="json"))
    )
    with pytest.raises(ModelContractError):
        await client.embed(["x"], EmbedMode.QUERY, deadline=deadline())
    assert route.call_count == 1


async def test_connection_failure_retries_then_succeeds(
    client: ModelRuntimeClient,
    respx_mock: respx.MockRouter,
    no_backoff: None,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("connection unavailable", request=request)
        return embedding_response(request)

    respx_mock.post(BASE_URL + "/v1/embed").mock(side_effect=handler)
    output = await client.embed(["x"], EmbedMode.QUERY, deadline=deadline())
    assert output.batches[0].attempts == calls == 2


async def test_circuit_breaker_opens_after_threshold(
    client: ModelRuntimeClient,
    respx_mock: respx.MockRouter,
    no_backoff: None,
) -> None:
    route = respx_mock.post(BASE_URL + "/v1/embed").mock(return_value=failure(ModelError()))
    for _ in range(5):
        with pytest.raises(ModelError):
            await client.embed(["x"], EmbedMode.QUERY, deadline=deadline())
    assert route.call_count == 10
    assert client.breaker.failures == 5
    with pytest.raises(ModelError):
        await client.rerank("x", ["x"], deadline=deadline())
    assert route.call_count == 10
    ready_route = respx_mock.get(BASE_URL + "/ready").mock(
        return_value=httpx.Response(
            200,
            json=ReadyResult(request_id="r", metadata=FakeModels().metadata()).model_dump(
                mode="json"
            ),
        )
    )
    assert (await client.ready(deadline=deadline())).ready
    assert ready_route.call_count == 1
    assert client.breaker.failures == 5
    with pytest.raises(ModelError):
        await client.embed(["x"], EmbedMode.QUERY, deadline=deadline())
    assert route.call_count == 10


async def test_half_open_has_one_probe_and_success_recovers(
    client: ModelRuntimeClient,
    respx_mock: respx.MockRouter,
) -> None:
    now = [0.0]
    client.breaker = ModelCircuitBreaker(clock=lambda: now[0])
    for _ in range(5):
        client.breaker.failed()
    now[0] = 30
    entered, release = asyncio.Event(), asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        entered.set()
        await release.wait()
        return embedding_response(request)

    route = respx_mock.post(BASE_URL + "/v1/embed").mock(side_effect=handler)
    async with asyncio.timeout(3), asyncio.TaskGroup() as group:
        group.create_task(client.embed(["x"], EmbedMode.QUERY, deadline=deadline()))
        await entered.wait()
        try:
            with pytest.raises(ModelError):
                await client.embed(["x"], EmbedMode.QUERY, deadline=deadline())
        finally:
            release.set()
    assert route.call_count == 1
    assert client.breaker.opened_at is None
    await client.embed(["x"], EmbedMode.QUERY, deadline=deadline())
    assert route.call_count == 2


async def test_half_open_failure_restarts_cooldown(
    client: ModelRuntimeClient,
    respx_mock: respx.MockRouter,
    no_backoff: None,
) -> None:
    now = [0.0]
    client.breaker = ModelCircuitBreaker(clock=lambda: now[0])
    for _ in range(5):
        client.breaker.failed()
    now[0] = 30
    route = respx_mock.post(BASE_URL + "/v1/embed").mock(return_value=failure(ModelQueueError()))
    with pytest.raises(ModelQueueError):
        await client.embed(["x"], EmbedMode.QUERY, deadline=deadline())
    assert client.breaker.opened_at == 30
    assert not client.breaker.probing
    now[0] = 59
    with pytest.raises(ModelError):
        await client.embed(["x"], EmbedMode.QUERY, deadline=deadline())
    assert route.call_count == 2


async def test_deadline_shrinks_timeout(
    client: ModelRuntimeClient, respx_mock: respx.MockRouter
) -> None:
    observed = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return embedding_response(request)

    respx_mock.post(BASE_URL + "/v1/embed").mock(side_effect=handler)
    await client.embed(["x"], EmbedMode.QUERY, deadline=deadline(0.5))
    request = observed[0]
    assert 0 < request.extensions["timeout"]["read"] <= 0.5
    assert 0 < int(request.headers["X-Request-Timeout-Ms"]) <= 500


async def test_partial_batch_failure_fails_whole_call(
    client: ModelRuntimeClient,
    respx_mock: respx.MockRouter,
    no_backoff: None,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return embedding_response(request) if calls == 1 else failure(ModelError())

    respx_mock.post(BASE_URL + "/v1/embed").mock(side_effect=handler)
    with pytest.raises(ModelError):
        await client.embed(["x"] * 33, EmbedMode.DOCUMENT, deadline=deadline())
    assert calls == 3
    assert client.breaker.failures == 1


async def test_revision_mismatch_is_rejected(
    client: ModelRuntimeClient, respx_mock: respx.MockRouter
) -> None:
    metadata = FakeModels().metadata().model_copy(update={"embed_revision": "a" * 40})
    route = respx_mock.post(BASE_URL + "/v1/embed").mock(
        side_effect=lambda request: embedding_response(request, metadata)
    )
    with pytest.raises(ModelContractError):
        await client.embed(["x"], EmbedMode.QUERY, deadline=deadline())
    assert route.call_count == 1


@pytest.mark.parametrize("field", ["embed_max_length", "embed_batch"])
async def test_batch_metadata_preserves_recovery_and_rejects_semantic_drift(
    client: ModelRuntimeClient,
    respx_mock: respx.MockRouter,
    field: str,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        metadata = FakeModels().metadata()
        if calls > 1:
            metadata = metadata.model_copy(update={field: 8})
        return embedding_response(request, metadata)

    respx_mock.post(BASE_URL + "/v1/embed").mock(side_effect=handler)
    if field == "embed_max_length":
        with pytest.raises(ModelContractError):
            await client.embed(["x"] * 33, EmbedMode.DOCUMENT, deadline=deadline())
        assert calls == 2
    else:
        output = await client.embed(["x"] * 17, EmbedMode.DOCUMENT, deadline=deadline())
        assert [batch.metadata.embed_batch for batch in output.batches] == [16, 8]


@pytest.mark.parametrize("texts", [[], [""], ["x" * 32_001], ["ok"] * 16 + [""]])
async def test_invalid_input_rejected_before_http(
    client: ModelRuntimeClient, texts: list[str]
) -> None:
    with pytest.raises(ModelInputError):
        await client.embed(texts, EmbedMode.DOCUMENT, deadline=deadline())
    assert client.breaker.failures == 0


@pytest.mark.parametrize("defect", ["gap", "count", "metadata"])
def test_aggregate_rejects_invalid_receipts(defect: str) -> None:
    body = embed_output(result()).model_dump()
    if defect == "gap":
        body["batches"][0]["start_index"] = 1
    elif defect == "count":
        body["sparse"] = []
    else:
        extra = body["batches"][0].copy()
        extra["metadata"] = {**extra["metadata"], "embed_max_length": 256}
        extra["start_index"] = 1
        body["batches"].append(extra)
        body["dense"] *= 2
        body["sparse"] *= 2
    with pytest.raises(ValidationError):
        EmbedOutput.model_validate(body)


async def test_model_token_not_logged(
    client: ModelRuntimeClient,
    respx_mock: respx.MockRouter,
    no_backoff: None,
) -> None:
    route = respx_mock.post(BASE_URL + "/v1/embed").mock(return_value=failure(ModelError()))
    with capture_logs() as logs:
        for _ in range(5):
            with pytest.raises(ModelError):
                await client.embed(["private source body"], EmbedMode.DOCUMENT, deadline=deadline())
    assert route.call_count == 10
    assert any(item["event"] == "model_circuit_opened" for item in logs)
    assert any(item["event"] == "operation_retry_scheduled" for item in logs)
    assert TOKEN not in str(logs)
    assert "private source body" not in str(logs)
