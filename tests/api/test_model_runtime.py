"""Model HTTP acceptance with an injected synchronous backend, never real weights."""
# ruff: noqa: PLR2004 -- exact protocol dimensions, scores and deadlines are test expectations.

import asyncio
import time
from collections.abc import AsyncIterator

import httpx
import pytest
from structlog.testing import capture_logs

from app.clients.model_runtime import ModelRuntimeClient
from app.core.config_models import ModelRuntimeClientSettings
from app.core.deadline import Deadline
from app.schemas.model_runtime import EMBED_REVISION, RERANK_REVISION, EmbedMode
from model_runtime.config import ModelServerSettings
from model_runtime.errors import ModelError
from model_runtime.server import BODY_LIMIT, create_app
from tests.fakes.model_runtime import FakeModels

TOKEN = "synthetic-model-test-token"  # noqa: S105 -- public synthetic test token.


def config() -> ModelServerSettings:
    return ModelServerSettings(
        auth_token=TOKEN, embed_revision=EMBED_REVISION, rerank_revision=RERANK_REVISION
    )


async def test_authenticated_embed_rerank_ready_contracts() -> None:
    fake = FakeModels()
    app = create_app(config(), fake)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://model") as http,
    ):
        client = ModelRuntimeClient(
            ModelRuntimeClientSettings(base_url="http://model", auth_token=TOKEN), http
        )
        result = await client.embed(
            ["七天退货", "SKU-A1023"], EmbedMode.QUERY, deadline=Deadline(time.monotonic() + 2)
        )
        assert len(result.dense) == 2
        assert len(result.dense[0]) == 1024
        assert result.sparse == [{42: 0.5}, {42: 0.5}]
        assert result.batches[0].ms >= result.batches[0].inference_ms
        scores = await client.rerank(
            "退款", ["政策", "天气"], deadline=Deadline(time.monotonic() + 2)
        )
        assert scores.scores == [1, 0.5]
        assert (await client.ready(deadline=Deadline(time.monotonic() + 2))).ready
        assert fake.calls.count("load") == 1


async def test_ready_false_before_warmup_and_token_required() -> None:
    app = create_app(config(), FakeModels())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://model"
    ) as http:
        assert (await http.get("/health")).status_code == 200
        assert (await http.get("/ready")).status_code == 401
        assert (
            await http.get("/ready", headers={"Authorization": "Bearer " + TOKEN})
        ).status_code == 503


async def test_cuda_unavailable_fails_startup() -> None:
    fake = FakeModels()
    fake.failure = ModelError("CUDA unavailable")
    app = create_app(config(), fake)
    with pytest.raises(ModelError):
        async with app.router.lifespan_context(app):
            raise AssertionError("Failed startup must not yield")
    assert app.state.ready is False


@pytest.mark.parametrize(
    "payload",
    [
        {"texts": ["x"] * 17, "mode": "query"},
        {"texts": ["x" * 32_001], "mode": "query"},
        {"texts": ["x"], "mode": "unknown"},
        {"texts": [], "mode": "query"},
    ],
)
async def test_invalid_requests_rejected_before_admission(payload: dict) -> None:
    fake = FakeModels()
    app = create_app(config(), fake)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://model") as http,
    ):
        count = len(fake.calls)
        response = await http.post(
            "/v1/embed", json=payload, headers={"Authorization": "Bearer " + TOKEN}
        )
        assert response.status_code == 422
        assert len(fake.calls) == count
        assert TOKEN not in response.text


async def test_streamed_body_limit_is_enforced_without_content_length() -> None:
    app = create_app(config(), FakeModels())

    async def chunks() -> AsyncIterator[bytes]:
        yield b"x" * (BODY_LIMIT // 2)
        yield b"x" * (BODY_LIMIT // 2 + 1)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://model"
    ) as http:
        response = await http.post("/v1/embed", content=chunks())
        assert response.status_code == 422


async def test_health_responsive_during_inference() -> None:
    fake = FakeModels()
    app = create_app(config(), fake)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://model") as http,
    ):
        fake.entered.clear()
        fake.release.clear()
        work = asyncio.create_task(
            http.post(
                "/v1/embed",
                json={"texts": ["政策"], "mode": "query"},
                headers={"Authorization": "Bearer " + TOKEN},
            )
        )
        try:
            assert await asyncio.to_thread(fake.entered.wait, 1)
            async with asyncio.timeout(0.5):
                assert (await http.get("/health")).status_code == 200
        finally:
            fake.release.set()
            await work


async def test_body_upload_consumes_operation_deadline() -> None:
    app = create_app(config(), FakeModels())

    async def delayed() -> AsyncIterator[bytes]:
        await asyncio.sleep(0.05)
        yield b"{}"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://model"
    ) as http:
        response = await http.post(
            "/v1/embed", content=delayed(), headers={"X-Request-Timeout-Ms": "1"}
        )
    assert response.status_code == 504
    assert response.json()["retryable"] is False


async def test_adapter_failure_never_logs_input_or_exception_prose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeModels()
    app = create_app(config(), fake)
    sensitive = "private-document-fragment"

    def failing(*args: object) -> object:
        raise RuntimeError(sensitive)

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://model") as http,
    ):
        monkeypatch.setattr(fake, "embed", failing)
        with capture_logs() as logs:
            response = await http.post(
                "/v1/embed",
                json={"texts": [sensitive], "mode": "query"},
                headers={"Authorization": "Bearer " + TOKEN},
            )
            await asyncio.sleep(0)
        assert response.status_code == 503
        assert sensitive not in response.text
        assert sensitive not in str(logs)
        assert TOKEN not in str(logs)
        assert len(response.headers.get_list("X-Request-ID")) == 1
