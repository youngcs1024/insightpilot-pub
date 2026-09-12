"""Untrusted model responses cannot bypass identity, shape or retry policy."""
# ruff: noqa: PLR2004 -- exact protocol dimensions, scores and deadlines are test expectations.

import time

import httpx
import pytest
from asgi_correlation_id import correlation_id

from app.clients.model_runtime import ModelRuntimeClient
from app.core.config_models import ModelRuntimeClientSettings
from app.core.deadline import Deadline
from app.schemas.model_runtime import EmbedMode, EmbedResult, ModelFailure, ModelFailureKind
from model_runtime.errors import (
    ModelAuthError,
    ModelContractError,
    ModelDeadlineError,
    ModelOOMError,
)
from tests.fakes.model_runtime import FakeModels


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
