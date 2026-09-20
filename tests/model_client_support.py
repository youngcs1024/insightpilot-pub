"""Deterministic model HTTP responses and client-only output factories."""

import time

import httpx

from app.core.config_models import ModelRuntimeClientSettings
from app.core.deadline import Deadline
from app.schemas.model_runtime import (
    EmbedBatchReceipt,
    EmbedOutput,
    EmbedRequest,
    EmbedResult,
    ModelFailure,
    ModelMetadata,
)
from model_runtime.errors import ModelError
from tests.fakes.model_runtime import FakeModels

TOKEN = "synthetic-client-token"  # noqa: S105 -- public test credential.
BASE_URL = "http://model.test"


def settings(**updates: object) -> ModelRuntimeClientSettings:
    return ModelRuntimeClientSettings.model_validate(
        {"base_url": BASE_URL, "auth_token": TOKEN, **updates}
    )


def deadline(seconds: float = 5) -> Deadline:
    return Deadline(time.monotonic() + seconds)


def failure(error: ModelError) -> httpx.Response:
    return httpx.Response(
        error.http_status,
        json=ModelFailure(
            code=error.kind,
            message=error.user_message,
            request_id="failure",
            retryable=error.retryable,
        ).model_dump(mode="json"),
    )


def embedding_response(
    request: httpx.Request, metadata: ModelMetadata | None = None
) -> httpx.Response:
    value = EmbedRequest.model_validate_json(request.content)
    result = EmbedResult(
        dense=[[float(len(text)), *([0.0] * 1023)] for text in value.texts],
        sparse=[{len(text): 1.0} for text in value.texts],
        ms=3,
        queue_ms=1,
        inference_ms=2,
        request_id=request.headers["X-Request-ID"],
        metadata=metadata or FakeModels().metadata(),
    )
    return httpx.Response(200, json=result.model_dump(mode="json"))


def embed_output(response: EmbedResult, *, attempts: int = 1) -> EmbedOutput:
    """Wrap a single transport response with its exact receipt for injected clients."""
    return EmbedOutput(
        dense=response.dense,
        sparse=response.sparse,
        client_ms=response.ms,
        batches=[
            EmbedBatchReceipt(
                **response.model_dump(exclude={"dense", "sparse"}),
                start_index=0,
                text_count=len(response.dense),
                attempts=attempts,
            )
        ],
    )
