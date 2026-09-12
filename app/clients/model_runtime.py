"""Step 3.2 authenticated typed transport; resilience is introduced in Step 3.12."""

import asyncio
import time
from http import HTTPStatus
from uuid import uuid4

import httpx
from asgi_correlation_id import correlation_id
from pydantic import ValidationError

from app.core.config_models import ModelRuntimeClientSettings
from app.core.deadline import Deadline
from app.core.observability import TraceMetadata, observe
from app.schemas.model_runtime import (
    EmbedMode,
    EmbedRequest,
    EmbedResult,
    ModelFailure,
    ModelFailureKind,
    ModelMetadata,
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


class ModelRuntimeClient:
    """One HTTP attempt per call, bounded by an absolute caller-owned deadline."""

    def __init__(
        self, settings: ModelRuntimeClientSettings, http: httpx.AsyncClient | None = None
    ) -> None:
        self._settings = settings
        self._http = http or httpx.AsyncClient(trust_env=False)
        self._owns_http = http is None

    def _identity(self, metadata: ModelMetadata) -> None:
        if (
            metadata.embed_revision != self._settings.embed_revision
            or metadata.rerank_revision != self._settings.rerank_revision
            or metadata.precision != self._settings.precision
        ):
            raise ModelContractError()

    async def _call(self, path: str, body: str | None, deadline: Deadline, timeout: float) -> bytes:
        started = time.monotonic()
        with observe("model_runtime", TraceMetadata(tool=path)) as observation:
            try:
                return await self._request(path, body, deadline, timeout)
            finally:
                if observation is not None:
                    observation.update(
                        TraceMetadata(execution_ms=int((time.monotonic() - started) * 1000))
                    )

    async def _request(
        self, path: str, body: str | None, deadline: Deadline, timeout: float
    ) -> bytes:
        budget = deadline.budget(timeout)
        if budget <= 0:
            raise ModelDeadlineError()
        headers = {
            "Authorization": "Bearer " + self._settings.auth_token.get_secret_value(),
            "Content-Type": "application/json",
            "X-Request-ID": correlation_id.get() or uuid4().hex,
            "X-Request-Timeout-Ms": str(max(1, int(budget * 1000))),
        }
        try:
            # httpx timeouts bound each I/O stage; the outer timeout bounds the sum.
            async with asyncio.timeout(budget):
                response = await self._http.request(
                    "GET" if body is None else "POST",
                    str(self._settings.base_url).rstrip("/") + path,
                    content=body,
                    headers=headers,
                    timeout=budget,
                )
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise ModelDeadlineError() from exc
        except httpx.TransportError as exc:
            raise ModelError() from exc
        if response.status_code == HTTPStatus.OK:
            return response.content
        try:
            failure = ModelFailure.model_validate_json(response.content)
        except ValidationError as exc:
            raise ModelContractError() from exc
        types: dict[ModelFailureKind, type[ModelError]] = {
            ModelFailureKind.AUTH: ModelAuthError,
            ModelFailureKind.CONTRACT: ModelContractError,
            ModelFailureKind.DEADLINE: ModelDeadlineError,
            ModelFailureKind.OOM: ModelOOMError,
            ModelFailureKind.QUEUE_FULL: ModelQueueError,
            ModelFailureKind.INPUT: ModelInputError,
            ModelFailureKind.UNAVAILABLE: ModelError,
        }
        error_type = types.get(failure.code, ModelContractError)
        if (
            failure.retryable != error_type.retryable
            or response.status_code != error_type.http_status
        ):
            raise ModelContractError()
        raise error_type()

    async def ready(self, *, deadline: Deadline) -> ReadyResult:
        """Authenticated identity validation, not a forwarded TCP/liveness test."""
        raw = await self._call("/ready", None, deadline, 2)
        try:
            result = ReadyResult.model_validate_json(raw)
        except ValidationError as exc:
            raise ModelContractError() from exc
        self._identity(result.metadata)
        return result

    async def embed(self, texts: list[str], mode: EmbedMode, *, deadline: Deadline) -> EmbedResult:
        """Require a complete pair of dense/sparse outputs for every input text."""
        request = EmbedRequest(texts=texts, mode=mode)
        raw = await self._call(
            "/v1/embed", request.model_dump_json(), deadline, self._settings.embed_timeout_s
        )
        try:
            result = EmbedResult.model_validate_json(raw)
        except ValidationError as exc:
            raise ModelContractError() from exc
        self._identity(result.metadata)
        if len(result.dense) != len(texts) or len(result.sparse) != len(texts):
            raise ModelContractError()
        return result

    async def rerank(
        self, query: str, passages: list[str], *, deadline: Deadline, max_length: int = 320
    ) -> RerankResult:
        """Submit all candidates once, preserving service metadata and score order."""
        request = RerankRequest(query=query, passages=passages, max_length=max_length)
        raw = await self._call(
            "/v1/rerank", request.model_dump_json(), deadline, self._settings.rerank_timeout_s
        )
        try:
            result = RerankResult.model_validate_json(raw)
        except ValidationError as exc:
            raise ModelContractError() from exc
        self._identity(result.metadata)
        if len(result.scores) != len(passages) or result.metadata.rerank_max_length > max_length:
            raise ModelContractError()
        return result

    async def aclose(self) -> None:
        """Close only resources owned by this client."""
        if self._owns_http:
            await self._http.aclose()


class ModelRuntimeProbe:
    """Optional API dependency probe with its own bounded HTTP resource lifetime."""

    def __init__(self, settings: ModelRuntimeClientSettings) -> None:
        self.client = ModelRuntimeClient(settings)

    async def check(self) -> None:
        """Verify both loaded model identities within two seconds."""
        await self.client.ready(deadline=Deadline(time.monotonic() + 2))

    async def aclose(self) -> None:
        """Release the model dependency client's connection pool."""
        await self.client.aclose()
