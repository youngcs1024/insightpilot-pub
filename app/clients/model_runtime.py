"""Authenticated model transport with bounded batching, retries and startup polling."""

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from functools import partial
from http import HTTPStatus
from uuid import uuid4

import httpx
import structlog
from asgi_correlation_id import correlation_id
from pydantic import ValidationError

from app.clients.model_resilience import (
    STARTUP_BUDGET_S,
    WARMUP_POLL_S,
    ModelCircuitBreaker,
    retryable_model_failure,
)
from app.core.config_models import ModelRuntimeClientSettings
from app.core.deadline import Deadline
from app.core.errors import DeadlineExceededError, OperationTimeoutError
from app.core.observability import TraceMetadata, observe
from app.core.retry import run_operation
from app.schemas.model_runtime import (
    EmbedBatchReceipt,
    EmbedMode,
    EmbedOutput,
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

logger = structlog.get_logger(__name__)


class ModelRuntimeClient:
    """One inference admission and one retry owner per batch, within a shared deadline."""

    def __init__(
        self, settings: ModelRuntimeClientSettings, http: httpx.AsyncClient | None = None
    ) -> None:
        self._settings = settings
        self._http = http or httpx.AsyncClient(trust_env=False)
        self._owns_http = http is None
        self._closed = False
        self._inference_slots = asyncio.Semaphore(settings.max_concurrency)
        self.breaker = ModelCircuitBreaker()

    @asynccontextmanager
    async def _inference(self, deadline: Deadline) -> AsyncIterator[None]:
        if deadline.remaining() <= 0:
            raise ModelDeadlineError()
        if self._closed:
            raise ModelError()
        self.breaker.check()
        try:
            async with asyncio.timeout(deadline.remaining()), self._inference_slots:
                if self._closed:
                    raise ModelError()
                async with self._admitted(deadline):
                    yield
        except (TimeoutError, DeadlineExceededError, OperationTimeoutError) as exc:
            raise ModelDeadlineError() from exc

    @asynccontextmanager
    async def _admitted(self, deadline: Deadline) -> AsyncIterator[None]:
        probe = self.breaker.enter()
        try:
            yield
        except ModelError as exc:
            if retryable_model_failure(exc):
                self.breaker.failed()
            raise
        else:
            if deadline.remaining() <= 0:
                raise ModelDeadlineError()
            self.breaker.success()
        finally:
            self.breaker.release(probe)

    async def _retry[T](
        self, operation: Callable[[], Awaitable[T]], deadline: Deadline, name: str
    ) -> tuple[T, int]:
        attempts = 0

        async def attempt() -> T:
            nonlocal attempts
            attempts += 1
            return await operation()

        result = await run_operation(
            attempt,
            deadline=deadline,
            timeout_s=deadline.remaining(),
            name=name,
            attempts=2,
            retry_policy=retryable_model_failure,
        )
        return result, attempts

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
        if self._closed:
            raise ModelError()
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
        except (TimeoutError, httpx.TimeoutException):
            raise ModelDeadlineError() from None
        except (httpx.NetworkError, httpx.ProxyError, httpx.RemoteProtocolError):
            raise ModelError() from None
        except httpx.TransportError:
            raise ModelContractError() from None
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

    async def warmup(self, *, deadline: Deadline) -> ReadyResult:
        """Poll readiness without an inference retry layer or renewed startup budget."""
        if self._closed:
            raise ModelError()
        started = time.monotonic()
        budget = Deadline(min(deadline.at, started + STARTUP_BUDGET_S))
        try:
            async with asyncio.timeout(budget.remaining()):
                result, polls = await self._warmup_until(budget)
        except TimeoutError:
            raise ModelDeadlineError() from None
        logger.info(
            "model_warmup_ready",
            polls=polls,
            duration_ms=int((time.monotonic() - started) * 1000),
        )
        return result

    async def _warmup_until(self, budget: Deadline) -> tuple[ReadyResult, int]:
        polls = 0
        while budget.remaining() > 0:
            polls += 1
            result = await self._ready_poll(budget)
            if result is not None:
                return result, polls
            await asyncio.sleep(min(WARMUP_POLL_S, budget.remaining()))
        raise ModelDeadlineError()

    async def _ready_poll(self, deadline: Deadline) -> ReadyResult | None:
        try:
            return await self.ready(deadline=deadline)
        except ModelError as exc:
            if not retryable_model_failure(exc) and not isinstance(exc, ModelDeadlineError):
                raise
        return None

    async def embed(self, texts: list[str], mode: EmbedMode, *, deadline: Deadline) -> EmbedOutput:
        """Return every vector in input order or fail the complete logical operation."""
        started = time.monotonic()
        if not texts:
            raise ModelInputError()
        try:
            requests = [
                EmbedRequest(texts=texts[start : start + self._settings.embed_batch], mode=mode)
                for start in range(0, len(texts), self._settings.embed_batch)
            ]
        except ValidationError:
            raise ModelInputError() from None
        async with self._inference(deadline):
            return await self._embed_batches(requests, deadline, started)

    async def _embed_batches(
        self, requests: list[EmbedRequest], deadline: Deadline, started: float
    ) -> EmbedOutput:
        dense: list[list[float]] = []
        sparse: list[dict[int, float]] = []
        receipts: list[EmbedBatchReceipt] = []
        for request in requests:
            response, attempts = await self._retry(
                partial(self._embed_batch, request, deadline),
                deadline,
                "model_embed",
            )
            if receipts:
                first = receipts[0].metadata
                if response.metadata.model_copy(update={"embed_batch": first.embed_batch}) != first:
                    raise ModelContractError()
            receipts.append(
                EmbedBatchReceipt(
                    **response.model_dump(exclude={"dense", "sparse"}),
                    start_index=len(dense),
                    text_count=len(request.texts),
                    attempts=attempts,
                )
            )
            dense.extend(response.dense)
            sparse.extend(response.sparse)
        return EmbedOutput(
            dense=dense,
            sparse=sparse,
            batches=receipts,
            client_ms=int((time.monotonic() - started) * 1000),
        )

    async def _embed_batch(self, request: EmbedRequest, deadline: Deadline) -> EmbedResult:
        raw = await self._call(
            "/v1/embed",
            request.model_dump_json(),
            deadline,
            min(20, self._settings.embed_timeout_s),
        )
        try:
            result = EmbedResult.model_validate_json(raw)
        except ValidationError as exc:
            raise ModelContractError() from exc
        self._identity(result.metadata)
        if len(result.dense) != len(request.texts) or len(result.sparse) != len(request.texts):
            raise ModelContractError()
        return result

    async def rerank(
        self, query: str, passages: list[str], *, deadline: Deadline, max_length: int = 320
    ) -> RerankResult:
        """Submit all candidates once, preserving service metadata and score order."""
        try:
            request = RerankRequest(query=query, passages=passages, max_length=max_length)
        except ValidationError:
            raise ModelInputError() from None
        async with self._inference(deadline):
            result, _ = await self._retry(
                lambda: self._rerank_once(request, deadline), deadline, "model_rerank"
            )
            return result

    async def _rerank_once(self, request: RerankRequest, deadline: Deadline) -> RerankResult:
        raw = await self._call(
            "/v1/rerank",
            request.model_dump_json(),
            deadline,
            min(30, self._settings.rerank_timeout_s),
        )
        try:
            result = RerankResult.model_validate_json(raw)
        except ValidationError as exc:
            raise ModelContractError() from exc
        self._identity(result.metadata)
        if (
            len(result.scores) != len(request.passages)
            or result.metadata.rerank_max_length > request.max_length
        ):
            raise ModelContractError()
        return result

    async def aclose(self) -> None:
        """Close only resources owned by this client."""
        self._closed = True
        if self._owns_http:
            await self._http.aclose()


class ModelRuntimeProbe:
    """Borrow the lifespan-owned client without affecting inference circuit state."""

    def __init__(self, client: ModelRuntimeClient) -> None:
        self.client = client

    async def check(self) -> None:
        """Verify both loaded model identities within two seconds."""
        await self.client.ready(deadline=Deadline(time.monotonic() + 2))

    async def aclose(self) -> None:
        """The API lifespan, rather than a borrowed health probe, closes the client."""
