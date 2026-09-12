"""Authenticated model service; HTTP stays responsive during synchronous CUDA work."""

import asyncio
import hmac
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import partial
from typing import cast
from uuid import UUID, uuid4

import structlog
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.background import spawn
from app.schemas.model_runtime import (
    EmbedMode,
    EmbedRequest,
    EmbedResult,
    ModelFailure,
    ModelFailureKind,
    ReadyResult,
    RerankRequest,
    RerankResult,
)
from model_runtime.config import ModelRuntimeSettings, ModelServerSettings
from model_runtime.errors import (
    ModelAuthError,
    ModelContractError,
    ModelDeadlineError,
    ModelError,
    ModelInputError,
)
from model_runtime.executor import InferenceExecutor
from model_runtime.models import CudaModels, ModelBackend

logger = structlog.get_logger(__name__)
BODY_LIMIT = 8 * 1024 * 1024


def error_response(error: ModelError, request_id: str) -> JSONResponse:
    """Expose a fixed typed envelope, never the exception detail or request input."""
    headers = {"X-Request-ID": request_id}
    if error.retryable:
        headers["Retry-After"] = "1"
    failure = ModelFailure(
        code=error.kind,
        message=error.user_message,
        request_id=request_id,
        retryable=error.retryable,
    )
    return JSONResponse(
        failure.model_dump(mode="json"), status_code=error.http_status, headers=headers
    )


class RequestBoundary:
    """Limit streamed bytes before JSON parsing and start deadlines on receipt."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = time.monotonic()
        headers = dict(scope["headers"])
        request_id = self._request_id(headers)
        limit = 20.0 if scope["path"] == "/v1/embed" else 30.0
        try:
            raw_timeout = headers.get(b"x-request-timeout-ms", str(int(limit * 1000)).encode())
            timeout_ms = int(raw_timeout)
            if timeout_ms <= 0:
                raise ModelInputError()
            at = started + min(limit, timeout_ms / 1000)
            async with asyncio.timeout_at(at):
                body = await self._body(receive, headers)
        except TimeoutError:
            await error_response(ModelDeadlineError(), request_id)(scope, receive, send)
            return
        except (ValueError, ModelInputError):
            response = error_response(ModelInputError(), request_id)
            await response(scope, receive, send)
            return
        scope.setdefault("state", {}).update(request_id=request_id, started=started, at=at)
        consumed = False

        async def replay() -> Message:
            nonlocal consumed
            if consumed:
                return await receive()
            consumed = True
            return {"type": "http.request", "body": body, "more_body": False}

        async def correlate(message: Message) -> None:
            if message["type"] == "http.response.start":
                message["headers"] = [
                    (key, value)
                    for key, value in message.get("headers", [])
                    if key.lower() != b"x-request-id"
                ] + [(b"x-request-id", request_id.encode())]
            await send(message)

        await self.app(scope, replay, correlate)

    @staticmethod
    def _request_id(headers: dict[bytes, bytes]) -> str:
        try:
            value = UUID(headers.get(b"x-request-id", b"").decode())
            return value.hex if value.int else uuid4().hex
        except (ValueError, UnicodeError):
            return uuid4().hex

    @staticmethod
    async def _body(receive: Receive, headers: dict[bytes, bytes]) -> bytes:
        if int(headers.get(b"content-length", b"0")) > BODY_LIMIT:
            raise ModelInputError()
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                raise ModelInputError()
            body.extend(message.get("body", b""))
            if len(body) > BODY_LIMIT:
                raise ModelInputError()
            if not message.get("more_body", False):
                return bytes(body)


class Liveness(BaseModel):
    """Only process liveness is public."""

    status: str = "ok"
    request_id: str


class ModelService:
    """Own model state, request admission and synchronous adapter execution."""

    def __init__(self, settings: ModelServerSettings, backend: ModelBackend | None) -> None:
        self.settings = settings
        self.models = backend or CudaModels(settings)
        self.executor = InferenceExecutor(settings.queue_capacity)
        self.loaded = False

    def warmup(self) -> None:
        self.models.load()
        at = time.monotonic() + 150
        self.models.embed(EmbedRequest(texts=["七天无理由退货政策"], mode=EmbedMode.QUERY), at)
        self.models.rerank(RerankRequest(query="退款政策", passages=["七天内可以申请退货"]), at)

    @asynccontextmanager
    async def lifespan(self, app: FastAPI) -> AsyncIterator[None]:
        try:
            await self.executor.submit(self.warmup, time.monotonic() + 180)
            self.loaded = app.state.ready = True
            logger.info("model_runtime_ready", metadata=self.models.metadata().model_dump())
            yield
        finally:
            self.loaded = app.state.ready = False
            await self.executor.aclose()

    def authenticate(self, request: Request) -> None:
        supplied = request.headers.get("authorization", "")
        expected = "Bearer " + self.settings.auth_token.get_secret_value()
        if not hmac.compare_digest(supplied.encode(), expected.encode()):
            raise ModelAuthError()
        if not self.loaded:
            raise ModelError()

    async def health(self, request: Request) -> Liveness:
        return Liveness(request_id=request.state.request_id)

    async def ready(self, request: Request) -> ReadyResult:
        self.authenticate(request)
        return ReadyResult(request_id=request.state.request_id, metadata=self.models.metadata())

    def encode(self, payload: EmbedRequest, request: Request) -> EmbedResult:
        began = time.monotonic()
        values, metadata = self.models.embed(payload, request.state.at)
        if len(values.dense) != len(payload.texts) or len(values.sparse) != len(payload.texts):
            raise ModelContractError()
        ended = time.monotonic()
        return EmbedResult(
            dense=values.dense,
            sparse=values.sparse,
            metadata=metadata,
            request_id=request.state.request_id,
            ms=int((ended - request.state.started) * 1000),
            queue_ms=int((began - request.state.started) * 1000),
            inference_ms=int((ended - began) * 1000),
        )

    def rank(self, payload: RerankRequest, request: Request) -> RerankResult:
        began = time.monotonic()
        values, metadata = self.models.rerank(payload, request.state.at)
        if len(values.scores) != len(payload.passages):
            raise ModelContractError()
        ended = time.monotonic()
        return RerankResult(
            scores=values.scores,
            metadata=metadata,
            request_id=request.state.request_id,
            ms=int((ended - request.state.started) * 1000),
            queue_ms=int((began - request.state.started) * 1000),
            inference_ms=int((ended - began) * 1000),
        )

    async def execute(self, call: Callable[[], object], request: Request) -> object:
        async def operation() -> object:
            try:
                return await self.executor.submit(call, request.state.at)
            except ModelError as error:
                return error
            except Exception as error:
                logger.exception(
                    "model_adapter_failed", exception_type=type(error).__name__, exc_info=False
                )
                return ModelContractError()

        work = spawn(operation(), name="model-http-operation")

        async def disconnected() -> None:
            while True:
                message = await request.receive()
                if message["type"] == "http.disconnect":
                    work.cancel()
                    return

        monitor = spawn(disconnected(), name="model-disconnect-monitor")
        try:
            result = await work
            if isinstance(result, ModelError):
                raise result
            return result
        finally:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)

    async def embed(self, payload: EmbedRequest, request: Request) -> EmbedResult:
        self.authenticate(request)
        return cast(
            "EmbedResult", await self.execute(partial(self.encode, payload, request), request)
        )

    async def rerank(self, payload: RerankRequest, request: Request) -> RerankResult:
        self.authenticate(request)
        return cast(
            "RerankResult", await self.execute(partial(self.rank, payload, request), request)
        )


async def known(request: Request, exc: Exception) -> JSONResponse:
    """Map only fixed project failures into the model wire envelope."""
    return error_response(cast("ModelError", exc), request.state.request_id)


async def invalid(request: Request, exc: Exception) -> JSONResponse:
    """Validation never echoes supplied input values."""
    return error_response(ModelInputError(), request.state.request_id)


async def unknown(request: Request, exc: Exception) -> JSONResponse:
    """Log exception identity only; SDK prose can contain model input."""
    logger.error("model_runtime_internal_error", exception_type=type(exc).__name__)
    failure = ModelFailure(
        code=ModelFailureKind.INTERNAL,
        message="Model operation failed.",
        request_id=request.state.request_id,
        retryable=False,
    )
    return JSONResponse(failure.model_dump(mode="json"), status_code=500)


def create_app(settings: ModelServerSettings, backend: ModelBackend | None = None) -> FastAPI:
    """Construct without loading models; lifespan owns startup and draining."""
    service = ModelService(settings, backend)
    app = FastAPI(lifespan=service.lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.ready = False
    app.add_middleware(RequestBoundary)
    app.add_exception_handler(ModelError, known)
    app.add_exception_handler(RequestValidationError, invalid)
    app.add_exception_handler(HTTPException, invalid)
    app.add_exception_handler(Exception, unknown)
    app.add_api_route("/health", service.health, methods=["GET"])
    app.add_api_route("/ready", service.ready, methods=["GET"])
    app.add_api_route("/v1/embed", service.embed, methods=["POST"])
    app.add_api_route("/v1/rerank", service.rerank, methods=["POST"])
    return app


def main() -> None:
    """One process only; no reload or worker fan-out can duplicate GPU allocations."""
    settings = ModelRuntimeSettings.load()
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )
    uvicorn.run(
        create_app(settings.model_server),
        host="0.0.0.0",  # noqa: S104 -- container only; Compose publishes loopback.
        port=8100,
        workers=1,
        access_log=False,
        timeout_graceful_shutdown=35,
    )


if __name__ == "__main__":
    main()
