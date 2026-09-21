"""Pure ASGI middleware preserving streaming, cancellation, and request context."""

import asyncio
from time import monotonic
from uuid import UUID, uuid4

import structlog
from asgi_correlation_id import CorrelationIdMiddleware, correlation_id
from fastapi import Request
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.exception_handlers import handle_known
from app.core.deadline import Deadline, ResponseBudget
from app.core.errors import DeadlineExceededError
from app.schemas.chat import ErrorEvent

logger = structlog.get_logger(__name__)


class RequestIdMiddleware(CorrelationIdMiddleware):
    """Use library validation/headers and restore its ContextVar even after errors."""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app, validator=None, transformer=lambda value: UUID(value).hex)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        token = correlation_id.set(None)
        try:
            if scope["type"] in {"http", "websocket"}:
                headers = MutableHeaders(scope=scope)
                try:
                    supplied = UUID(headers.get("X-Request-ID", ""))
                except ValueError:
                    supplied = uuid4()
                headers["X-Request-ID"] = supplied.hex if supplied.int else uuid4().hex
            await super().__call__(scope, receive, send)
        finally:
            if scope["type"] == "http":
                scope.setdefault("state", {})["request_id"] = correlation_id.get()
            correlation_id.reset(token)


class DeadlineMiddleware:
    """Enforce one request budget without buffering streaming responses."""

    def __init__(self, app: ASGIApp, timeout_s: float, finalization_grace_s: float = 0) -> None:
        self.app = app
        self.timeout_s = timeout_s
        self.finalization_grace_s = finalization_grace_s

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        deadline = Deadline(monotonic() + self.timeout_s)
        scope.setdefault("state", {})["deadline"] = deadline
        started = False
        finished = False
        terminal = False

        async def track_send(message: Message) -> None:
            nonlocal started, finished, terminal
            # Sending headers can be cancelled after the server has accepted them.
            # Mark the attempt first so timeout handling never sends a second status.
            if message["type"] == "http.response.start":
                started = True
            elif message["type"] == "http.response.body" and not message.get("more_body", False):
                finished = True
            terminal = terminal or _is_terminal_frame(message, scope)
            await send(message)

        timer = asyncio.timeout(deadline.remaining())
        scope["state"]["response_budget"] = ResponseBudget(
            timer, Deadline(deadline.at + self.finalization_grace_s)
        )
        try:
            async with timer:
                await self.app(scope, receive, track_send)
        except TimeoutError:
            if not timer.expired():
                raise
            error = DeadlineExceededError()
            logger.warning("request_deadline_exceeded", response_started=started)
            if not started:
                response = await handle_known(Request(scope), error)
                await response(scope, receive, send)
            elif not finished:
                await _stream_deadline(scope, send, terminal, error)
                await send({"type": "http.response.body", "body": b"", "more_body": False})


class LoggingContextMiddleware:
    """Bind trusted request metadata and always clear it after request completion."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=correlation_id.get())
        status_code = 500

        async def capture_status(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, capture_status)
        except Exception:
            logger.exception("request_failed")
            raise
        finally:
            # Only authenticated/owned dependency state may supply identity fields.
            state = scope.get("state", {})
            for field in ("user_id", "conversation_id"):
                if state.get(field) is not None:
                    structlog.contextvars.bind_contextvars(**{field: str(state[field])})
            logger.info("request_completed", status_code=status_code, method=scope["method"])
            structlog.contextvars.clear_contextvars()


async def _stream_deadline(
    scope: Scope, send: Send, terminal: bool, error: DeadlineExceededError
) -> None:
    if not scope["state"].get("chat_stream") or terminal:
        return
    payload = ErrorEvent(code=error.code, message=error.user_message)
    await send(
        {
            "type": "http.response.body",
            "body": ("event: error\ndata: " + payload.model_dump_json() + "\n\n").encode(),
            "more_body": True,
        }
    )


def _is_terminal_frame(message: Message, scope: Scope) -> bool:
    return bool(
        scope["state"].get("chat_stream")
        and message["type"] == "http.response.body"
        and message.get("body", b"").startswith((b"event: done\n", b"event: error\n"))
    )
