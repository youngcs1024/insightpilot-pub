"""ASGI timeout/cancellation contracts, including responses already in progress."""

import asyncio
from time import monotonic

import httpx
import pytest
from fastapi import Request
from starlette.types import Message, Receive, Scope, Send

from app.application import create_app
from app.core.config_models import Settings
from app.core.deadline import Deadline
from app.core.middleware import DeadlineMiddleware, RequestIdMiddleware

GATEWAY_TIMEOUT = 504


async def test_timeout_returns_safe_504_with_request_id(settings: Settings) -> None:
    cancelled = asyncio.Event()
    app = create_app(settings)

    @app.get("/slow")
    async def slow(request: Request) -> None:
        assert isinstance(request.state.deadline, Deadline)
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    wrapped = RequestIdMiddleware(DeadlineMiddleware(app, timeout_s=0.02))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=wrapped), base_url="http://test"
    ) as client:
        response = await client.get("/slow")
    assert response.status_code == GATEWAY_TIMEOUT
    assert response.json()["code"] == "DEADLINE_EXCEEDED"
    assert response.json()["request_id"] == response.headers["X-Request-ID"]
    assert cancelled.is_set()


async def wait_receive() -> Message:
    await asyncio.Event().wait()
    return {"type": "http.disconnect"}


async def test_started_response_is_terminated_without_second_status() -> None:
    messages: list[Message] = []
    cancelled = asyncio.Event()

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"partial", "more_body": True})
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def send(message: Message) -> None:
        messages.append(message)

    await DeadlineMiddleware(app, 0.02)({"type": "http"}, wait_receive, send)
    assert [message["type"] for message in messages].count("http.response.start") == 1
    assert messages[-1] == {"type": "http.response.body", "body": b"", "more_body": False}
    assert cancelled.is_set()


async def test_external_cancellation_is_not_504() -> None:
    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        raise asyncio.CancelledError()

    async def send(message: Message) -> None:
        pytest.fail("external cancellation must not send a timeout response")

    with pytest.raises(asyncio.CancelledError):
        await DeadlineMiddleware(app, 1)({"type": "http"}, wait_receive, send)


async def test_unrelated_timeout_is_not_request_deadline() -> None:
    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        raise TimeoutError("upstream")

    async def send(message: Message) -> None:
        pytest.fail("unrelated timeout must propagate")

    with pytest.raises(TimeoutError, match="upstream"):
        await DeadlineMiddleware(app, 1)({"type": "http"}, wait_receive, send)


async def test_concurrent_requests_have_distinct_deadlines() -> None:
    deadlines: list[Deadline] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        deadlines.append(scope["state"]["deadline"])
        await asyncio.sleep(0)
        assert scope["state"]["deadline"].remaining() > 0

    async def send(message: Message) -> None:
        pass

    start = monotonic()
    middleware = DeadlineMiddleware(app, 1)
    async with asyncio.TaskGroup() as group:
        group.create_task(middleware({"type": "http"}, wait_receive, send))
        group.create_task(middleware({"type": "http"}, wait_receive, send))
    assert deadlines[0] is not deadlines[1]
    assert all(item.at >= start + 1 for item in deadlines)


async def test_timeout_during_header_send_never_sends_second_status() -> None:
    messages: list[Message] = []

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})

    async def send(message: Message) -> None:
        messages.append(message)
        if message["type"] == "http.response.start":
            await asyncio.Event().wait()

    await DeadlineMiddleware(app, 0.02)({"type": "http"}, wait_receive, send)
    assert [message["type"] for message in messages].count("http.response.start") == 1
    assert messages[-1]["type"] == "http.response.body"
