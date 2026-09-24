"""Shared chat harness and ASGI stream driver with explicit PostgreSQL fixtures."""

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from langchain_core.callbacks import BaseCallbackHandler
from starlette.types import Message

from app.agents.contracts import Route, RouteDecision
from app.agents.runtime import RuntimeContext
from app.agents.state import GraphOutput
from app.api.dependencies import get_current_user
from app.application import create_app
from app.core.config_models import Settings
from app.core.background import shutdown
from app.schemas.memory_extraction import MemoryExtraction
from app.db.models import TurnStatus
from app.db.session import Database
from app.schemas.auth import UserResponse
from tests.agents.support import invoke, metric_intent, sql_candidate
from tests.answer_support import data_draft
from tests.factories import business_schema
from tests.factories import query_result as result
from tests.fakes.chat_model import FakeChatModel
from tests.fakes.mcp_client import FakeMcpClient

OK, CREATED = 200, 201


class ControlledGraph:
    """Run the actual Phase 1 nodes, with observable gates around graph execution."""

    def __init__(self) -> None:
        self.calls = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.cancelled = asyncio.Event()
        self.error: Exception | None = None
        self.last: RuntimeContext | None = None

    async def invoke(
        self, ctx: RuntimeContext, *, callbacks: list[BaseCallbackHandler] | None = None
    ) -> GraphOutput:
        self.calls += 1
        self.last = ctx
        self.entered.set()
        try:
            await self.release.wait()
            if self.error:
                raise self.error
            return await invoke(ctx, callbacks)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


@dataclass
class Harness:
    app: FastAPI
    client: httpx.AsyncClient
    graph: ControlledGraph
    database: Database
    user: UserResponse
    cid: UUID

    @property
    def url(self) -> str:
        return f"/api/v1/conversations/{self.cid}/messages"

    async def stored(self) -> list[dict[str, object]]:
        response = await self.client.get(f"/api/v1/conversations/{self.cid}/turns")
        assert response.status_code == OK
        return response.json()["items"]


@pytest.fixture
async def chat(settings: Settings, auth_database: Database) -> AsyncIterator[Harness]:
    graph = ControlledGraph()
    mcp = FakeMcpClient(
        [result() for _ in range(10)],
        schema_responses=[
            business_schema(),
            *[business_schema() for _ in range(19)],
        ],
    )
    application = create_app(settings, database=auth_database, mcp_client=mcp)
    application.state.chat.graph = graph
    application.state.chat.memory.llm = FakeChatModel([MemoryExtraction() for _ in range(20)])
    application.state.llm = FakeChatModel(
        [
            RouteDecision(route=Route.DATA_ONLY, confidence=1, data_intent="2026年8月GMV"),
            metric_intent(),
            sql_candidate(),
            data_draft(markdown="订单总数为 42。\nSQL 已执行。" * 20, confidence=0.9),
        ]
        * 10
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/auth/register",
            json={
                "email": f"{uuid4().hex}@example.com",
                "password": "Strong-pass-123!",
                "display_name": "Chat",
            },
        )
        assert response.status_code == CREATED, response.text
        user = UserResponse.model_validate(response.json())

        async def identity() -> UserResponse:
            return user

        application.dependency_overrides[get_current_user] = identity
        response = await client.post("/api/v1/conversations", json={})
        assert response.status_code == CREATED, response.text
        try:
            yield Harness(application, client, graph, auth_database, user, UUID(response.json()["id"]))
        finally:
            await shutdown(settings.http.shutdown_timeout_s)


def events(response: httpx.Response) -> list[tuple[str, dict[str, object]]]:
    return parse_events(response.text)


def parse_events(body: str) -> list[tuple[str, dict[str, object]]]:
    parsed = []
    for block in body.split("\n\n"):
        if block:
            name, data = block.split("\n", 1)
            parsed.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return parsed


async def asgi_stream(
    chat: Harness,
    mode: Literal["disconnect", "committed", "complete", "socket"],
    monkeypatch: pytest.MonkeyPatch,
) -> list[Message]:
    """Drive actual receive/send independently; ASGITransport buffers the response."""
    monkeypatch.setattr("app.services.chat_stream.HEARTBEAT_SECONDS", 0.01)
    inbox: asyncio.Queue[Message] = asyncio.Queue()
    await inbox.put({"type": "http.request", "body": b'{"content":"count"}', "more_body": False})
    sent: list[Message] = []

    async def send(message: Message) -> None:
        sent.append(message)
        body = message.get("body", b"")
        if body.startswith(b"event: heartbeat"):
            if mode == "socket":
                raise OSError("client socket closed")
            if mode == "disconnect":
                await inbox.put({"type": "http.disconnect"})
            elif mode == "complete":
                chat.graph.release.set()
        if body.startswith(b"event: token"):
            assert chat.graph.last is not None
            stored = await chat.app.state.chat.read(chat.graph.last.identity)
            assert stored.status == TurnStatus.SUCCEEDED
            assert stored.evidence_refs.data_snapshot_id is not None
            if mode == "committed":
                await inbox.put({"type": "http.disconnect"})
                await asyncio.Event().wait()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": chat.url + "/stream",
        "raw_path": (chat.url + "/stream").encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1234),
        "server": ("test", 80),
    }
    async with asyncio.timeout(10):
        await chat.app(scope, inbox.get, send)
    return sent
