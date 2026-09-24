"""Real PostgreSQL pending writes, durable partial answers and bounded HTTP/SSE."""

# ruff: noqa: PLR2004 -- explicit HTTP and snapshot commit-count acceptance values.

import asyncio
import time
from dataclasses import replace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.agents.contracts import Route
from app.agents.runtime import RuntimeContext
from app.agents.state import AgentState
from app.core.config_models import DatabaseSettings
from app.core.deadline import Deadline
from app.core.errors import ConflictError, DeadlineExceededError, EvidenceIntegrityError
from app.db.models import Turn
from app.db.session import Database
from app.repositories.turns import TurnRepository
from app.services import chat_stream
from app.services.memory.extract import MemoryExtractionService
from app.services.chat import AdmittedTurn, ChatService
from app.services.conversations import ConversationService
from app.services.evidence import EvidenceService
from app.services.graph import GraphService
from tests.agents.parent_support import parent_context
from tests.api.chat_support import Harness, chat, events
from tests.integration.checkpoint_support import admitted, checkpoint_setup, graph_database

pytestmark = pytest.mark.integration
__all__ = ["chat", "checkpoint_setup", "graph_database"]


class BlockedOperation:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def __call__(self, *args: object, **kwargs: object) -> None:
        self.entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            self.cancelled.set()


async def running_context(database: Database) -> RuntimeContext:
    identity = await admitted(database)
    async with database.session() as session, session.begin():
        row = await session.get(Turn, identity.turn_id)
        row.trace_id = identity.turn_id.hex
        question = await session.scalar(
            TurnRepository(session, identity.user_id).recent_topics(
                identity.conversation_id, row.seq
            )
        )
        question.content = "请分析2026年8月的经营情况"
    return replace(
        parent_context(Route.BOTH),
        identity=identity,
        trace_id=identity.turn_id.hex,
        conversations=ConversationService(database),
        evidence=EvidenceService(database),
    )


@pytest.mark.parametrize("slow", ["data", "knowledge", "synthesis"])
async def test_completed_pending_writes_survive_deadline(
    graph_database: tuple[Database, DatabaseSettings], monkeypatch: pytest.MonkeyPatch, slow: str
) -> None:
    database, settings = graph_database
    ctx = await running_context(database)
    block = BlockedOperation()
    if slow == "data":
        monkeypatch.setattr(ctx.mcp, "call_tool", block)
    elif slow == "knowledge":
        monkeypatch.setattr(ctx.retrieval, "retrieve", block)
    else:
        original = ctx.llm.generate_structured

        async def generate(
            role: object, messages: object, schema: type, **kwargs: object
        ) -> object:
            if schema.__name__ == "SynthesisOutput":
                return await block()
            return await original(role, messages, schema, **kwargs)

        monkeypatch.setattr(ctx.llm, "generate_structured", generate)
    graph = GraphService(settings)
    await graph.start()
    ctx = replace(ctx, deadline=Deadline(time.monotonic() + 2))
    service = ChatService(
        database, ctx.settings, graph,
        memory=MemoryExtractionService(database, ctx.settings, ctx.llm),
    )
    claim = AdmittedTurn(identity=ctx.identity, result=await service.read(ctx.identity))
    try:
        result = await service.execute(claim, ctx)
        assert block.entered.is_set()
        assert block.cancelled.is_set()
        assert result.status.value == "degraded"
        assert result.answer.synthesis.attempts == 0
        assert "deadline" in result.answer.degraded_components
        bundle = await ctx.evidence.read_bundle(ctx.identity)
        assert (bundle.data is not None) is (slow != "data")
        assert (bundle.knowledge is not None) is (slow != "knowledge")
        assert result.evidence_refs == bundle.refs
        checkpoint = await graph.graph.aget_state(
            {"configurable": {"thread_id": str(ctx.identity.turn_id)}}
        )
        state = AgentState.model_validate(checkpoint.values)
        assert state.data_evidence or state.knowledge_evidence
        assert time.monotonic() < ctx.finalization_deadline.at
        calls = (len(ctx.llm.calls), len(ctx.mcp.calls), len(ctx.retrieval.calls))
        replay = await service.execute(
            AdmittedTurn(
                identity=ctx.identity, result=result.model_copy(update={"replayed": True})
            ),
            ctx,
        )
        assert replay.answer == result.answer
        assert calls == (len(ctx.llm.calls), len(ctx.mcp.calls), len(ctx.retrieval.calls))
    finally:
        await graph.aclose()


@pytest.mark.parametrize("damage", ["write", "integrity", "grace"])
async def test_finalization_failure_never_commits_answer(
    graph_database: tuple[Database, DatabaseSettings], monkeypatch: pytest.MonkeyPatch, damage: str
) -> None:
    database, settings = graph_database
    ctx = await running_context(database)
    monkeypatch.setattr(ctx.mcp, "call_tool", BlockedOperation())
    original = ctx.evidence.commit_bundle

    async def commit(*args: object) -> object:
        if damage == "write":
            raise ConflictError()
        if damage == "integrity":
            raise EvidenceIntegrityError()
        await asyncio.Event().wait()
        return await original(*args)

    monkeypatch.setattr(ctx.evidence, "commit_bundle", commit)
    ctx.settings.http.finalization_grace_s = 0.3
    graph = GraphService(settings)
    await graph.start()
    ctx = replace(ctx, deadline=Deadline(time.monotonic() + 2))
    service = ChatService(
        database, ctx.settings, graph,
        memory=MemoryExtractionService(database, ctx.settings, ctx.llm),
    )
    try:
        claim = AdmittedTurn(identity=ctx.identity, result=await service.read(ctx.identity))
        with pytest.raises((ConflictError, EvidenceIntegrityError, DeadlineExceededError)):
            await service.execute(claim, ctx)
        stored = await service.read(ctx.identity)
        assert stored.status.value == "failed"
        assert stored.answer is None
    finally:
        await graph.aclose()


@pytest.mark.parametrize("field", ["user_id", "turn_id", "conversation_id", "graph_version"])
async def test_deadline_recovery_validates_checkpoint_identity(field: str) -> None:
    ctx = parent_context(Route.BOTH)
    state = AgentState(**ctx.identity.model_dump()).model_dump()
    state[field] = "old-version" if field == "graph_version" else uuid4()
    with pytest.raises(ConflictError):
        GraphService._owned_state(state, ctx)


@pytest.mark.parametrize("streaming", [False, True])
async def test_http_and_sse_return_committed_partial_and_replay(
    chat: Harness,
    checkpoint_setup: None,
    graph_database: tuple[Database, DatabaseSettings],
    monkeypatch: pytest.MonkeyPatch,
    streaming: bool,
) -> None:
    _, settings = graph_database
    graph = GraphService(settings)
    await graph.start()
    ctx = parent_context(Route.BOTH, settings=chat.app.state.settings)
    block = BlockedOperation()
    monkeypatch.setattr(chat.app.state.mcp, "call_tool", block)
    chat.app.state.chat.graph = graph
    chat.app.state.llm = ctx.llm
    chat.app.state.retrieval = ctx.retrieval
    chat.app.state.knowledge_generation = ctx.knowledge_generation
    for middleware in chat.app.user_middleware:
        if middleware.cls.__name__ == "DeadlineMiddleware":
            middleware.kwargs["timeout_s"] = 2
    chat.app.middleware_stack = None
    monkeypatch.setattr("app.services.chat_stream.HEARTBEAT_SECONDS", 0.1)
    committed = asyncio.Event()
    original_commit = chat.app.state.chat._succeed
    original_frame = chat_stream.frame

    async def commit(*args: object) -> None:
        await original_commit(*args)
        committed.set()

    def frame(name: str, payload: object) -> str:
        if name in {"token", "done"}:
            assert committed.is_set(), "answer bytes escaped before commit"
        return original_frame(name, payload)

    monkeypatch.setattr(chat.app.state.chat, "_succeed", commit)
    monkeypatch.setattr(chat_stream, "frame", frame)
    url = chat.url + ("/stream" if streaming else "")
    try:
        response = await chat.client.post(
            url,
            json={"content": "请分析2026年8月经营情况"},
            headers={"Idempotency-Key": "deadline-partial"},
        )
        assert response.status_code == 200, response.text
        frames = events(response) if streaming else []
        result = frames[-1][1] if streaming else response.json()
        if streaming:
            assert sum(name in {"done", "error"} for name, _ in frames) == 1
            assert frames[-1][0] == "done"
            assert (
                "".join(value["delta"] for name, value in frames if name == "token")
                == result["answer"]["markdown"]
            )
        assert result["status"] == "degraded"
        assert "data" in result["answer"]["degraded_components"]
        assert "deadline" in result["answer"]["degraded_components"]
        assert (await chat.stored())[-1]["answer"] == result["answer"]
        assert block.cancelled.is_set()
        graph.invoke = AsyncMock(side_effect=AssertionError("replay executed graph"))
        replay = await chat.client.post(
            chat.url,
            json={"content": "请分析2026年8月经营情况"},
            headers={"Idempotency-Key": "deadline-partial"},
        )
        assert replay.json()["answer"] == result["answer"]
        graph.invoke.assert_not_awaited()
    finally:
        await graph.aclose()


async def test_commit_before_checkpoint_receipt_is_reused(
    graph_database: tuple[Database, DatabaseSettings], monkeypatch: pytest.MonkeyPatch
) -> None:
    database, settings = graph_database
    ctx = await running_context(database)
    original = ctx.evidence.commit_bundle
    committed = []
    block = BlockedOperation()

    async def commit(*args: object) -> object:
        bundle = await original(*args)
        committed.append(bundle)
        if len(committed) == 1:
            await block()
        return bundle

    monkeypatch.setattr(ctx.evidence, "commit_bundle", commit)
    graph = GraphService(settings)
    await graph.start()
    ctx = replace(ctx, deadline=Deadline(time.monotonic() + 2))
    try:
        output = await graph.invoke(ctx)
        assert output.status == "degraded"
        assert block.cancelled.is_set()
        assert len(committed) == 2
        assert committed[0] == committed[1]
        assert output.evidence_refs == committed[0].refs
    finally:
        await graph.aclose()
