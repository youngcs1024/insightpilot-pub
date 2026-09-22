"""Real PostgreSQL lifecycle with deterministic graph and controlled ASGI transport."""

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from app.agents.contracts import Route, RouteDecision, TurnIdentity
from app.agents.failures import FailureKind
from app.agents.runtime import RuntimeContext
from app.agents.state import GraphOutput
from app.api.dependencies import get_current_user
from app.application import create_app
from app.core.config_models import RateRule
from app.core.errors import InsightPilotError, McpUnavailableError
from app.db.models import TurnStatus
from app.repositories import lifecycle
from app.repositories.turns import TurnRepository
from app.schemas.auth import UserResponse
from app.services.chat import ChatService
from app.services.graph import GraphService
from app.services.idempotency import IdempotencyService, MessageAdmission
from scripts.migration_settings import MigrationSettings
from scripts.setup_checkpointer import setup_checkpointer
from tests.agents.support import metric_intent
from tests.api.chat_support import Harness, asgi_stream, chat, events, parse_events
from tests.auth_support import deadline
from tests.fakes.chat_model import FakeChatModel
from tests.fakes.mcp_client import FakeMcpClient

if TYPE_CHECKING:
    import httpx

pytestmark = pytest.mark.integration
__all__ = ["chat"]
OK, CREATED, ACCEPTED, CONFLICT, NOT_FOUND, INVALID, LIMITED = 200, 201, 202, 409, 404, 422, 429
ORIGINAL_RECONCILE = ChatService.reconcile


async def test_sync_message_returns_answer_with_sql(chat: Harness) -> None:
    response = await chat.client.post(chat.url, json={"content": "订单数?"})
    assert response.status_code == OK, response.text
    body = response.json()
    assert body["answer"]["sql"] == "SELECT 42 LIMIT 1001"
    assert body["answer"]["evidence_refs"] == body["evidence_refs"]
    assert body["status"] == "succeeded"


async def test_turn_persisted_with_status_and_trace_id(chat: Harness) -> None:
    trace = uuid4().hex
    response = await chat.client.post(
        chat.url, json={"content": "count"}, headers={"X-Request-ID": trace}
    )
    rows = await chat.stored()
    assert len(rows) == 2  # noqa: PLR2004 -- user/assistant pair.
    assert rows[1]["trace_id"] == trace == response.json()["trace_id"]
    assert rows[1]["answer"] == response.json()["answer"]
    assert rows[1]["status"] == "succeeded"
    assert rows[1]["latency_ms"] is not None


async def test_stream_emits_answer_after_evidence_then_done(chat: Harness) -> None:
    response = await chat.client.post(chat.url + "/stream", json={"content": "count"})
    assert response.status_code == OK
    frames = events(response)
    assert frames[-1][0] == "done"
    assert all(name == "token" for name, _ in frames[:-1])
    assert "".join(str(payload["delta"]) for _, payload in frames[:-1]) == frames[-1][1]["content"]
    assert response.headers["x-accel-buffering"] == "no"
    assert response.headers["cache-control"] == "no-cache"


async def test_replay_returns_assistant_not_user_message(chat: Harness) -> None:
    headers = {"Idempotency-Key": "same-key", "X-Request-ID": uuid4().hex}
    first = await chat.client.post(chat.url, json={"content": "count"}, headers=headers)
    headers["X-Request-ID"] = uuid4().hex
    second = await chat.client.post(chat.url, json={"content": "count"}, headers=headers)
    assert second.headers["Idempotency-Replayed"] == "true"
    assert second.json()["answer"] == first.json()["answer"]
    assert second.json()["id"] == first.json()["id"] != first.json()["reply_to_turn_id"]
    assert second.json()["trace_id"] == first.json()["trace_id"]
    assert second.json()["request_id"] == headers["X-Request-ID"]
    assert chat.graph.calls == 1
    replay = await chat.client.post(
        chat.url + "/stream", json={"content": "count"}, headers=headers
    )
    assert events(replay)[-1][1]["answer"] == first.json()["answer"]
    assert chat.graph.calls == 1


async def test_running_key_does_not_reinvoke_graph(chat: Harness) -> None:
    chat.graph.release.clear()
    async with asyncio.TaskGroup() as group:
        original = group.create_task(
            chat.client.post(
                chat.url, json={"content": "count"}, headers={"Idempotency-Key": "key"}
            )
        )
        await chat.graph.entered.wait()
        try:
            for suffix in ("", "/stream"):
                replay = await chat.client.post(
                    chat.url + suffix, json={"content": "count"}, headers={"Idempotency-Key": "key"}
                )
                assert replay.status_code == ACCEPTED, replay.text
                assert replay.json()["status"] == "running"
                assert replay.json()["replayed"]
        finally:
            chat.graph.release.set()
    assert original.result().status_code == OK
    assert chat.graph.calls == 1


async def test_concurrent_turns_in_same_conversation_conflict(chat: Harness) -> None:
    chat.graph.release.clear()
    async with asyncio.TaskGroup() as group:
        original = group.create_task(chat.client.post(chat.url, json={"content": "count"}))
        await chat.graph.entered.wait()
        try:
            response = await chat.client.post(chat.url + "/stream", json={"content": "different"})
            assert response.status_code == CONFLICT
            assert "text/event-stream" not in response.headers["content-type"]
        finally:
            chat.graph.release.set()
    assert original.result().status_code == OK


async def test_same_key_different_content_conflict(chat: Harness) -> None:
    await chat.client.post(chat.url, json={"content": "count"}, headers={"Idempotency-Key": "key"})
    response = await chat.client.post(
        chat.url, json={"content": "different"}, headers={"Idempotency-Key": "key"}
    )
    assert response.status_code == CONFLICT
    assert chat.graph.calls == 1


async def test_stream_error_becomes_terminal_frame_not_500(chat: Harness) -> None:
    chat.graph.error = InsightPilotError("private connection detail")
    response = await chat.client.post(chat.url + "/stream", json={"content": "count"})
    assert response.status_code == OK
    assert [name for name, _ in events(response)] == ["error"]
    assert (await chat.stored())[1]["status"] == "failed"


async def test_stream_error_frame_contains_no_internal_detail(chat: Harness) -> None:
    chat.graph.error = RuntimeError("secret-internal-marker")
    response = await chat.client.post(chat.url + "/stream", json={"content": "count"})
    assert "secret-internal-marker" not in response.text
    assert set(events(response)[-1][1]) == {"code", "message"}


async def test_failed_original_replays_without_graph(chat: Harness) -> None:
    chat.graph.error = InsightPilotError()
    await chat.client.post(
        chat.url, json={"content": "count"}, headers={"Idempotency-Key": "failed"}
    )
    replay = await chat.client.post(
        chat.url, json={"content": "count"}, headers={"Idempotency-Key": "failed"}
    )
    assert replay.status_code == OK
    assert replay.json()["status"] == "failed"
    assert chat.graph.calls == 1


async def test_history_evidence_requires_no_external_calls(chat: Harness) -> None:
    response = await chat.client.post(chat.url, json={"content": "count"})
    tid = response.json()["id"]
    url = f"/api/v1/conversations/{chat.cid}/turns/{tid}/evidence"
    first = await chat.client.get(url)
    chat.app.state.llm = AsyncMock()
    chat.app.state.mcp = AsyncMock()
    second = await chat.client.get(url)
    assert first.json()["data"] == second.json()["data"]
    assert second.json()["data"]["id"] == response.json()["evidence_refs"]["data_snapshot_id"]
    assert not chat.app.state.llm.mock_calls
    assert not chat.app.state.mcp.mock_calls


async def test_client_disconnect_finalizes_turn(
    chat: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat.graph.release.clear()
    await asgi_stream(chat, "disconnect", monkeypatch)
    assert chat.graph.cancelled.is_set()
    rows = await chat.stored()
    assert rows[1]["status"] == "failed"
    assert rows[1]["failure_reason"] == "client_disconnected"
    assert rows[1]["latency_ms"] is not None


async def test_disconnect_after_commit_preserves_success(
    chat: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    await asgi_stream(chat, "committed", monkeypatch)
    assert (await chat.stored())[1]["status"] == "succeeded"


async def test_evidence_persisted_before_answer(
    chat: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat.graph.release.clear()
    sent = await asgi_stream(chat, "complete", monkeypatch)
    frames = parse_events(b"".join(item.get("body", b"") for item in sent).decode())
    assert frames[0][0] == "heartbeat"
    assert frames[-1][0] == "done"


async def test_answer_not_emitted_before_snapshot_commit(
    chat: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = chat.app.state.evidence
    original = evidence.commit_bundle
    committed = asyncio.Event()

    async def commit(identity: TurnIdentity, data: object, knowledge: object) -> object:
        snapshot = await original(identity, data, knowledge)
        committed.set()
        return snapshot

    monkeypatch.setattr(evidence, "commit_bundle", commit)
    await asgi_stream(chat, "complete", monkeypatch)
    assert committed.is_set()


async def test_deadline_emits_single_error_and_finalizes(
    chat: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Rebuild middleware after changing the typed setting.
    for middleware in chat.app.user_middleware:
        if middleware.cls.__name__ == "DeadlineMiddleware":
            middleware.kwargs["timeout_s"] = 0.15
    chat.app.middleware_stack = None
    chat.graph.release.clear()
    monkeypatch.setattr("app.services.chat_stream.HEARTBEAT_SECONDS", 0.01)
    response = await chat.client.post(chat.url + "/stream", json={"content": "count"})
    frames = events(response)
    assert [name for name, _ in frames].count("error") == 1
    assert frames[-1][0] == "error"
    assert frames[-1][1]["code"] == "DEADLINE_EXCEEDED"
    assert chat.graph.cancelled.is_set()
    assert (await chat.stored())[1]["failure_reason"] == "deadline_exceeded"


async def test_interrupted_turn_finalized_on_startup(chat: Harness) -> None:
    admission = await IdempotencyService(chat.database, 10).admit(
        MessageAdmission(user_id=chat.user.id, conversation_id=chat.cid, content="abandoned"),
        deadline=deadline(),
    )
    await ORIGINAL_RECONCILE(chat.app.state.chat)
    stored = await chat.app.state.chat.read(
        TurnIdentity(user_id=chat.user.id, conversation_id=chat.cid, turn_id=admission.id)
    )
    assert stored.status == TurnStatus.FAILED
    assert stored.failure_reason == FailureKind.INTERRUPTED
    assert stored.latency_ms is not None
    assert chat.graph.calls == 0


async def test_startup_preserves_live_lease(chat: Harness) -> None:
    chat.graph.release.clear()
    async with asyncio.TaskGroup() as group:
        group.create_task(chat.client.post(chat.url, json={"content": "count"}))
        await chat.graph.entered.wait()
        try:
            await ORIGINAL_RECONCILE(chat.app.state.chat)
            assert (await chat.stored())[1]["status"] == "running"
        finally:
            chat.graph.release.set()


async def test_startup_preserves_internal_graph_guard(chat: Harness) -> None:
    admission = await IdempotencyService(chat.database, 10).admit(
        MessageAdmission(user_id=chat.user.id, conversation_id=chat.cid, content="internal graph"),
        deadline=deadline(),
    )
    async with chat.database.session() as session, session.begin():
        assert await lifecycle.try_graph_guard(session, chat.cid)
        await ORIGINAL_RECONCILE(chat.app.state.chat)
        stored = await TurnRepository(session, chat.user.id).get(chat.cid, admission.id)
        assert stored.status == TurnStatus.RUNNING


async def test_conversation_pagination_and_archive(chat: Harness) -> None:
    created = await chat.client.post("/api/v1/conversations", json={"title": "second"})
    assert created.status_code == CREATED
    page = await chat.client.get("/api/v1/conversations?limit=1&offset=0")
    assert page.json()["items"][0]["id"] == created.json()["id"]
    archived = await chat.client.delete(f"/api/v1/conversations/{chat.cid}")
    assert archived.status_code == OK
    assert archived.json()["archived_at"]
    for url in (f"/api/v1/conversations/{chat.cid}", f"/api/v1/conversations/{chat.cid}/turns"):
        assert (await chat.client.get(url)).status_code == NOT_FOUND
    assert (await chat.client.post(chat.url, json={"content": "count"})).status_code == NOT_FOUND


async def test_foreign_and_missing_conversations_hidden(chat: Harness) -> None:
    foreign = chat.user.model_copy(update={"id": uuid4()})

    async def identity() -> UserResponse:
        return foreign

    chat.app.dependency_overrides[get_current_user] = identity
    for cid in (chat.cid, uuid4()):
        base = f"/api/v1/conversations/{cid}"
        assert (await chat.client.get(base)).status_code == NOT_FOUND
        assert (await chat.client.get(base + "/turns")).status_code == NOT_FOUND
        assert (await chat.client.delete(base)).status_code == NOT_FOUND
        assert (
            await chat.client.post(base + "/messages", json={"content": "x"})
        ).status_code == NOT_FOUND
        assert (await chat.client.get(base + f"/turns/{uuid4()}/evidence")).status_code == NOT_FOUND


@pytest.mark.parametrize("content", ["", "x" * 32001])
async def test_message_bounds(chat: Harness, content: str) -> None:
    assert (await chat.client.post(chat.url, json={"content": content})).status_code == INVALID
    assert chat.graph.calls == 0


async def test_every_chat_route_requires_auth(chat: Harness) -> None:
    chat.app.dependency_overrides.clear()
    response = await chat.client.post(chat.url, json={"content": "count"})
    assert response.status_code == 401  # noqa: PLR2004 -- authentication contract.


async def test_user_rate_limits(chat: Harness) -> None:
    chat.app.state.auth_limiter.settings.messages_rules = [RateRule(requests=1, seconds=60)]
    first = await chat.client.post(chat.url, json={"content": "count"})
    assert first.status_code == OK
    second = await chat.client.post(chat.url + "/stream", json={"content": "count"})
    assert second.status_code == LIMITED
    assert "Retry-After" in second.headers


@pytest.mark.parametrize("case", ["refs", "sql", "missing"])
async def test_invalid_evidence_never_becomes_success(chat: Harness, case: str) -> None:
    original = chat.graph.invoke

    async def invalid(ctx: RuntimeContext) -> GraphOutput:
        output = await original(ctx)
        assert output.answer is not None
        if case == "refs":
            output.answer.evidence_refs.data_snapshot_id = uuid4()
        elif case == "sql":
            output.answer.sql = "SELECT fabricated"
        else:
            output.evidence_refs = None
        return output

    chat.graph.invoke = invalid
    response = await chat.client.post(chat.url + "/stream", json={"content": "count"})
    assert [name for name, _ in events(response)] == ["error"]
    row = (await chat.stored())[1]
    assert row["status"] == "failed"
    assert row["answer"] is None


async def test_graph_typed_failure_is_persisted(chat: Harness) -> None:
    chat.app.state.mcp = FakeMcpClient([McpUnavailableError()], schema_responses=[McpUnavailableError()])
    response = await chat.client.post(chat.url + "/stream", json={"content": "count"})
    assert events(response)[-1][1]["code"] == "mcp_unavailable"
    assert (await chat.stored())[1]["failure_reason"] == "mcp_unavailable"


async def test_sync_deadline_finalizes(chat: Harness) -> None:
    middleware = next(
        item for item in chat.app.user_middleware if item.cls.__name__ == "DeadlineMiddleware"
    )
    original_timeout = middleware.kwargs["timeout_s"]
    middleware.kwargs["timeout_s"] = 0.15
    chat.app.middleware_stack = None
    chat.graph.release.clear()
    try:
        response = await chat.client.post(chat.url, json={"content": "count"})
    finally:
        # The evidence read is a separate request, not part of the induced timeout.
        middleware.kwargs["timeout_s"] = original_timeout
        chat.app.middleware_stack = None
    assert response.status_code == 504  # noqa: PLR2004 -- deadline contract.
    assert (await chat.stored())[1]["failure_reason"] == "deadline_exceeded"


async def test_archive_running_turn_conflicts(chat: Harness) -> None:
    chat.graph.release.clear()
    async with asyncio.TaskGroup() as group:
        group.create_task(chat.client.post(chat.url, json={"content": "count"}))
        await chat.graph.entered.wait()
        try:
            response = await chat.client.delete(f"/api/v1/conversations/{chat.cid}")
            assert response.status_code == CONFLICT
        finally:
            chat.graph.release.set()


@pytest.mark.parametrize("length", [1, 32000])
async def test_valid_message_bounds_preserve_text(chat: Harness, length: int) -> None:
    # Keep this test about admission validation, rather than the separate context budget.
    admission = await chat.app.state.chat.idempotency.admit(
        MessageAdmission(
            user_id=chat.user.id,
            conversation_id=chat.cid,
            content="x" * length,
            idempotency_key="bounds",
        ),
        deadline=deadline(),
    )
    response = await chat.client.post(
        chat.url, json={"content": "x" * length}, headers={"Idempotency-Key": "bounds"}
    )
    assert response.status_code == ACCEPTED
    assert response.json()["id"] == str(admission.id)


@pytest.mark.parametrize("key", ["", "x" * 129])
async def test_idempotency_key_bounds(chat: Harness, key: str) -> None:
    response = await chat.client.post(
        chat.url, json={"content": "count"}, headers={"Idempotency-Key": key}
    )
    assert response.status_code == INVALID


async def test_turn_pagination_is_ordered(chat: Harness) -> None:
    await chat.client.post(chat.url, json={"content": "count"})
    response = await chat.client.get(f"/api/v1/conversations/{chat.cid}/turns?limit=1&offset=1")
    assert len(response.json()["items"]) == 1
    assert response.json()["items"][0]["role"] == "assistant"


async def test_startup_lifespan_calls_reconciliation(
    chat: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ChatService, "reconcile", ORIGINAL_RECONCILE)
    admission = await chat.app.state.chat.idempotency.admit(
        MessageAdmission(user_id=chat.user.id, conversation_id=chat.cid, content="interrupted"),
        deadline=deadline(),
    )
    # Keep actual DB and reconciliation, replace only unrelated startup dependencies.
    settings = chat.app.state.settings
    application = create_app(
        settings,
        database=chat.database,
        llm_service=AsyncMock(),
        health_service=AsyncMock(),
        graph_service=AsyncMock(),
    )
    async with application.router.lifespan_context(application):
        row = await application.state.chat.read(
            TurnIdentity(user_id=chat.user.id, conversation_id=chat.cid, turn_id=admission.id)
        )
        assert row.failure_reason == FailureKind.INTERRUPTED
    chat.database.start()


async def test_real_checkpointer_chat_and_replay(
    chat: Harness, migrated: MigrationSettings
) -> None:
    await setup_checkpointer(migrated)
    graph = GraphService(chat.database._settings)
    await graph.start()
    chat.app.state.chat.graph = graph
    try:
        first = await chat.client.post(
            chat.url, json={"content": "count"}, headers={"Idempotency-Key": "durable"}
        )
        assert first.status_code == OK, first.text
        checkpoint = await graph.graph.aget_state(
            {"configurable": {"thread_id": first.json()["id"]}}
        )
        assert checkpoint.values["turn_id"] == UUID(first.json()["id"])
        await graph.aclose()
        # Replay still works with graph and external calls unavailable.
        replay = await chat.client.post(
            chat.url, json={"content": "count"}, headers={"Idempotency-Key": "durable"}
        )
        assert replay.json()["answer"] == first.json()["answer"]
    finally:
        await graph.aclose()


async def test_socket_failure_closes_generator_and_cancels_graph(
    chat: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat.graph.release.clear()
    await asgi_stream(chat, "socket", monkeypatch)
    assert chat.graph.cancelled.is_set()
    assert (await chat.stored())[1]["failure_reason"] == "client_disconnected"


async def test_simultaneous_same_key_requests_replay_original(chat: Harness) -> None:
    chat.graph.release.clear()
    completed: asyncio.Queue[httpx.Response] = asyncio.Queue()

    async def post() -> None:
        response = await chat.client.post(
            chat.url, json={"content": "count"}, headers={"Idempotency-Key": "simultaneous"}
        )
        await completed.put(response)

    async with asyncio.timeout(10), asyncio.TaskGroup() as group:
        for _ in range(5):
            group.create_task(post())
        try:
            replays = [await completed.get() for _ in range(4)]
            assert all(response.status_code == ACCEPTED for response in replays)
            assert len({response.json()["id"] for response in replays}) == 1
        finally:
            chat.graph.release.set()
    original = await completed.get()
    assert original.status_code == OK
    assert original.json()["id"] == replays[0].json()["id"]
    assert chat.graph.calls == 1


async def test_create_conversation_without_body(chat: Harness) -> None:
    response = await chat.client.post("/api/v1/conversations")
    assert response.status_code == CREATED
    assert response.json()["title"] == ""


@pytest.mark.parametrize("streamed", [False, True])
async def test_clarification_is_persisted_and_replayed(chat: Harness, streamed: bool) -> None:
    chat.app.state.llm = FakeChatModel(
        [
            RouteDecision(route=Route.DATA_ONLY, confidence=1, data_intent="查询指标"),
            metric_intent().model_copy(update={"metric_keys": []}),
        ]
    )
    chat.app.state.mcp = FakeMcpClient([])
    suffix = "/stream" if streamed else ""
    headers = {"Idempotency-Key": "clarification"}
    response = await chat.client.post(
        chat.url + suffix, json={"content": "帮我算一下"}, headers=headers
    )
    assert response.status_code == OK, response.text
    body = events(response)[-1][1] if streamed else response.json()
    assert body["status"] == "abstained"
    assert body["failure_reason"] is None
    assert body["answer"]["abstained"]
    assert body["answer"]["claims"] == []
    assert body["clarification"]["kind"] == "metric_not_identified"
    assert body["clarification"]["message"] in body["content"]
    assert body["content"] == body["answer"]["markdown"]
    assert body["evidence_refs"]["data_snapshot_id"] is None
    if streamed:
        assert all(name != "error" for name, _ in events(response))
        assert (
            "".join(frame["delta"] for name, frame in events(response) if name == "token")
            == body["content"]
        )
    stored = (await chat.stored())[-1]
    assert stored["clarification"] == body["clarification"]
    assert stored["content"] == body["content"]
    replay = await chat.client.post(
        chat.url + "/stream", json={"content": "帮我算一下"}, headers=headers
    )
    replayed = events(replay)[-1][1]
    assert replayed["clarification"] == body["clarification"]
    assert replayed["id"] == body["id"]
    assert replayed["replayed"]
    assert chat.graph.calls == 1
    assert not chat.app.state.mcp.calls
    evidence = await chat.client.get(
        f"/api/v1/conversations/{chat.cid}/turns/{body['id']}/evidence"
    )
    assert evidence.status_code == OK, evidence.text
    assert evidence.json()["data"] is None


async def test_clarification_cannot_be_read_by_another_user(chat: Harness) -> None:
    chat.app.state.llm = FakeChatModel(
        [
            RouteDecision(route=Route.DATA_ONLY, confidence=1, data_intent="查询指标"),
            metric_intent().model_copy(update={"metric_keys": []}),
        ]
    )
    response = await chat.client.post(chat.url, json={"content": "算一下"})
    assert response.status_code == OK
    other = chat.user.model_copy(update={"id": uuid4()})

    async def identity() -> UserResponse:
        return other

    chat.app.dependency_overrides[get_current_user] = identity
    response = await chat.client.get(f"/api/v1/conversations/{chat.cid}/turns")
    assert response.status_code == NOT_FOUND


async def test_schema_and_query_consumers_share_injected_mcp(chat: Harness) -> None:
    mcp = chat.app.state.mcp
    assert chat.app.state.schema_catalog._client is mcp
    response = await chat.client.post(chat.url, json={"content": "订单数?"})
    assert response.status_code == OK, response.text
    assert chat.graph.last is not None
    assert chat.graph.last.mcp is mcp
    assert chat.graph.last.schema_catalog is chat.app.state.schema_catalog


async def test_idempotent_replay_does_not_execute_sql_again(chat: Harness) -> None:
    mcp = chat.app.state.mcp
    headers = {"Idempotency-Key": "sql-once"}
    first = await chat.client.post(chat.url, json={"content": "订单数?"}, headers=headers)
    assert first.status_code == OK, first.text
    assert len(mcp.calls) == 1
    replay = await chat.client.post(chat.url, json={"content": "订单数?"}, headers=headers)
    assert replay.status_code == OK, replay.text
    assert replay.json()["answer"] == first.json()["answer"]
    assert len(mcp.calls) == 1
