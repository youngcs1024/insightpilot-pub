"""Persisted clarification policy across HTTP, SSE, history and new user turns."""

# ruff: noqa: PLR2004 -- wire versions and exact turn/call counts are acceptance contracts.

import json
from uuid import UUID, uuid4

import pytest
from sqlalchemy import update

from app.agents.contracts import Route, RouteDecision, TurnIdentity
from app.db.models import Turn, TurnStatus
from app.core.errors import NotFoundError
from app.schemas.clarification import ClarificationCategory, ClarificationIntent, MissingDimension
from app.schemas.metric_resolution import MetricIntent
from app.services.conversations import ConversationService
from tests.agents.support import metric_intent, sql_candidate
from tests.answer_support import data_draft
from tests.api.chat_support import OK, Harness, chat, events
from tests.fakes.chat_model import FakeChatModel

pytestmark = pytest.mark.integration
__all__ = ["chat"]


def scope_decision() -> RouteDecision:
    return RouteDecision(
        route=Route.CLARIFY, confidence=1,
        clarification_intent=ClarificationIntent(
            category=ClarificationCategory.AMBIGUOUS_SCOPE,
            missing_dimensions=[MissingDimension.PERIOD], metric_keys=["gmv"],
            subject="查询GMV",
        ),
    )


@pytest.mark.parametrize("streamed", [False, True])
async def test_mixed_clarifications_third_and_fourth_turn_stop_asking(
    chat: Harness, streamed: bool
) -> None:
    model = FakeChatModel([
        RouteDecision(route=Route.DATA_ONLY, confidence=1, data_intent="查询GMV"),
        MetricIntent(metric_keys=["gmv"], period_expression=""),
        RouteDecision(route=Route.KNOWLEDGE_ONLY, confidence=1, knowledge_intent="2026年13月退货政策"),
        RouteDecision(route=Route.CLARIFY, confidence=1),
    ])
    chat.app.state.llm = model
    questions = ["那个", "查一下GMV", "2026年13月退货政策", "那个问题"]
    bodies = []
    for index, question in enumerate(questions):
        response = await chat.client.post(
            chat.url + ("/stream" if streamed else ""), json={"content": question},
            headers={"Idempotency-Key": f"clarify-{index}"},
        )
        assert response.status_code == OK, response.text
        body = events(response)[-1][1] if streamed else response.json()
        assert body["status"] == "abstained", body
        assert body["clarification"]["loop_prevented"] is (index >= 2)
        assert body["answer"]["claims"] == []
        assert body["clarification"] == (await chat.stored())[-1]["clarification"]
        bodies.append(body)
    assert bodies[1]["clarification"]["kind"] == "period_unresolved"
    assert bodies[2]["clarification"]["category"] == "ambiguous_scope"
    assert chat.app.state.mcp.calls == []
    before = len(model.calls), chat.graph.calls, len(await chat.stored())
    replay = await chat.client.post(
        chat.url, json={"content": questions[-1]}, headers={"Idempotency-Key": "clarify-3"}
    )
    assert replay.json()["answer"] == bodies[-1]["answer"]
    assert replay.json()["clarification"] == bodies[-1]["clarification"]
    assert before == (len(model.calls), chat.graph.calls, len(await chat.stored()))


async def test_accepting_suggestion_is_a_fresh_analysis_with_owned_history(chat: Harness) -> None:
    model = FakeChatModel([scope_decision()])
    chat.app.state.llm = model
    first = (await chat.client.post(chat.url, json={"content": "查GMV"})).json()
    assert first["status"] == "abstained"
    suggestion = first["clarification"]["suggested_question"]
    assert not chat.app.state.mcp.calls
    model.enqueue(
        RouteDecision(route=Route.DATA_ONLY, confidence=1, data_intent="查询2026年8月GMV"),
        metric_intent(), sql_candidate(), data_draft(markdown="GMV 为42。"),
    )
    second = (await chat.client.post(chat.url, json={"content": "就按你建议的"})).json()
    assert second["status"] == "succeeded", second
    assert second["id"] != first["id"]
    routing = json.loads(model.calls[1].messages[1].content)
    assert any(suggestion in item["content"] for item in routing["routing_context"]["recent_messages"])
    assert len(chat.app.state.mcp.calls) == 1


@pytest.mark.parametrize("prior_status", [TurnStatus.SUCCEEDED, TurnStatus.FAILED, TurnStatus.ABSTAINED])
async def test_nonclarification_outcome_breaks_sequence(chat: Harness, prior_status: TurnStatus) -> None:
    chat.app.state.llm = FakeChatModel([scope_decision(), scope_decision(), scope_decision()])
    await chat.client.post(chat.url, json={"content": "查GMV"})
    second = (await chat.client.post(chat.url, json={"content": "查GMV"})).json()
    async with chat.database.session() as session, session.begin():
        await session.execute(update(Turn).where(Turn.id == UUID(second["id"])).values(
            status=prior_status, clarification=None
        ))
    third = (await chat.client.post(chat.url, json={"content": "查GMV"})).json()
    assert not third["clarification"]["loop_prevented"]


async def test_counter_survives_trimming_and_new_service_instance(chat: Harness) -> None:
    chat.app.state.llm = FakeChatModel([scope_decision(), scope_decision(), scope_decision()])
    bodies = [
        (await chat.client.post(chat.url, json={"content": "查GMV"})).json()
        for _ in range(2)
    ]
    async with chat.database.session() as session, session.begin():
        for body in bodies:
            await session.execute(update(Turn).where(Turn.id == UUID(body["id"])).values(content="长" * 5000))
    chat.app.state.conversations = ConversationService(chat.database)
    third = (await chat.client.post(chat.url, json={"content": "查GMV"})).json()
    assert third["clarification"]["loop_prevented"]
    created = await chat.client.post("/api/v1/conversations", json={})
    other = created.json()["id"]
    chat.app.state.llm = FakeChatModel([scope_decision()])
    fresh = (await chat.client.post(
        f"/api/v1/conversations/{other}/messages", json={"content": "查GMV"}
    )).json()
    assert not fresh["clarification"]["loop_prevented"]
    assert fresh["clarification"]["recent_topics"] == []


async def test_legacy_clarification_replays_without_upgrade(chat: Harness) -> None:
    chat.app.state.llm = FakeChatModel([])
    headers = {"Idempotency-Key": "legacy-clarification"}
    body = (await chat.client.post(chat.url, json={"content": "那个"}, headers=headers)).json()
    legacy = {key: value for key, value in body["clarification"].items() if key in {
        "schema_version", "kind", "message", "metric_key", "available_metrics", "supported_grains"
    }}
    legacy["schema_version"] = 1
    async with chat.database.session() as session, session.begin():
        await session.execute(update(Turn).where(Turn.id == UUID(body["id"])).values(clarification=legacy))
    replay = (await chat.client.post(chat.url, json={"content": "那个"}, headers=headers)).json()
    assert replay["clarification"] == legacy
    assert replay["answer"] == body["answer"]
    assert not chat.app.state.llm.calls


async def test_clarification_history_repository_rejects_foreign_identity(chat: Harness) -> None:
    chat.app.state.llm = FakeChatModel([])
    await chat.client.post(chat.url, json={"content": "那个"})
    identity = TurnIdentity(
        user_id=uuid4(), conversation_id=chat.cid, turn_id=chat.graph.last.identity.turn_id
    )
    with pytest.raises(NotFoundError):
        await ConversationService(chat.database).prepare(identity)


async def test_sse_clarification_tokens_follow_database_commit(
    chat: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pydantic import BaseModel

    from app.agents.state import GraphOutput
    from app.services import chat_stream

    committed = False
    original_commit = chat.app.state.chat._clarify
    original_frame = chat_stream.frame

    async def commit(identity: TurnIdentity, output: GraphOutput, latency_ms: int) -> None:
        nonlocal committed
        await original_commit(identity, output, latency_ms)
        persisted = await chat.app.state.chat.read(identity)
        assert persisted.status is TurnStatus.ABSTAINED
        assert persisted.clarification == output.clarification
        committed = True

    def frame(event: str, payload: BaseModel) -> str:
        if event in {"token", "done"}:
            assert committed
        return original_frame(event, payload)

    monkeypatch.setattr(chat.app.state.chat, "_clarify", commit)
    monkeypatch.setattr(chat_stream, "frame", frame)
    chat.app.state.llm = FakeChatModel([])
    response = await chat.client.post(chat.url + "/stream", json={"content": "那个"})
    assert response.status_code == OK
    assert events(response)[-1][1]["status"] == "abstained"
    assert committed
