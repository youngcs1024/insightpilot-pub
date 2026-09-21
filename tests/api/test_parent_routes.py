"""Real HTTP/SSE persistence and replay for the newly connected source routes."""

import pytest

from app.agents.contracts import Route
from app.core.errors import McpUnavailableError, RetrievalUnavailableError
from app.schemas.schema_catalog import BusinessSchemaResponse
from app.services.schema_catalog import SchemaCatalogService
from tests.agents.parent_support import parent_context
from tests.api.chat_support import OK, Harness, chat, events
from tests.factories import business_schema

pytestmark = pytest.mark.integration
__all__ = ["chat"]


@pytest.mark.parametrize("route", [Route.KNOWLEDGE_ONLY, Route.BOTH, Route.CLARIFY])
@pytest.mark.parametrize("streaming", [False, True])
async def test_parent_route_response_replay_and_historical_evidence(
    chat: Harness,
    route: Route,
    streaming: bool,
) -> None:
    ctx = parent_context(route, settings=chat.app.state.settings)
    chat.app.state.llm = ctx.llm
    chat.app.state.retrieval = ctx.retrieval
    chat.app.state.knowledge_generation = ctx.knowledge_generation
    suffix = "/stream" if streaming else ""
    content = {"content": "请分析2026年8月的经营情况"}
    headers = {"Idempotency-Key": "four-route"}
    response = await chat.client.post(chat.url + suffix, json=content, headers=headers)
    assert response.status_code == OK
    body = events(response)[-1][1] if streaming else response.json()
    assert body["status"] == ("abstained" if route is Route.CLARIFY else "succeeded")
    stored = (await chat.stored())[-1]
    assert body["answer"] == stored["answer"]
    assert body["content"] == stored["content"]
    evidence = await chat.client.get(
        f"/api/v1/conversations/{chat.cid}/turns/{body['id']}/evidence"
    )
    assert evidence.status_code == OK
    snapshots = evidence.json()
    assert (snapshots["data"] is not None) == (route is Route.BOTH)
    assert (snapshots["knowledge"] is not None) == (route is not Route.CLARIFY)
    before = len(ctx.llm.calls), len(ctx.retrieval.calls), len(chat.app.state.mcp.calls)
    replay = await chat.client.post(chat.url, json=content, headers=headers)
    assert replay.json()["answer"] == body["answer"]
    assert replay.json()["replayed"]
    assert (len(ctx.llm.calls), len(ctx.retrieval.calls), len(chat.app.state.mcp.calls)) == before


@pytest.mark.parametrize("empty", [False, True])
async def test_knowledge_failure_or_absence_uses_correct_terminal_status(
    chat: Harness,
    empty: bool,
) -> None:
    ctx = parent_context(
        Route.KNOWLEDGE_ONLY,
        empty=empty,
        settings=chat.app.state.settings,
        knowledge_error=None if empty else RetrievalUnavailableError(),
    )
    chat.app.state.llm = ctx.llm
    chat.app.state.retrieval = ctx.retrieval
    chat.app.state.knowledge_generation = ctx.knowledge_generation
    response = await chat.client.post(chat.url, json={"content": "请分析2026年8月的经营情况"})
    assert response.status_code == (OK if empty else 500)
    assert (await chat.stored())[-1]["status"] == ("abstained" if empty else "failed")
    assert chat.app.state.mcp.calls == []


@pytest.mark.parametrize("missing", ["data", "knowledge"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_partial_both_answer_is_durable_and_replayable(
    chat: Harness,
    missing: str,
    streaming: bool,
) -> None:
    ctx = parent_context(
        Route.BOTH,
        data_error=McpUnavailableError() if missing == "data" else None,
        knowledge_error=RetrievalUnavailableError() if missing == "knowledge" else None,
        settings=chat.app.state.settings,
    )
    ctx.mcp.enqueue_schema(
        business_schema(),
        *[BusinessSchemaResponse(revision="business-v1", unchanged=True) for _ in range(4)],
    )
    chat.app.state.mcp = ctx.mcp
    chat.app.state.schema_catalog = SchemaCatalogService(
        chat.database, ctx.mcp, chat.app.state.settings.schema_catalog
    )
    chat.app.state.llm = ctx.llm
    chat.app.state.retrieval = ctx.retrieval
    chat.app.state.knowledge_generation = ctx.knowledge_generation
    content = {"content": "请分析2026年8月的经营情况"}
    headers = {"Idempotency-Key": "partial-both"}
    response = await chat.client.post(
        chat.url + ("/stream" if streaming else ""), json=content, headers=headers
    )
    assert response.status_code == OK
    body = events(response)[-1][1] if streaming else response.json()
    assert body["status"] == "degraded"
    assert missing in body["answer"]["degraded_components"]
    assert body["answer"] == (await chat.stored())[-1]["answer"]
    assert not body["answer"]["abstained"]
    evidence = await chat.client.get(
        f"/api/v1/conversations/{chat.cid}/turns/{body['id']}/evidence"
    )
    assert evidence.status_code == OK
    assert (evidence.json()["data"] is None) == (missing == "data")
    assert (evidence.json()["knowledge"] is None) == (missing == "knowledge")
    before = len(ctx.llm.calls), len(ctx.mcp.calls), len(ctx.retrieval.calls)
    replay = await chat.client.post(chat.url, json=content, headers=headers)
    assert replay.json()["replayed"]
    assert replay.json()["status"] == "degraded"
    assert replay.json()["answer"] == body["answer"]
    assert (len(ctx.llm.calls), len(ctx.mcp.calls), len(ctx.retrieval.calls)) == before


@pytest.mark.parametrize("streaming", [False, True])
async def test_both_failures_name_sources_without_emitting_an_answer(
    chat: Harness, streaming: bool
) -> None:
    ctx = parent_context(
        Route.BOTH,
        data_error=McpUnavailableError(),
        knowledge_error=RetrievalUnavailableError(),
        settings=chat.app.state.settings,
    )
    ctx.mcp.enqueue_schema(
        business_schema(),
        *[BusinessSchemaResponse(revision="business-v1", unchanged=True) for _ in range(4)],
    )
    chat.app.state.mcp = ctx.mcp
    chat.app.state.schema_catalog = SchemaCatalogService(
        chat.database, ctx.mcp, chat.app.state.settings.schema_catalog
    )
    chat.app.state.llm = ctx.llm
    chat.app.state.retrieval = ctx.retrieval
    chat.app.state.knowledge_generation = ctx.knowledge_generation
    response = await chat.client.post(
        chat.url + ("/stream" if streaming else ""),
        json={"content": "请分析2026年8月的经营情况"},
    )
    assert response.status_code == (OK if streaming else 500)
    body = events(response)[-1][1] if streaming else response.json()
    assert body["message"] == "业务数据与知识来源均未能提供可用证据，本次分析失败。"
    stored = (await chat.stored())[-1]
    assert stored["status"] == "failed"
    assert stored["answer"] is None
    assert all(call.schema_name not in {"AnswerDraft", "KnowledgeDraft"} for call in ctx.llm.calls)
    if streaming:
        assert not any(name == "token" for name, _ in events(response))
