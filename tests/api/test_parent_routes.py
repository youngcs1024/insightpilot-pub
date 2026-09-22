"""Real HTTP/SSE persistence and replay for the newly connected source routes."""

from uuid import UUID

import pytest
from sqlalchemy import update

from app.agents.contracts import Route
from app.agents.runtime import RuntimeContext
from app.agents.state import GraphOutput
from app.core.errors import McpUnavailableError, RetrievalUnavailableError
from app.db.models import Turn
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
        *[business_schema() for _ in range(4)],
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
        *[business_schema() for _ in range(4)],
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
    assert body["message"].startswith("业务数据与知识来源均未能提供可用证据，本次分析失败。")
    assert "数据源当前不可用" in body["message"]
    assert "企业知识库当前不可用" in body["message"]
    stored = (await chat.stored())[-1]
    assert stored["status"] == "failed"
    assert stored["answer"] is None
    assert all(
        call.schema_name not in {"DataAnswerDraft", "KnowledgeDraft", "SynthesisOutput"}
        for call in ctx.llm.calls
    )
    if streaming:
        assert not any(name == "token" for name, _ in events(response))


@pytest.mark.parametrize("streaming", [False, True])
async def test_synthesis_claims_are_committed_before_delivery(
    chat: Harness, streaming: bool
) -> None:
    ctx = parent_context(Route.BOTH, settings=chat.app.state.settings)
    chat.app.state.llm = ctx.llm
    chat.app.state.retrieval = ctx.retrieval
    response = await chat.client.post(
        chat.url + ("/stream" if streaming else ""),
        json={"content": "请分析2026年8月的经营情况"},
    )
    assert response.status_code == OK
    body = events(response)[-1][1] if streaming else response.json()
    answer = body["answer"]
    assert answer["schema_version"] == 3  # noqa: PLR2004 -- released Answer v3.
    assert answer["synthesis"]["evidence_refs"] == body["evidence_refs"]
    assert {claim["kind"] for claim in answer["synthesis"]["claims"]} == {
        "fact_data",
        "fact_document",
    }
    assert answer == (await chat.stored())[-1]["answer"]
    assert [call.schema_name for call in ctx.llm.calls][-1] == "SynthesisOutput"
    assert not any(
        call.schema_name in {"DataAnswerDraft", "KnowledgeDraft"} for call in ctx.llm.calls
    )
    if streaming:
        frames = events(response)
        assert (
            "".join(str(payload["delta"]) for name, payload in frames if name == "token")
            == answer["markdown"]
        )


@pytest.mark.parametrize("field", ["markdown", "synthesis"])
async def test_chat_commit_rejects_tampered_synthesis(
    chat: Harness, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    ctx = parent_context(Route.BOTH, settings=chat.app.state.settings)
    chat.app.state.llm = ctx.llm
    chat.app.state.retrieval = ctx.retrieval
    original = chat.graph.invoke

    async def tampered(runtime: RuntimeContext, **kwargs: object) -> GraphOutput:
        output = await original(runtime, **kwargs)
        output.answer = output.answer.model_copy(
            update={field: "未经核验的断言" if field == "markdown" else None}
        )
        return output

    monkeypatch.setattr(chat.graph, "invoke", tampered)
    response = await chat.client.post(chat.url, json={"content": "请分析2026年8月的经营情况"})
    assert response.status_code == 409  # noqa: PLR2004 -- typed internal consistency conflict.
    stored = (await chat.stored())[-1]
    assert stored["status"] == "failed"
    assert stored["answer"] is None
    assert "未经核验" not in response.text


async def test_legacy_answer_json_replays_without_a_model_call(chat: Harness) -> None:

    response = await chat.client.post(
        chat.url, json={"content": "count"}, headers={"Idempotency-Key": "legacy"}
    )
    assert response.status_code == OK
    body = response.json()
    legacy = dict(body["answer"])
    legacy["schema_version"] = 1
    legacy.pop("synthesis")
    async with chat.database.session() as session, session.begin():
        await session.execute(update(Turn).where(Turn.id == UUID(body["id"])).values(answer=legacy))
    calls = len(chat.app.state.llm.calls)
    replay = await chat.client.post(
        chat.url, json={"content": "count"}, headers={"Idempotency-Key": "legacy"}
    )
    assert replay.json()["replayed"]
    assert replay.json()["answer"]["schema_version"] == 1
    assert replay.json()["answer"]["synthesis"] is None
    assert replay.json()["answer"]["markdown"] == body["answer"]["markdown"]
    assert len(chat.app.state.llm.calls) == calls
