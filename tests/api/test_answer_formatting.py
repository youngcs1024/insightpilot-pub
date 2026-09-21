"""V3 formatting is committed before HTTP/SSE delivery and survives exact replay."""

from uuid import UUID

import pytest
from sqlalchemy import update

from app.agents.contracts import Route
from app.agents.runtime import RuntimeContext
from app.agents.state import GraphOutput
from app.db.models import Turn
from tests.agents.parent_support import parent_context
from tests.api.chat_support import OK, Harness, chat, events

pytestmark = pytest.mark.integration
__all__ = ["chat"]


@pytest.mark.parametrize("route", list(Route))
@pytest.mark.parametrize("streamed", [False, True])
async def test_v3_all_route_format_commit_and_replay(chat: Harness, route: Route, streamed: bool) -> None:
    ctx = parent_context(route, settings=chat.app.state.settings)
    chat.app.state.llm = ctx.llm
    chat.app.state.retrieval = ctx.retrieval
    chat.app.state.knowledge_generation = ctx.knowledge_generation
    question = {"content": "请分析2026年8月的经营情况，请用表格回答，保留两位小数。"}
    headers = {"Idempotency-Key": "v3-formatting"}
    response = await chat.client.post(chat.url + ("/stream" if streamed else ""), json=question, headers=headers)
    assert response.status_code == OK, response.text
    body = events(response)[-1][1] if streamed else response.json()
    answer = body["answer"]
    assert answer["schema_version"] == 3  # noqa: PLR2004 -- released wire version.
    assert answer["trace_id"] == body["trace_id"]
    assert answer["format_preference"] == {"prefer": "table", "decimals": 2}
    assert "| --- |" in answer["markdown"]
    assert answer == (await chat.stored())[-1]["answer"]
    if route is Route.CLARIFY:
        assert answer["abstained"] and not answer["claims"]
        assert body["clarification"]["message"] in answer["markdown"]
    elif route in {Route.DATA_ONLY, Route.BOTH}:
        assert "**统计口径**" in answer["markdown"]
        assert answer["sql"] in answer["markdown"]
    if streamed:
        assert "".join(value["delta"] for name, value in events(response) if name == "token") == answer["markdown"]
    before = len(ctx.llm.calls), len(chat.app.state.mcp.calls), len(ctx.retrieval.calls)
    replay = await chat.client.post(chat.url, json=question, headers=headers)
    assert replay.json()["answer"] == answer
    assert replay.json()["replayed"]
    assert (len(ctx.llm.calls), len(chat.app.state.mcp.calls), len(ctx.retrieval.calls)) == before


@pytest.mark.parametrize("field", ["claims", "trace_id", "confidence", "markdown"])
async def test_data_commit_rejects_envelope_tampering(chat: Harness, monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    original = chat.graph.invoke

    async def tampered(ctx: RuntimeContext, **kwargs: object) -> GraphOutput:
        output = await original(ctx, **kwargs)
        value = {"claims": [], "trace_id": "wrong-turn", "confidence": 0, "markdown": "forged"}[field]
        output.answer = output.answer.model_copy(update={field: value})
        return output

    monkeypatch.setattr(chat.graph, "invoke", tampered)
    response = await chat.client.post(chat.url, json={"content": "count"})
    assert response.status_code == 409  # noqa: PLR2004 -- consistency conflict.
    assert (await chat.stored())[-1]["answer"] is None


@pytest.mark.parametrize("version", [1, 2])
async def test_legacy_wire_fields_replay_without_reformatting(chat: Harness, version: int) -> None:
    question = {"content": "count"}
    headers = {"Idempotency-Key": "old-format"}
    body = (await chat.client.post(chat.url, json=question, headers=headers)).json()
    legacy = dict(body["answer"])
    legacy["schema_version"] = version
    for field in ("claims", "trace_id", "format_preference", "attempted_sources", "unanswered"):
        legacy.pop(field)
    async with chat.database.session() as session, session.begin():
        await session.execute(update(Turn).where(Turn.id == UUID(body["id"])).values(answer=legacy))
    before = len(chat.app.state.llm.calls)
    replay = (await chat.client.post(chat.url, json=question, headers=headers)).json()
    assert replay["answer"]["schema_version"] == version
    assert replay["answer"]["markdown"] == body["answer"]["markdown"]
    assert replay["answer"]["claims"] == []
    assert replay["answer"]["trace_id"] is None
    assert len(chat.app.state.llm.calls) == before
