"""Real HTTP/SSE persistence and replay for the newly connected source routes."""

import pytest

from app.agents.contracts import Route
from app.core.errors import RetrievalUnavailableError
from tests.agents.parent_support import parent_context
from tests.api.chat_support import OK, Harness, chat, events

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
