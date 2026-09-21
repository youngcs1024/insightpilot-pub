"""Every Step 4.10 matrix cell uses the actual parent and specialist contracts."""

import asyncio
import json
import time
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from starlette.requests import Request

from app.agents.contracts import Route
from app.agents.degradation import deadline_synthesis
from app.agents.failures import FailureKind, NodeFailure
from app.agents.synthesis_answer import synthesis_answer, validate_synthesis_answer
from app.api.exception_handlers import handle_known
from app.core.config_models import HTTPSettings
from app.core.deadline import Deadline, ResponseBudget
from app.core.errors import (
    ConflictError,
    DeadlineExceededError,
    McpUnavailableError,
    RetrievalUnavailableError,
)
from app.schemas.model_runtime import ModelFailureKind
from app.services.chat_stream import error_event
from app.services.deadline_finalization import finalize_deadline
from app.services.turn_results import BothSourcesFailedError, TurnFailedError
from model_runtime.errors import ModelError
from tests.agents.parent_support import parent_context
from tests.agents.support import invoke
from tests.agents.synthesis_support import synthesis_context

ROUTES = [Route.DATA_ONLY, Route.KNOWLEDGE_ONLY, Route.BOTH]


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("failure", ["mcp", "empty", "retrieval", "encode", "both", "rerank"])
async def test_degradation_matrix(route: Route, failure: str) -> None:
    data_down = failure in {"mcp", "both"}
    knowledge_down = failure in {"retrieval", "encode", "both"}
    ctx = parent_context(
        route,
        data_error=McpUnavailableError("private secret") if data_down else None,
        knowledge_error=(ModelError() if failure == "encode" else RetrievalUnavailableError())
        if knowledge_down
        else None,
        empty=failure == "empty",
    )
    if failure == "rerank":
        value = ctx.retrieval.responses[0]
        value.retrieval_config.use_rerank = True
        value.degradation = ModelFailureKind.UNAVAILABLE
        value.reranked = False
        value.top_rerank_score = None
        value.meets_floor = None
        for candidate in value.candidates:
            candidate.scores.rerank = None
    output = await invoke(ctx)
    if route is Route.DATA_ONLY:
        expected = "failed" if data_down else "succeeded"
        assert not ctx.retrieval.calls
    elif route is Route.KNOWLEDGE_ONLY:
        expected = (
            "failed"
            if knowledge_down
            else (
                "abstained"
                if failure == "empty"
                else ("degraded" if failure == "rerank" else "succeeded")
            )
        )
        assert not ctx.mcp.calls
    else:
        expected = "failed" if failure == "both" else "degraded"
    assert output.status == expected
    assert "private secret" not in output.model_dump_json()
    if output.answer is not None and expected == "degraded":
        assert "⚠️" in output.answer.markdown
        assert output.answer.evidence_refs == (await ctx.evidence.read_bundle(ctx.identity)).refs
        if failure == "rerank":
            assert output.answer.degraded_components == ["rerank"]
            assert output.answer.evidence_refs.knowledge_snapshot_id
            if route is Route.BOTH:
                assert output.answer.evidence_refs.data_snapshot_id
        elif route is Route.BOTH:
            assert ("data" if data_down else "knowledge") in output.answer.degraded_components


async def test_mcp_down_never_connects_directly(monkeypatch: pytest.MonkeyPatch) -> None:
    connect = AsyncMock(side_effect=AssertionError("business database fallback"))
    monkeypatch.setattr("app.db.session.create_async_engine", connect)
    monkeypatch.setattr("psycopg.AsyncConnection.connect", connect)
    output = await invoke(parent_context(Route.BOTH, data_error=McpUnavailableError()))
    assert output.status == "degraded"
    assert "数据源当前不可用" in output.answer.markdown
    assert "企业知识库" in output.answer.markdown
    connect.assert_not_called()


def test_both_unavailable_names_both_reasons() -> None:
    failures = [
        NodeFailure(node="data", kind=FailureKind.MCP_UNAVAILABLE, detail="secret", retryable=True),
        NodeFailure(
            node="knowledge",
            kind=FailureKind.RETRIEVAL_UNAVAILABLE,
            detail="secret",
            retryable=True,
        ),
    ]
    error = BothSourcesFailedError(FailureKind.RETRIEVAL_UNAVAILABLE, failures)
    assert "数据源当前不可用" in error.public_message
    assert "企业知识库当前不可用" in error.public_message
    assert "secret" not in error.public_message
    assert "数据源当前不可用" in TurnFailedError(FailureKind.MCP_UNAVAILABLE).public_message


async def test_source_failure_http_and_sse_share_safe_message() -> None:
    error = TurnFailedError(FailureKind.MCP_UNAVAILABLE)
    error.detail = "private connection password"
    response = await handle_known(Request({"type": "http"}), error)
    body = json.loads(response.body)
    assert "数据源当前不可用" in body["message"]
    assert error_event(error).message == body["message"]
    assert "password" not in response.body.decode()


@pytest.mark.parametrize("source", ["data", "knowledge", "both"])
async def test_deadline_partial_uses_completed_specialist(source: str) -> None:
    ctx, state, bundle = await synthesis_context(
        data=source != "knowledge", knowledge=source != "data"
    )
    ctx = replace(ctx, deadline=Deadline(time.monotonic() - 0.01))
    ctx.llm.generate_structured = AsyncMock(side_effect=AssertionError("post-deadline model call"))
    output = await finalize_deadline(state, ctx)
    assert output.status == "degraded"
    assert output.answer.synthesis.attempts == 0
    assert "deadline" in output.answer.degraded_components
    assert output.evidence_refs == bundle.refs
    assert output.answer.claims
    validate_synthesis_answer(output.answer, bundle)
    ctx.llm.generate_structured.assert_not_awaited()
    assert ctx.mcp.calls == []


@pytest.mark.parametrize("route", [Route.DATA_ONLY, Route.KNOWLEDGE_ONLY])
async def test_deadline_single_source_fails(route: Route) -> None:
    ctx, state, _ = await synthesis_context()
    state.route = state.route.model_copy(update={"route": route})
    with pytest.raises(DeadlineExceededError):
        await finalize_deadline(state, ctx)


async def test_deadline_empty_or_failed_persistence_never_publishes() -> None:
    ctx, state, _ = await synthesis_context(data=False, knowledge=False)
    with pytest.raises(DeadlineExceededError):
        await finalize_deadline(state, ctx)
    ctx, state, _ = await synthesis_context()
    ctx.evidence.commit_bundle = AsyncMock(side_effect=ConflictError())
    with pytest.raises(ConflictError):
        await finalize_deadline(state, ctx)


async def test_expired_grace_cannot_restart_budget() -> None:
    ctx, state, _ = await synthesis_context()
    ctx = replace(ctx, deadline=Deadline(time.monotonic() - 10))
    with pytest.raises(DeadlineExceededError):
        await finalize_deadline(state, ctx)
    assert ctx.finalization_deadline.at == ctx.deadline.at + 5


@pytest.mark.parametrize("value", [-1, 5.1, float("inf"), float("nan")])
def test_finalization_grace_bounded(value: float) -> None:
    with pytest.raises(ValidationError):
        HTTPSettings(finalization_grace_s=value)


async def test_response_grace_does_not_renew_analysis_deadline() -> None:
    analysis = Deadline(time.monotonic() + 1)
    async with asyncio.timeout_at(analysis.at) as timer:
        response = ResponseBudget(timer, Deadline(analysis.at + 5))
        response.allow_finalization()
        assert timer.when() == analysis.at + 5
        response.allow_finalization()
        assert timer.when() == analysis.at + 5
        assert analysis.remaining() <= 1


async def test_deterministic_answer_validates_exact_references() -> None:
    ctx, _, bundle = await synthesis_context()
    result = deadline_synthesis(bundle, [])
    answer = synthesis_answer(result, bundle, ["deadline"], trace_id=ctx.trace_id)
    assert "42" in answer.markdown
    assert answer.citations
    changed = answer.model_copy(update={"sql": "SELECT invented"})
    with pytest.raises(ConflictError):
        validate_synthesis_answer(changed, bundle)


async def test_degraded_status_distinct_from_failed_and_abstained() -> None:
    cases = [
        parent_context(Route.DATA_ONLY),
        parent_context(Route.BOTH, data_error=McpUnavailableError()),
        parent_context(Route.KNOWLEDGE_ONLY, empty=True),
        parent_context(Route.DATA_ONLY, data_error=McpUnavailableError()),
    ]
    assert [(await invoke(ctx)).status for ctx in cases] == [
        "succeeded",
        "degraded",
        "abstained",
        "failed",
    ]
