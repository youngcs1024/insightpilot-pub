"""Useful terminal clarification without an extra model or analytical call."""

# ruff: noqa: PLR2004 -- exact policy thresholds and wire versions are acceptance contracts.

from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from langgraph.runtime import Runtime
from pydantic import ValidationError

from app.agents.clarification_policy import render_clarification
from app.agents.contracts import PreparedContext, Route, RouteDecision
from app.agents.nodes.clarify import clarify
from app.agents.state import AgentState
from app.core.errors import ConflictError, DeadlineExceededError
from app.schemas.clarification import (
    ClarificationCapabilities,
    ClarificationCategory,
    ClarificationHistory,
    ClarificationIntent,
    MissingDimension,
)
from app.schemas.knowledge_query import KnowledgeClarification, KnowledgeClarificationKind
from app.schemas.metric_resolution import ClarificationKind, MetricClarification
from tests.agents.parent_support import parent_context
from tests.agents.support import context, invoke


def clarification_state(
    category: ClarificationCategory = ClarificationCategory.AMBIGUOUS_SCOPE,
    *, question: str = "华东GMV（排除测试账号）", dimensions: list[MissingDimension] | None = None,
    topics: list[str] | None = None, consecutive: int = 0,
) -> AgentState:
    return AgentState(
        **context().identity.model_dump(), question=question,
        route=RouteDecision(
            route=Route.CLARIFY, confidence=1,
            clarification_intent=ClarificationIntent(
                category=category, missing_dimensions=dimensions or [],
                metric_keys=["gmv"], subject=question,
            ),
        ),
        prepared=PreparedContext(
            question=question, summary="", messages=[], prior_sql=[],
            clarification_history=ClarificationHistory(
                consecutive=consecutive, recent_topics=topics or []
            ),
        ),
    )


async def test_ambiguous_reference_lists_recent_topics() -> None:
    ctx = context(responses=[])
    state = clarification_state(
        ClarificationCategory.AMBIGUOUS_REFERENCE, question="那个问题",
        topics=["2026年8月GMV", "退货政策"],
    )
    result = await clarify(state, Runtime(context=ctx))
    value = result.update["route_clarification"]
    assert "2026年8月GMV" in value.message and "退货政策" in value.message
    assert "具体指哪个问题" in value.message
    assert value.suggested_question == "2026年8月GMV"
    assert ctx.llm.calls == ctx.mcp.calls == []


async def test_missing_period_offers_default() -> None:
    ctx = context(responses=[])
    result = await clarify(
        clarification_state(dimensions=[MissingDimension.PERIOD]), Runtime(context=ctx)
    )
    value = result.update["route_clarification"]
    assert value.kind is ClarificationKind.PERIOD_UNRESOLVED
    assert "2026-08-01 00:00 至 2026-09-01 00:00" in value.message
    assert "Asia/Shanghai，左闭右开" in value.message
    assert "华东GMV（排除测试账号）" in value.suggested_question
    assert "尚未执行" in value.message
    assert not ctx.llm.calls and not ctx.mcp.calls


async def test_out_of_scope_lists_capabilities() -> None:
    ctx = context(responses=[])
    result = await clarify(
        clarification_state(ClarificationCategory.OUT_OF_SCOPE, question="请替我退款"),
        Runtime(context=ctx),
    )
    value = result.update["route_clarification"]
    assert value.kind is ClarificationKind.OUT_OF_SCOPE
    assert "只读分析范围" in value.message
    assert "GMV" in value.message and "操作流程" in value.message and "促销活动规则" in value.message
    assert "请替我退款" not in value.suggested_question
    assert not ctx.mcp.calls and not ctx.llm.calls


async def test_clarification_recorded_as_abstained_not_failed() -> None:
    ctx = parent_context(Route.CLARIFY)
    result = await invoke(ctx)
    assert result.status == "abstained" and result.answer.abstained
    assert result.clarification.schema_version == 2
    assert not result.failures
    assert not result.answer.claims and not result.answer.citations and not result.answer.sql
    assert result.answer.confidence == 0
    assert ctx.mcp.calls == ctx.retrieval.calls == []
    assert len(ctx.llm.calls) == 1


async def test_three_consecutive_clarifications_offers_interpretation() -> None:
    ctx = context(responses=[])
    state = clarification_state(dimensions=[MissingDimension.PERIOD], consecutive=2)
    result = await clarify(state, Runtime(context=ctx))
    value = result.update["route_clarification"]
    assert value.loop_prevented
    assert "替代解释" in value.message and "2026年8月" in value.suggested_question
    assert "？" not in value.message
    assert "尚未执行" in value.message and not ctx.mcp.calls


@pytest.mark.parametrize(("instant", "expected"), [
    (datetime(2026, 1, 1, tzinfo=UTC), "2025-12-01 00:00 至 2026-01-01 00:00"),
    (datetime(2026, 8, 31, 16, tzinfo=UTC), "2026-08-01 00:00 至 2026-09-01 00:00"),
    (datetime(2026, 8, 31, 15, 59, tzinfo=UTC), "2026-07-01 00:00 至 2026-08-01 00:00"),
])
async def test_default_month_uses_request_shanghai_calendar(instant: datetime, expected: str) -> None:
    ctx = replace(context(responses=[]), now=instant)
    result = await clarify(
        clarification_state(dimensions=[MissingDimension.PERIOD]), Runtime(context=ctx)
    )
    assert expected in result.update["route_clarification"].message


async def test_missing_metric_preserves_explicit_period_and_region() -> None:
    ctx = context(responses=[])
    result = await clarify(
        clarification_state(question="2024年7月华南的业务数据", dimensions=[MissingDimension.METRIC]),
        Runtime(context=ctx),
    )
    value = result.update["route_clarification"]
    assert "2024年7月华南" in value.suggested_question
    assert "2026年8月" not in value.suggested_question
    assert "指标选择GMV" in value.suggested_question


@pytest.mark.parametrize(("kind", "dimension"), [
    (ClarificationKind.PERIOD_UNRESOLVED, MissingDimension.PERIOD),
    (ClarificationKind.METRIC_NOT_IDENTIFIED, MissingDimension.METRIC),
    (ClarificationKind.REGION_UNRESOLVED, MissingDimension.REGION),
    (ClarificationKind.UNSUPPORTED_GRAIN, MissingDimension.GRAIN),
    (ClarificationKind.INVALID_EXPLICIT_PATCH, MissingDimension.DEFINITION),
    (ClarificationKind.METRIC_NOT_FOUND, MissingDimension.METRIC),
])
async def test_specialist_reason_uses_enum_not_error_prose(
    kind: ClarificationKind, dimension: MissingDimension
) -> None:
    ctx = context(responses=[])
    state = clarification_state()
    state.data_clarification = MetricClarification(kind=kind, message="misleading error prose")
    result = await clarify(state, Runtime(context=ctx))
    value = result.update["route_clarification"]
    assert value.kind is kind
    assert dimension in value.intent.missing_dimensions
    assert "misleading error prose" not in value.message


async def test_knowledge_and_data_missing_dimensions_are_both_retained() -> None:
    ctx = context(responses=[])
    state = clarification_state()
    state.data_clarification = MetricClarification(
        kind=ClarificationKind.METRIC_NOT_IDENTIFIED, message="metric"
    )
    state.knowledge_clarification = KnowledgeClarification(
        kind=KnowledgeClarificationKind.PERIOD_UNRESOLVED, message="period"
    )
    result = await clarify(state, Runtime(context=ctx))
    assert result.update["route_clarification"].intent.missing_dimensions == [
        MissingDimension.METRIC, MissingDimension.PERIOD
    ]


async def test_no_history_or_catalog_does_not_invent_capabilities() -> None:
    ctx = context(responses=[])
    capabilities = ClarificationCapabilities()
    ctx = replace(ctx, clarification_capabilities=AsyncMock(read=AsyncMock(return_value=capabilities)))
    result = await clarify(
        clarification_state(ClarificationCategory.AMBIGUOUS_REFERENCE, question="那个"),
        Runtime(context=ctx),
    )
    value = result.update["route_clarification"]
    assert "没有可用的近期话题" in value.message
    assert "暂无已发布指标" in value.message and "暂无已发布知识文档" in value.message
    assert value.suggested_question.startswith("请先配置")


@pytest.mark.parametrize("error", [ConflictError(), DeadlineExceededError()])
async def test_capability_failures_remain_failures(error: Exception) -> None:
    ctx = replace(context(responses=[]), clarification_capabilities=AsyncMock(
        read=AsyncMock(side_effect=error)
    ))
    result = await clarify(clarification_state(), Runtime(context=ctx))
    assert result.update["status"] == "failed"
    assert "route_clarification" not in result.update
    assert ctx.llm.calls == ctx.mcp.calls == []


async def test_missing_service_is_typed_failure() -> None:
    ctx = replace(context(responses=[]), clarification_capabilities=None)
    result = await clarify(clarification_state(), Runtime(context=ctx))
    assert result.update["status"] == "failed"


async def test_clarification_does_not_mutate_prior_contracts() -> None:
    ctx = context(responses=[])
    state = clarification_state(dimensions=[MissingDimension.PERIOD])
    before = state.model_dump_json()
    capabilities = await ctx.clarification_capabilities.read(deadline=ctx.deadline)
    render_clarification(state, capabilities, now=ctx.now)
    assert state.model_dump_json() == before


def test_legacy_clarification_roundtrips_without_v2_fields() -> None:
    payload = {
        "schema_version": 1, "kind": "reference_unresolved", "message": "old question",
        "metric_key": None, "available_metrics": [], "supported_grains": [],
    }
    assert MetricClarification.model_validate(payload).model_dump(mode="json") == payload


def test_v2_requires_policy_and_suggestion() -> None:
    with pytest.raises(ValidationError):
        MetricClarification(schema_version=2, kind=ClarificationKind.OUT_OF_SCOPE, message="invalid")


async def test_policy_prompt_shares_the_single_router_call() -> None:
    from app.agents.contracts import RouterInput
    from app.agents.nodes.router import route_question
    from app.agents.runtime import RoutingRuntime

    decision = clarification_state(dimensions=[MissingDimension.PERIOD]).route
    ctx = context(responses=[decision])
    result = await route_question(
        RouterInput(question="查一下华东GMV"),
        RoutingRuntime(llm=ctx.llm, settings=ctx.settings.router, deadline=ctx.deadline),
    )
    assert result.clarification_intent == decision.clarification_intent
    assert len(ctx.llm.calls) == 1
    assert "Clarification policy v1" in ctx.llm.calls[0].messages[0].content
    assert "untrusted DATA" in ctx.llm.calls[0].messages[0].content


async def test_document_only_capability_gives_document_example() -> None:
    from app.schemas.corpus import DocumentType

    ctx = replace(context(responses=[]), clarification_capabilities=AsyncMock(
        read=AsyncMock(return_value=ClarificationCapabilities(document_categories=[DocumentType.SOP]))
    ))
    result = await clarify(
        clarification_state(ClarificationCategory.OUT_OF_SCOPE), Runtime(context=ctx)
    )
    assert "操作流程" in result.update["route_clarification"].suggested_question


async def test_failed_or_evidence_bearing_state_cannot_become_clarification() -> None:
    from app.agents.nodes.common import node_failure

    ctx = context(responses=[])
    state = clarification_state()
    state.failures = [node_failure("data", ConflictError())]
    result = await clarify(state, Runtime(context=ctx))
    assert result.update["status"] == "failed"
    assert "route_clarification" not in result.update
