"""Production graph gates memory before specialist projection and restarts at most once."""

# ruff: noqa: PLR2004 -- bounded restart/call-count acceptance contracts.

import asyncio
import inspect
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from langgraph.runtime import Runtime

from app.agents.contracts import PreparedContext, Route, RouteDecision
from app.agents.nodes.prepare import prepare
from app.agents.nodes.finalize_context import finalize_context
from app.agents.state import AgentState
from app.agents.runtime import RuntimeContext
from app.core.deadline import Deadline
from app.core.errors import LlmUnavailableError
from app.schemas.memory import MemoryType
from app.schemas.memory_retrieval import MemoryStage
from app.schemas.metric_resolution import RegionReference, RegionScope
from app.services.metric_intent import MetricIntentService
from app.services.knowledge_generation import KnowledgeGenerationService
from tests.agents.knowledge_support import FakeRetrieval, ranked
from tests.agents.parent_support import parent_context
from tests.agents.support import invoke
from tests.agents.synthesis_support import synthesis_draft
from tests.fakes.chat_model import FakeChatModel
from tests.knowledge_support import draft
from tests.memory_retrieval_support import MemoryPort, stored


def memory_context(route: Route = Route.DATA_ONLY) -> RuntimeContext:
    ctx = parent_context(route)
    port = MemoryPort([])
    return replace(ctx, memories=port, metric_intents=MetricIntentService(ctx.llm, ctx.metrics))


async def test_format_preference_applies_to_data_answer() -> None:
    ctx = memory_context()
    ctx.memories.rows = [stored(MemoryType.FORMAT_PREFERENCE, user=ctx.identity.user_id)]
    result = await invoke(ctx)
    assert result.status == "succeeded"
    assert result.answer.format_preference.prefer == "table"
    assert result.answer.format_preference.decimals == 3
    assert sum(c.schema_name == "MetricIntent" for c in ctx.llm.calls) == 1
    assert [r.stage for r in ctx.memories.calls] == [MemoryStage.PREPARE, MemoryStage.FINALIZE]
    for call in ctx.llm.calls:
        if call.schema_name == "MetricIntent":
            assert '"prefer"' not in str(call.messages)


async def test_explicit_format_fields_beat_saved_fields() -> None:
    ctx = memory_context()
    ctx.memories.rows = [stored(MemoryType.FORMAT_PREFERENCE, user=ctx.identity.user_id)]
    ctx.conversations.prepare.return_value = PreparedContext(question="2026年8月经营情况，用文字回答")
    result = await invoke(ctx)
    assert result.answer.format_preference.prefer == "prose"
    assert result.answer.format_preference.decimals == 3


async def test_metric_override_not_retrieved_for_knowledge_question() -> None:
    ctx = memory_context(Route.KNOWLEDGE_ONLY)
    ctx.memories.rows = [stored(MemoryType.METRIC_OVERRIDE, user=ctx.identity.user_id)]
    result = await invoke(ctx)
    assert result.status == "succeeded"
    assert ctx.memories.calls[-1].data_route is False
    assert ctx.mcp.calls == []
    assert not any(c.schema_name == "MetricIntent" for c in ctx.llm.calls)


@pytest.mark.parametrize("failure_stage", [1, 2])
async def test_failure_returns_degraded_not_silent_empty(failure_stage: int) -> None:
    ctx = memory_context()
    ctx.memories.fail_at = failure_stage
    ctx.memories.rows = [stored(MemoryType.FORMAT_PREFERENCE, user=ctx.identity.user_id)]
    result = await invoke(ctx)
    assert result.status == "degraded"
    assert result.answer.degraded_components == ["memory"]
    assert result.answer.format_preference is None
    assert len(ctx.memories.calls) == failure_stage


async def test_failed_final_read_removes_previously_used_terminology_and_reroutes_once() -> None:
    ctx = memory_context(Route.KNOWLEDGE_ONLY)
    row = stored(user=ctx.identity.user_id)
    ctx.memories.rows = [row]
    ctx.memories.fail_at = 2
    ctx.conversations.prepare.return_value = PreparedContext(question="大促政策是什么")
    route = RouteDecision(route=Route.KNOWLEDGE_ONLY, confidence=1, knowledge_intent="查找大促政策")
    retrieval = ranked()
    llm = FakeChatModel([route, route, draft(retrieval.candidates[0].chunk_uuid)])
    ctx = replace(ctx, llm=llm, retrieval=FakeRetrieval(retrieval), knowledge_generation=KnowledgeGenerationService(llm))
    result = await invoke(ctx)
    assert result.status == "degraded"
    assert len(ctx.memories.calls) == 2
    routes = [c for c in llm.calls if c.schema_name == "RouteDecision"]
    assert len(routes) == 2
    assert '618' in str(routes[0].messages)
    assert '618' not in str(routes[1].messages)
    assert result.answer.degraded_components == ["memory"]


async def test_region_focus_not_retrieved_when_region_stated() -> None:
    ctx = memory_context(Route.KNOWLEDGE_ONLY)
    ctx.memories.rows = [stored(MemoryType.REGION_FOCUS, user=ctx.identity.user_id)]
    regions = AsyncMock(resolve=AsyncMock(return_value=RegionScope()))
    ctx = replace(ctx, regions=regions)
    route = RouteDecision(route=Route.KNOWLEDGE_ONLY, confidence=1, knowledge_intent="核查全部区域政策",
                          region_mentioned=True, region=RegionReference(all_regions=True))
    state = AgentState(**ctx.identity.model_dump(), question="全部区域政策", route=route,
                       prepared=PreparedContext(question="全部区域政策"), routing_context={})
    result = await finalize_context(state, Runtime(context=ctx))
    assert result.update["context"].region_scope == RegionScope()
    assert result.update["context"].memories == []
    assert ctx.memories.calls[-1].region_mentioned is True


async def test_intent_failure_keeps_both_knowledge_branch_alive() -> None:
    ctx = memory_context(Route.BOTH)
    decision = RouteDecision(route=Route.BOTH, confidence=1, data_intent="计算GMV", knowledge_intent="核查退款政策")
    retrieval = ranked()
    llm = FakeChatModel([decision, LlmUnavailableError(), synthesis_draft(data=False, knowledge_id=retrieval.candidates[0].chunk_uuid)])
    ctx = replace(ctx, llm=llm, retrieval=FakeRetrieval(retrieval), metric_intents=MetricIntentService(llm, ctx.metrics))
    result = await invoke(ctx)
    assert result.answer.evidence_refs.data_snapshot_id is None
    assert result.answer.evidence_refs.knowledge_snapshot_id is not None
    assert len(ctx.retrieval.calls) == 1
    assert len(ctx.mcp.calls) == 0


async def test_superseded_preroute_term_forces_memory_free_restart() -> None:
    ctx = memory_context(Route.KNOWLEDGE_ONLY)
    row = stored(user=ctx.identity.user_id)
    ctx.memories.rows = [row]
    ctx.conversations.prepare.return_value = PreparedContext(question="大促政策")
    state = AgentState(**ctx.identity.model_dump())
    prepared = await prepare(state, Runtime(context=ctx))
    state = AgentState.model_validate({**state.model_dump(), **prepared.update})
    state.route = RouteDecision(route=Route.KNOWLEDGE_ONLY, confidence=1, knowledge_intent="查找大促政策")
    ctx.memories.rows = [row.model_copy(update={"is_active": False})]
    final = await finalize_context(state, Runtime(context=ctx))
    assert final.update["memory_disabled"] is True
    assert final.update["memory_restart_pending"] is True
    assert final.update["degraded_components"] == []


async def test_cancellation_is_not_memory_degradation() -> None:
    ctx = memory_context()
    ctx = replace(ctx, memories=AsyncMock(retrieve=AsyncMock(side_effect=asyncio.CancelledError())))
    with pytest.raises(asyncio.CancelledError):
        await invoke(ctx)


async def test_request_deadline_is_not_renewed() -> None:
    ctx = replace(memory_context(), deadline=Deadline(0))
    result = await invoke(ctx)
    assert result.status == "failed"
    assert ctx.memories.calls == []


def test_no_node_signature_changed_since_phase_4() -> None:
    from app.agents.nodes.router import parent_router
    from app.agents.nodes.answer_data import answer_data_routed_node
    from app.agents.nodes.answer_knowledge import answer_knowledge_node
    for node in [prepare, parent_router, finalize_context]:
        assert list(inspect.signature(node).parameters) == ["state", "runtime"]
    for node in [answer_data_routed_node, answer_knowledge_node]:
        assert list(inspect.signature(node).parameters) == ["state", "runtime"]
        assert inspect.signature(node).parameters["runtime"].kind is inspect.Parameter.KEYWORD_ONLY


async def test_budget_eviction_of_used_term_requests_one_restart() -> None:
    ctx = memory_context(Route.DATA_ONLY)
    term = stored(user=ctx.identity.user_id, content={"term": "GMV", "means": "".join(chr(0x4e00 + i) for i in range(190))}, summary="词义")
    # A deliberately exact counter isolates the shared-budget eviction from tokenization.
    class Counter:
        name = "cl100k_base"
        def count(self, text: str) -> int:
            return 601 if "terminology" in text and "metric_override" in text else 100
    ctx = replace(ctx, schema_token_counter=Counter())
    ctx.memories.rows = [term]
    ctx.conversations.prepare.return_value = PreparedContext(question="GMV GMV GMV")
    state = AgentState(**ctx.identity.model_dump())
    prepared = await prepare(state, Runtime(context=ctx))
    state = AgentState.model_validate({**state.model_dump(), **prepared.update})
    state.route = RouteDecision(route=Route.DATA_ONLY, confidence=1, data_intent="计算GMV")
    override = stored(MemoryType.METRIC_OVERRIDE, user=ctx.identity.user_id,
                      content={"metric_key": "gmv", "patch": {"date_field": "o.created_at"}}, summary="GMV")
    ctx.memories.rows.append(override)
    from app.schemas.metric_resolution import MetricIntent
    ctx = replace(ctx, metric_intents=AsyncMock(interpret=AsyncMock(return_value=MetricIntent(metric_keys=["gmv"])) ))
    result = await finalize_context(state, Runtime(context=ctx))
    assert result.update["memory_restart_pending"] is True
    assert result.update["degraded_components"] == []


async def test_selected_metric_is_projected_with_provenance_and_no_hidden_lookup() -> None:
    ctx = memory_context()
    row = stored(MemoryType.METRIC_OVERRIDE, user=ctx.identity.user_id,
                 content={"metric_key": "gmv", "patch": {"date_field": "o.created_at"}})
    ctx.memories.rows = [row]
    from app.schemas.metric_resolution import MetricIntent
    ctx = replace(ctx, metric_intents=AsyncMock(interpret=AsyncMock(return_value=MetricIntent(metric_keys=["gmv"], period_expression="2026年8月"))))
    state = AgentState(**ctx.identity.model_dump(), question="2026年8月GMV",
        route=RouteDecision(route=Route.DATA_ONLY, confidence=1, data_intent="计算2026年8月GMV"),
        prepared=PreparedContext(question="2026年8月GMV"), routing_context={})
    result = await finalize_context(state, Runtime(context=ctx))
    context = result.update["context"]
    assert context.selected_overrides.items[0].id == row.id
    assert context.selected_overrides.items[0].created_at == row.created_at
    from app.agents.projections import to_data_input
    state.context = context
    inputs = to_data_input(state, token_counter=ctx.schema_token_counter)
    assert inputs.prepared_intent == context.prepared_intent
    assert inputs.selected_overrides == context.selected_overrides
    assert len(ctx.memories.calls) == 1
    from app.services.graph import serializer
    kind, payload = serializer().dumps_typed(state)
    restored = serializer().loads_typed((kind, payload))
    assert restored.context.selected_overrides == context.selected_overrides
