"""Real specialist wrappers preserve terminal isolation and invocation ownership."""

# ruff: noqa: PLR2004 -- exact call/reducer counts are acceptance contracts.

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.data.state import DataAgentOutput
from app.agents.failures import FailureKind
from app.agents.knowledge.nodes.package_evidence import package_evidence
from app.agents.knowledge.state import KnowledgeAgentOutput
from app.agents.nodes.answer_data import answer_data_routed, answer_data_routed_node
from app.agents.nodes.answer_knowledge import answer_knowledge, answer_knowledge_node
from app.agents.runtime import RuntimeContext
from app.agents.state import AgentState, TurnContext
from app.agents.summarize import package_result
from app.core.deadline import Deadline
from app.core.errors import McpUnavailableError, RetrievalUnavailableError, SqlExecutionError
from app.schemas.knowledge_query import KnowledgeClarification, KnowledgeClarificationKind
from app.schemas.mcp import SqlErrorKind
from app.schemas.metric_resolution import ClarificationKind, MetricClarification, MetricIntent
from app.schemas.model_runtime import ModelFailureKind
from app.services.graph import serializer
from tests.agents.correction_support import correction_context
from tests.agents.knowledge_support import BlockingRetrieval, FakeRetrieval, ranked
from tests.agents.projection_support import routed_state
from tests.agents.state_support import failure
from tests.agents.support import context, result
from tests.knowledge_support import retrieval as retrieval_result


async def test_routed_data_success_and_assumptions() -> None:
    ctx = context()
    state = routed_state(ctx)
    output = await answer_data_routed(state, Runtime(context=ctx), {})
    assert output.update["data_evidence"].metric_bindings
    assert output.update["assumptions"] == output.update["data_evidence"].assumptions
    assert output.update["failures"] == []
    assert not output.goto
    assert not {"status", "answer", "clarification", "knowledge_evidence"} & output.update.keys()


async def test_data_clarification_has_its_own_channel() -> None:
    ctx = context(responses=[MetricIntent(metric_keys=[])])
    output = await answer_data_routed(routed_state(ctx), Runtime(context=ctx), {})
    assert output.update["data_clarification"].kind is ClarificationKind.METRIC_NOT_IDENTIFIED
    assert output.update["data_evidence"] is None
    assert output.update["failures"] == []
    assert ctx.mcp.calls == []


async def test_terminal_failure_is_only_a_new_delta() -> None:
    ctx = context(mcp_results=[McpUnavailableError()])
    state = routed_state(ctx)
    state.failures = [failure("earlier")]
    output = await answer_data_routed(state, Runtime(context=ctx), {})
    assert len(output.update["failures"]) == len(state.failures) == 1
    assert output.update["failures"][0].kind is FailureKind.MCP_UNAVAILABLE
    assert state.failures[0].node == "earlier"
    assert not output.goto
    assert "status" not in output.update


async def test_recovered_child_error_does_not_fail_routed_parent() -> None:
    ctx = correction_context([SqlExecutionError(SqlErrorKind.UNDEFINED_TABLE), result()])
    state = routed_state(ctx)
    state.context = TurnContext(time_scope=state.context.time_scope)
    output = await answer_data_routed(state, Runtime(context=ctx), {})
    assert output.update["data_evidence"] is not None
    assert output.update["failures"] == []
    assert len(ctx.mcp.calls) == 2


async def test_data_evidence_reuse_precedes_strict_projection() -> None:
    ctx = context()
    snapshot = await ctx.evidence.commit(ctx.identity, package_result(result(), []))
    state = AgentState(**ctx.identity.model_dump(), failures=[failure("old")])
    replay = replace(ctx, llm=Mock(), mcp=Mock())
    output = await answer_data_routed(state, Runtime(context=replay), {})
    assert output.update["data_evidence"] == snapshot.data
    assert output.update["assumptions"] == output.update["failures"] == []
    replay.llm.generate_structured.assert_not_called()
    replay.mcp.call_tool.assert_not_called()


@pytest.mark.parametrize("wrapper", [answer_data_routed, answer_knowledge])
async def test_strict_wrapper_never_falls_back_to_prepared(
    wrapper: Callable[..., Awaitable[Command[str]]],
) -> None:
    ctx = context()
    state = routed_state(ctx)
    state.context = None
    output = await wrapper(state, Runtime(context=ctx), {})
    assert list(output.update) == ["failures"]
    assert output.update["failures"][0].kind is FailureKind.NODE_OPERATION_FAILED
    assert not output.goto
    assert ctx.llm.calls == ctx.mcp.calls == []


@pytest.mark.parametrize("wrapper", [answer_data_routed, answer_knowledge])
async def test_wrapper_deadline_is_a_delta(
    wrapper: Callable[..., Awaitable[Command[str]]],
) -> None:
    ctx = replace(context(), deadline=Deadline(0))
    output = await wrapper(routed_state(ctx), Runtime(context=ctx), {})
    assert output.update["failures"][0].kind is FailureKind.DEADLINE_EXCEEDED
    assert "status" not in output.update
    assert not output.goto


async def test_knowledge_success_uses_scoped_input_and_runtime() -> None:
    retrieval = FakeRetrieval(ranked())
    ctx = replace(context(), retrieval=retrieval)
    state = routed_state(ctx)
    output = await answer_knowledge(state, Runtime(context=ctx), {})
    assert output.update["knowledge_evidence"].chunks
    assert output.update["failures"] == []
    assert output.update["knowledge_clarification"] is None
    assert output.update["knowledge_abstention_reason"] is None
    assert retrieval.calls[0].standalone == state.route.knowledge_intent
    assert retrieval.deadlines == [ctx.deadline]
    assert ctx.llm.calls == ctx.mcp.calls == []
    assert (
        not {"status", "answer", "clarification", "abstained", "data_evidence"}
        & output.update.keys()
    )


async def test_knowledge_below_floor_refuses_without_failure() -> None:
    ctx = replace(context(), retrieval=FakeRetrieval(ranked(0.1)))
    output = await answer_knowledge(routed_state(ctx), Runtime(context=ctx), {})
    assert output.update["knowledge_evidence"] is None
    assert output.update["knowledge_abstention_reason"]
    assert output.update["failures"] == []


async def test_knowledge_rerank_degradation_is_preserved() -> None:
    value = retrieval_result()
    value.retrieval_config.use_rerank = True
    value.degradation = ModelFailureKind.UNAVAILABLE
    ctx = replace(context(), retrieval=FakeRetrieval(value))
    output = await answer_knowledge(routed_state(ctx), Runtime(context=ctx), {})
    assert output.update["knowledge_evidence"].degradation is ModelFailureKind.UNAVAILABLE
    assert output.update["degraded_components"] == ["rerank"]
    assert output.update["failures"] == []


async def test_knowledge_typed_service_failure_is_preserved() -> None:
    ctx = replace(context(), retrieval=FakeRetrieval(RetrievalUnavailableError()))
    output = await answer_knowledge(routed_state(ctx), Runtime(context=ctx), {})
    assert output.update["failures"][0].kind is FailureKind.RETRIEVAL_UNAVAILABLE
    assert output.update["knowledge_abstention_reason"] is None


async def test_knowledge_existing_evidence_skips_all_services() -> None:
    ctx = context()
    state = routed_state(ctx)
    state.knowledge_evidence = package_evidence(ranked(), ctx)
    state.context = None
    replay = replace(ctx, retrieval=Mock(), llm=Mock(), evidence=Mock())
    output = await answer_knowledge(state, Runtime(context=replay), {})
    assert output.update == {"knowledge_evidence": state.knowledge_evidence}
    replay.retrieval.retrieve.assert_not_called()
    replay.evidence.find.assert_not_called()
    replay.llm.generate_structured.assert_not_called()


@pytest.mark.parametrize(
    ("wrapper", "target", "outcome", "channel"),
    [
        (
            answer_data_routed,
            "app.agents.nodes.answer_data.DATA_GRAPH.ainvoke",
            DataAgentOutput(
                clarification=MetricClarification(
                    kind=ClarificationKind.PERIOD_UNRESOLVED, message="period?"
                )
            ),
            "data_clarification",
        ),
        (
            answer_knowledge,
            "app.agents.nodes.answer_knowledge.KNOWLEDGE_GRAPH.ainvoke",
            KnowledgeAgentOutput(
                clarification=KnowledgeClarification(
                    kind=KnowledgeClarificationKind.PERIOD_UNRESOLVED, message="policy period?"
                )
            ),
            "knowledge_clarification",
        ),
    ],
)
async def test_wrapper_preserves_checkpoint_config_and_clarification(
    monkeypatch: pytest.MonkeyPatch,
    wrapper: Callable[..., Awaitable[Command[str]]],
    target: str,
    outcome: DataAgentOutput | KnowledgeAgentOutput,
    channel: str,
) -> None:
    ctx = context()
    child = AsyncMock(return_value=outcome.model_dump())
    monkeypatch.setattr(target, child)
    config = {
        "configurable": {"thread_id": str(ctx.identity.turn_id), "checkpoint_ns": "parent:child"},
        "callbacks": [],
        "recursion_limit": 32,
    }
    output = await wrapper(routed_state(ctx), Runtime(context=ctx), config)
    assert output.update[channel] == outcome.clarification
    assert child.call_args.args[1] is config
    assert child.call_args.kwargs["context"] is ctx
    assert "status" not in output.update


async def test_cancellation_propagates_to_knowledge_service() -> None:
    retrieval = BlockingRetrieval()
    ctx = replace(context(), retrieval=retrieval)
    async with asyncio.TaskGroup() as group:
        task = group.create_task(answer_knowledge(routed_state(ctx), Runtime(context=ctx), {}))
        await asyncio.wait_for(retrieval.started.wait(), timeout=2)
        task.cancel()
    assert task.cancelled()
    assert retrieval.stopped.is_set()


def wrapper_graph() -> CompiledStateGraph[AgentState, RuntimeContext, AgentState, AgentState]:
    graph = StateGraph(AgentState, context_schema=RuntimeContext)
    graph.add_node("answer_data_routed", answer_data_routed_node)
    graph.add_node("answer_knowledge", answer_knowledge_node)
    graph.add_edge(START, "answer_data_routed")
    graph.add_edge(START, "answer_knowledge")
    graph.add_edge("answer_data_routed", END)
    graph.add_edge("answer_knowledge", END)
    return graph.compile(checkpointer=InMemorySaver(serde=serializer()))


@pytest.mark.parametrize("fail_data", [False, True])
@pytest.mark.parametrize("fail_knowledge", [False, True])
async def test_compiled_wrappers_preserve_parallel_ownership_and_deltas(
    fail_data: bool, fail_knowledge: bool
) -> None:
    ctx = replace(
        context(mcp_results=[McpUnavailableError()] if fail_data else [result()]),
        retrieval=FakeRetrieval(RetrievalUnavailableError() if fail_knowledge else ranked()),
    )
    state = routed_state(ctx)
    prior = failure("prior")
    state.failures = [prior]
    state.assumptions = ["prior assumption"]
    config = {"configurable": {"thread_id": str(ctx.identity.turn_id)}}
    graph = wrapper_graph()
    updates = [
        update async for update in graph.astream(state, config, context=ctx, stream_mode="updates")
    ]
    writes = {node: command for update in updates for node, command in update.items()}
    shared = writes["answer_data_routed"].keys() & writes["answer_knowledge"].keys()
    assert shared <= {"assumptions", "failures", "degraded_components"}
    restored = AgentState.model_validate((await graph.aget_state(config)).values)
    assert (restored.knowledge_evidence is None) is fail_knowledge
    assert (restored.data_evidence is None) is fail_data
    assert restored.failures.count(prior) == 1
    assert len(restored.failures) == 1 + int(fail_data) + int(fail_knowledge)
    assert restored.assumptions.count("prior assumption") == 1
    assert restored.context == state.context
    assert restored.route == state.route


@pytest.mark.parametrize(
    ("wrapper", "target"),
    [
        (answer_data_routed, "app.agents.nodes.answer_data.DATA_GRAPH.ainvoke"),
        (answer_knowledge, "app.agents.nodes.answer_knowledge.KNOWLEDGE_GRAPH.ainvoke"),
    ],
)
async def test_wrapper_does_not_translate_cancellation_into_failure(
    monkeypatch: pytest.MonkeyPatch,
    wrapper: Callable[..., Awaitable[Command[str]]],
    target: str,
) -> None:
    ctx = context()
    state = routed_state(ctx)
    child = AsyncMock(side_effect=asyncio.CancelledError())
    monkeypatch.setattr(target, child)
    with pytest.raises(asyncio.CancelledError):
        await wrapper(state, Runtime(context=ctx), {})
    assert state.failures == []
