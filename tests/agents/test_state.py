"""Parent v2 reducers, write ownership, serialization and fresh-turn isolation."""

import json
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.channels.binop import BinaryOperatorAggregate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from app.agents.contracts import HistoryMessage, Route, RouterInput, RoutingContext
from app.agents.nodes.common import failed
from app.agents.nodes.router import route_question
from app.agents.state import GRAPH_VERSION, AgentState, GraphInput, TurnContext
from app.core.errors import ConflictError
from app.schemas.memory import FormatPreferenceContent, TerminologyContent
from app.schemas.metric_resolution import RegionScope, SelectedMetricOverride, SelectedOverrides
from app.schemas.retrieval import PolicyPeriod, RangeTimeScope
from app.services.graph import serializer
from tests.agents.state_support import contract_topology, failure, finalized, memory
from tests.agents.support import context
from tests.fakes.chat_model import FakeChatModel
from tests.router_support import BOTH_QUESTION, decision, runtime


async def test_assumptions_accumulate_from_both_specialists() -> None:
    ctx = context(responses=[decision()])
    saver = InMemorySaver(serde=serializer())
    graph = contract_topology().compile(checkpointer=saver)
    config = {"configurable": {"thread_id": str(ctx.identity.turn_id)}}
    await graph.ainvoke(GraphInput(**ctx.identity.model_dump()), config, context=ctx)
    state = AgentState.model_validate((await graph.aget_state(config)).values)
    assert sorted(state.assumptions) == ["data_agent", "knowledge_agent"]
    assert state.data_evidence is not None
    assert state.knowledge_evidence is not None
    assert state.context.memories == []
    assert state.context.selected_overrides.items == []


async def test_failures_accumulate() -> None:
    ctx = context(responses=[decision()])
    graph = contract_topology(failed_nodes=frozenset({"data_agent", "knowledge_agent"})).compile(
        checkpointer=InMemorySaver(serde=serializer())
    )
    config = {"configurable": {"thread_id": str(ctx.identity.turn_id)}}
    await graph.ainvoke(GraphInput(**ctx.identity.model_dump()), config, context=ctx)
    state = AgentState.model_validate((await graph.aget_state(config)).values)
    assert sorted(item.node for item in state.failures) == ["data_agent", "knowledge_agent"]
    assert sorted(state.degraded_components) == ["data_agent", "knowledge_agent"]
    assert state.data_evidence is state.knowledge_evidence is None


async def test_no_concurrent_writes_to_same_field() -> None:
    ctx = context(responses=[decision()])
    graph = contract_topology().compile()
    writes = {}
    async for update in graph.astream(
        GraphInput(**ctx.identity.model_dump()), context=ctx, stream_mode="updates"
    ):
        for node, values in update.items():
            writes.setdefault(node, set()).update(values)
    assert writes["data_agent"] == {"data_evidence", "assumptions"}
    assert writes["knowledge_agent"] == {"knowledge_evidence", "assumptions"}
    assert writes["data_agent"] & writes["knowledge_agent"] == {"assumptions"}
    assert writes["prepare"] == {"prepared", "question", "messages", "routing_context"}
    assert writes["router"] == {"route"}
    assert writes["finalize_context"] == {"context"}
    assert writes["finish"] == {"status"}


async def test_conflicting_parallel_single_writer_field_is_rejected() -> None:
    ctx = context(responses=[decision()])
    with pytest.raises(InvalidUpdateError, match="question"):
        await (
            contract_topology(collide=True)
            .compile()
            .ainvoke(GraphInput(**ctx.identity.model_dump()), context=ctx)
        )


async def test_context_finalized_after_current_route() -> None:
    ctx = context(responses=[decision(Route.KNOWLEDGE_ONLY)])
    graph = contract_topology().compile()
    values = [
        AgentState.model_validate(value)
        async for value in graph.astream(
            GraphInput(**ctx.identity.model_dump()), context=ctx, stream_mode="values"
        )
    ]
    routed = next(item for item in values if item.route is not None)
    assert routed.context is None
    finalized_state = next(item for item in values if item.context is not None)
    assert finalized_state.context.summary == Route.KNOWLEDGE_ONLY.value
    assert finalized_state.route == decision(Route.KNOWLEDGE_ONLY)
    assert values[-1].context == finalized_state.context
    assert values[-1].routing_context == routed.routing_context
    with pytest.raises(ValidationError):
        finalized_state.context.summary = "changed"


async def test_both_then_data_only_has_no_old_knowledge() -> None:
    graph = contract_topology().compile(checkpointer=InMemorySaver(serde=serializer()))
    ctx = context()
    states = []
    for route in (Route.BOTH, Route.DATA_ONLY, Route.KNOWLEDGE_ONLY):
        identity = ctx.identity.model_copy(update={"turn_id": uuid4()})
        current = replace(ctx, identity=identity, llm=FakeChatModel([decision(route)]))
        config = {"configurable": {"thread_id": str(identity.turn_id)}}
        await graph.ainvoke(GraphInput(**identity.model_dump()), config, context=current)
        states.append(AgentState.model_validate((await graph.aget_state(config)).values))
    assert states[0].knowledge_evidence is not None
    assert states[1].knowledge_evidence is None
    assert states[2].data_evidence is None
    assert states[1].assumptions == ["data_agent"]
    assert states[2].assumptions == ["knowledge_agent"]
    for state in states:
        assert state.evidence_refs is state.answer is None
        assert state.failures == state.degraded_components == []
        assert not state.abstained


async def test_failed_then_successful_turn_has_no_old_failures() -> None:
    saver = InMemorySaver(serde=serializer())
    ctx = context(responses=[decision()])
    config = {"configurable": {"thread_id": str(ctx.identity.turn_id)}}
    first = contract_topology(failed_nodes=frozenset({"data_agent", "knowledge_agent"})).compile(
        checkpointer=saver
    )
    await first.ainvoke(GraphInput(**ctx.identity.model_dump()), config, context=ctx)
    assert (await first.aget_state(config)).values["failures"]
    identity = ctx.identity.model_copy(update={"turn_id": uuid4()})
    current = replace(ctx, identity=identity, llm=FakeChatModel([decision()]))
    config = {"configurable": {"thread_id": str(identity.turn_id)}}
    second = contract_topology().compile(checkpointer=saver)
    output = await second.ainvoke(GraphInput(**identity.model_dump()), config, context=current)
    assert output["status"] == "succeeded"
    state = AgentState.model_validate((await second.aget_state(config)).values)
    assert state.failures == state.degraded_components == []


def test_memory_slot_present_and_empty() -> None:
    first, second = finalized(), finalized()
    assert first.memories == second.memories == []
    first.memories.append(memory())
    assert second.memories == []
    assert first.selected_overrides.items == []
    assert first.format_preference is None


def test_state_serializable_for_checkpointer() -> None:
    ctx = context()
    messages = [
        HumanMessage(content="question", id="human"),
        AIMessage(content="", id="ai", tool_calls=[{"id": "call", "name": "lookup", "args": {}}]),
        ToolMessage(content="result", tool_call_id="call", id="tool"),
    ]
    state = AgentState(
        **ctx.identity.model_dump(),
        messages=messages,
        context=TurnContext(
            time_scope=finalized().time_scope,
            memories=[memory()],
            recent_messages=messages,
            format_preference=FormatPreferenceContent(prefer="table", decimals=2),
            prior_sql=["SELECT 42"],
            token_accounting={"history": 12},
            region_scope=RegionScope(region_ids=[1]),
            selected_overrides=SelectedOverrides(
                items=[
                    SelectedMetricOverride(
                        id=uuid4(),
                        user_id=ctx.identity.user_id,
                        created_at=datetime.now(UTC),
                        confidence=1,
                        metric_key="gmv",
                        patch={"date_field": "paid_at"},
                    )
                ]
            ),
        ),
        routing_context=RoutingContext(terminology=[TerminologyContent(term="大促", means="618")]),
        route=decision(),
        failures=[failure("test")],
    )
    codec = serializer()
    restored = codec.loads_typed(codec.dumps_typed(state))
    assert isinstance(restored, AgentState)
    assert restored == state
    assert restored.messages[-1].tool_call_id == "call"
    assert restored.context.memories[0].content.term == "大促"
    assert AgentState.model_validate_json(state.model_dump_json()) == state
    assert not codec.pickle_fallback


def test_only_concurrent_fields_have_reducers() -> None:
    graph = contract_topology().compile()
    assert {
        name
        for name, channel in graph.channels.items()
        if isinstance(channel, BinaryOperatorAggregate)
    } == {"messages", "assumptions", "failures", "degraded_components"}


def test_range_context_checkpoint_roundtrip() -> None:
    scope = RangeTimeScope(periods=[PolicyPeriod(start="2026-07-01", end="2026-09-01")])
    value = TurnContext(time_scope=scope)
    codec = serializer()
    assert codec.loads_typed(codec.dumps_typed(value)) == value
    assert TurnContext.model_validate_json(value.model_dump_json()) == value


async def test_messages_merge_by_id_preserves_tool_structure() -> None:
    ctx = context()
    graph = StateGraph(AgentState)

    def update(state: AgentState) -> dict[str, list[AIMessage]]:
        return {
            "messages": [
                AIMessage(
                    content="updated",
                    id="ai",
                    tool_calls=[{"id": "call", "name": "lookup", "args": {"value": 42}}],
                )
            ]
        }

    graph.add_node("update", update)
    graph.add_edge(START, "update")
    graph.add_edge("update", END)
    output = await graph.compile().ainvoke(
        AgentState(
            **ctx.identity.model_dump(),
            messages=[
                AIMessage(content="old", id="ai"),
                ToolMessage(content="result", tool_call_id="call", id="tool"),
            ],
        )
    )
    assert [item.id for item in output["messages"]] == ["ai", "tool"]
    assert output["messages"][0].tool_calls[0]["args"] == {"value": 42}
    assert output["messages"][1].tool_call_id == "call"


async def test_parent_failure_delta_preserves_prior_failures_exactly_once() -> None:
    ctx = context()
    graph = StateGraph(AgentState)

    def fail(state: AgentState) -> object:
        return failed("new", state, ConflictError())

    graph.add_node("fail", fail, destinations=(END,))
    graph.add_edge(START, "fail")
    output = await graph.compile().ainvoke(
        AgentState(**ctx.identity.model_dump(), failures=[failure("prior")])
    )
    assert [item.node for item in output["failures"]] == ["prior", "new"]


@pytest.mark.parametrize("field", ["question", "messages", "route", "context", "failures"])
def test_graph_input_cannot_override_loaded_state(field: str) -> None:
    with pytest.raises(ValidationError):
        GraphInput.model_validate({**context().identity.model_dump(), field: None})
    assert GRAPH_VERSION == "phase4-v4"


async def test_router_preserves_reserved_context_during_budgeting() -> None:
    preferences = RoutingContext(
        summary="history",
        recent_messages=[HistoryMessage(role="user", content="hello")],
        terminology=[TerminologyContent(term="大促", means="618")],
        format_preference=FormatPreferenceContent(prefer="table", decimals=2),
    )
    before = preferences.model_dump_json()
    ctx = runtime([decision()])
    await route_question(RouterInput(question=BOTH_QUESTION, routing_context=preferences), ctx)
    sent = json.loads(ctx.llm.calls[0].messages[-1].content)["routing_context"]
    assert sent["terminology"] == [{"term": "大促", "means": "618"}]
    assert sent["format_preference"] == {"prefer": "table", "decimals": 2}
    assert preferences.model_dump_json() == before
