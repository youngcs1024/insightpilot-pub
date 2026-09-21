"""Compiled state-contract fixtures; production route assembly remains Step 4.4."""

import asyncio
from datetime import date
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.contracts import Route
from app.agents.failures import FailureKind, NodeFailure
from app.agents.knowledge.nodes.package_evidence import package_evidence
from app.agents.nodes.prepare import prepare
from app.agents.nodes.router import router
from app.agents.runtime import RuntimeContext
from app.agents.state import AgentState, GraphInput, GraphOutput, TurnContext
from app.agents.summarize import package_result
from app.schemas.memory import Memory, MemoryType, TerminologyContent
from app.schemas.retrieval import PointTimeScope
from tests.agents.knowledge_support import ranked
from tests.agents.support import result
from tests.router_support import BOTH_QUESTION


def memory() -> Memory:
    return Memory(
        id=uuid4(),
        user_id=uuid4(),
        source_turn_id=uuid4(),
        memory_type=MemoryType.TERMINOLOGY,
        content=TerminologyContent(term="大促", means="618"),
        summary="Campaign terminology",
        confidence=1,
    )


def failure(node: str) -> NodeFailure:
    return NodeFailure(
        node=node, kind=FailureKind.NODE_OPERATION_FAILED, detail="unavailable", retryable=False
    )


def finalized() -> TurnContext:
    return TurnContext(time_scope=PointTimeScope(as_of=date(2026, 8, 1)))


async def _prepared(state: AgentState, runtime: Runtime[RuntimeContext]) -> Command[str]:
    command = await prepare(state, runtime)
    # Keep the reducer contract graph's scripted classification independent of
    # the production smoke fixture's high-precision DATA_ONLY question.
    update = dict(command.update)
    update["question"] = BOTH_QUESTION
    return Command(update=update, goto="router")


def _finalize(state: AgentState) -> dict[str, TurnContext]:
    assert state.context is None
    assert state.route is not None
    return {"context": finalized().model_copy(update={"summary": state.route.route.value})}


def _dispatch(state: AgentState) -> list[str]:
    match state.route.route:
        case Route.DATA_ONLY:
            return ["data_agent"]
        case Route.KNOWLEDGE_ONLY:
            return ["knowledge_agent"]
        case Route.BOTH:
            return ["data_agent", "knowledge_agent"]
        case Route.CLARIFY:
            return ["finish"]


def _finish(state: AgentState) -> dict[str, str]:
    return {"status": "failed" if state.failures else "succeeded"}


def contract_topology(
    *, failed_nodes: frozenset[str] = frozenset(), collide: bool = False
) -> StateGraph[AgentState, RuntimeContext, GraphInput, GraphOutput]:
    """Exercise actual reducers, preparation and router with synthetic specialists."""
    barrier = asyncio.Barrier(2)

    async def contribution(state: AgentState, node: str) -> dict[str, object]:
        assert state.context.summary == state.route.route.value
        if state.route.route is Route.BOTH:
            await asyncio.wait_for(barrier.wait(), timeout=5)
        update: dict[str, object] = {"assumptions": [node]}
        if node in failed_nodes:
            update.update(failures=[failure(node)], degraded_components=[node])
        if collide:
            update["question"] = node
        return update

    async def data(state: AgentState) -> dict[str, object]:
        update = await contribution(state, "data_agent")
        if "data_agent" not in failed_nodes:
            update["data_evidence"] = package_result(result(), ["data assumption"])
        return update

    async def knowledge(state: AgentState, runtime: Runtime[RuntimeContext]) -> dict[str, object]:
        update = await contribution(state, "knowledge_agent")
        if "knowledge_agent" not in failed_nodes:
            update["knowledge_evidence"] = package_evidence(ranked(), runtime.context)
        return update

    graph = StateGraph(
        AgentState,
        context_schema=RuntimeContext,
        input_schema=GraphInput,
        output_schema=GraphOutput,
    )
    graph.add_node("prepare", _prepared, destinations=("router",))
    graph.add_node("router", router)
    graph.add_node("finalize_context", _finalize)
    graph.add_node("data_agent", data)
    graph.add_node("knowledge_agent", knowledge)
    graph.add_node("finish", _finish)
    graph.add_edge(START, "prepare")
    graph.add_edge("router", "finalize_context")
    graph.add_conditional_edges(
        "finalize_context", _dispatch, ["data_agent", "knowledge_agent", "finish"]
    )
    graph.add_edge("data_agent", "finish")
    graph.add_edge("knowledge_agent", "finish")
    graph.add_edge("finish", END)
    return graph
