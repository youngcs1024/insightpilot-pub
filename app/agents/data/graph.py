"""Isolated data specialist; the caller's PostgreSQL checkpointer is inherited."""

from collections.abc import Awaitable, Callable
from typing import Protocol

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.data.nodes.correct_sql import correct_sql, correction_route
from app.agents.data.nodes.generate_sql import generate_sql
from app.agents.data.nodes.lifecycle import (
    execute_sql,
    package_evidence,
    package_failure,
    validate_sql,
)
from app.agents.data.nodes.native_tools import native_tools
from app.agents.data.nodes.resolve_metrics import resolve_metrics
from app.agents.data.nodes.sanity_check import sanity_check
from app.agents.data.nodes.select_schema import select_schema
from app.agents.data.state import DataAgentInput, DataAgentOutput, DataAgentState
from app.agents.nodes.common import node_failure
from app.agents.runtime import RuntimeContext
from app.core.errors import InsightPilotError, SqlExecutionError
from app.schemas.mcp import SqlErrorKind
from app.schemas.sql_correction import CorrectionRoute, CorrectionStatus

logger = structlog.get_logger(__name__)
RECURSION_LIMIT = 32
type DataGraph = CompiledStateGraph[DataAgentState, RuntimeContext, DataAgentInput, DataAgentOutput]
type AsyncNode = Callable[[DataAgentState, Runtime[RuntimeContext]], Awaitable[Command[str]]]


class RuntimeNode(Protocol):
    """Keep the injectable runtime keyword visible to LangGraph's node protocol."""

    def __call__(
        self, state: DataAgentState, *, runtime: Runtime[RuntimeContext]
    ) -> Awaitable[Command[str]]:
        """Invoke with the caller's runtime."""
        ...


def guarded(name: str, node: AsyncNode, destination: str) -> RuntimeNode:
    """Centralize typed operational failures without adding another retry layer."""

    async def invoke(state: DataAgentState, *, runtime: Runtime[RuntimeContext]) -> Command[str]:
        try:
            result = await node(state, runtime)
        except InsightPilotError as exc:
            logger.exception("data_node_failed", node=name, code=exc.code, exc_info=False)
            failure = node_failure(name, exc)
            if isinstance(exc, SqlExecutionError):
                failure.detail = exc.kind.value
            correctable = (
                isinstance(exc, SqlExecutionError)
                and exc.kind is not SqlErrorKind.OTHER
                and name == "execute_sql"
            )
            return Command(
                update={
                    "failures": [*state.failures, failure],
                    "correction_status": (
                        CorrectionStatus.IDLE if correctable else CorrectionStatus.TERMINAL
                    ),
                },
                goto="route_correction" if correctable else "package_failure",
            )
        return Command(update=result.update, goto=result.goto or destination)

    return invoke


async def preflight(state: DataAgentState, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """Adapt pure policy validation to the same typed failure wrapper."""
    return validate_sql(state, runtime)


async def evidence(state: DataAgentState, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """Adapt pure evidence packaging to the same typed failure wrapper."""
    return package_evidence(state, runtime)


async def metrics(state: DataAgentState, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """Never generate SQL from a request needing clarification."""
    result = await resolve_metrics(state, runtime)
    update = result.update if isinstance(result.update, dict) else {}
    updated = DataAgentState.model_validate({**state.model_dump(), **update})
    destination = (
        "package_failure"
        if updated.clarification is not None
        else ("native_tools" if updated.native_tool_kinds else "generate_sql")
    )
    return Command(update=result.update, goto=destination)


def route_correction(state: DataAgentState) -> Command[str]:
    """Use only the existing typed correction lifecycle to select a destination."""
    route = correction_route(state)
    destination = "package_failure" if route is CorrectionRoute.NO_CORRECTION else route.value
    return Command(goto=destination)


def topology() -> StateGraph[DataAgentState, RuntimeContext, DataAgentInput, DataAgentOutput]:
    """Every normal or failed path terminates at one of the two packagers."""
    graph = StateGraph(
        DataAgentState,
        context_schema=RuntimeContext,
        input_schema=DataAgentInput,
        output_schema=DataAgentOutput,
    )
    graph.add_node(
        "select_schema",
        guarded("select_schema", select_schema, "resolve_metrics"),
        destinations=("resolve_metrics", "package_failure"),
    )
    graph.add_node(
        "resolve_metrics",
        guarded("resolve_metrics", metrics, "generate_sql"),
        destinations=("native_tools", "generate_sql", "package_failure"),
    )
    graph.add_node(
        "native_tools",
        guarded("native_tools", native_tools, "generate_sql"),
        destinations=("generate_sql", "package_failure"),
    )
    graph.add_node(
        "generate_sql",
        guarded("generate_sql", generate_sql, "validate_sql"),
        destinations=("validate_sql", "package_failure"),
    )
    graph.add_node(
        "validate_sql",
        guarded("validate_sql", preflight, "execute_sql"),
        destinations=("execute_sql", "route_correction", "package_failure"),
    )
    graph.add_node(
        "execute_sql",
        guarded("execute_sql", execute_sql, "sanity_check"),
        destinations=("sanity_check", "route_correction", "package_failure"),
    )
    graph.add_node(
        "sanity_check",
        guarded("sanity_check", sanity_check, "package_evidence"),
        destinations=("package_evidence", "package_failure"),
    )
    graph.add_node(
        "route_correction",
        route_correction,
        destinations=("correct_sql", "validate_sql", "package_failure"),
    )
    graph.add_node(
        "correct_sql",
        guarded("correct_sql", correct_sql, "route_correction"),
        destinations=("route_correction", "package_failure"),
    )
    graph.add_node(
        "package_evidence",
        guarded("package_evidence", evidence, END),
        destinations=(END, "package_failure"),
    )
    graph.add_node("package_failure", package_failure)
    graph.add_edge("package_failure", END)
    graph.add_edge(START, "select_schema")
    return graph


def build() -> DataGraph:
    """Inherit parent checkpoint context; never allocate an in-memory saver."""
    return topology().compile(name="data_specialist_graph")
