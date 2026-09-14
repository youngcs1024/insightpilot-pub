"""Isolated knowledge graph around the existing bounded retrieval service."""

from collections.abc import Awaitable, Callable
from typing import Literal, Protocol

import structlog
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.knowledge.nodes.no_evidence import no_evidence
from app.agents.knowledge.nodes.package_evidence import package_evidence
from app.agents.knowledge.nodes.retrieve import retrieve
from app.agents.knowledge.nodes.rewrite_query import rewrite_query
from app.agents.knowledge.state import (
    KnowledgeAgentInput,
    KnowledgeAgentOutput,
    KnowledgeAgentState,
)
from app.agents.nodes.common import node_failure
from app.agents.runtime import RuntimeContext
from app.core.errors import DeadlineExceededError, InsightPilotError, KnowledgeEvidenceError
from app.schemas.knowledge import KnowledgeAbstention

logger = structlog.get_logger(__name__)
RECURSION_LIMIT = 32
type KnowledgeGraph = CompiledStateGraph[
    KnowledgeAgentState, RuntimeContext, KnowledgeAgentInput, KnowledgeAgentOutput
]
type AsyncNode = Callable[[KnowledgeAgentState, Runtime[RuntimeContext]], Awaitable[Command[str]]]


class RuntimeNode(Protocol):
    """Expose the runtime keyword required by LangGraph dependency injection."""

    def __call__(
        self, state: KnowledgeAgentState, *, runtime: Runtime[RuntimeContext]
    ) -> Awaitable[Command[str]]: ...


def guarded(name: str, node: AsyncNode, destination: str) -> RuntimeNode:
    """Translate operational errors without retrying or swallowing cancellation."""

    async def invoke(
        state: KnowledgeAgentState, *, runtime: Runtime[RuntimeContext]
    ) -> Command[str]:
        try:
            result = await node(state, runtime)
        except InsightPilotError as exc:
            logger.exception("knowledge_node_failed", node=name, code=exc.code, exc_info=False)
            failure = node_failure(
                name, DeadlineExceededError() if runtime.context.deadline.remaining() <= 0 else exc
            )
            return Command(update={"operation_failure": failure}, goto="finish_knowledge")
        return Command(update=result.update, goto=result.goto or destination)

    return invoke


async def package(state: KnowledgeAgentState, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """The existing packager may legitimately omit every chunk because of budget."""
    if state.retrieval_result is None:
        raise KnowledgeEvidenceError(reason="knowledge_result_missing")
    evidence = package_evidence(state.retrieval_result, runtime.context)
    return Command(
        update={"packaged": evidence},
        goto="finish_knowledge" if evidence.chunks else "no_evidence",
    )


def refusal_reason(state: KnowledgeAgentState) -> str:
    """Do not substitute an RRF score or zero for an unmeasured rerank score."""
    result = state.retrieval_result
    if result is None:
        raise KnowledgeEvidenceError(reason="knowledge_refusal_missing_result")
    floor = result.retrieval_config.filtering.absolute_floor
    best = "未测得" if result.top_rerank_score is None else f"{result.top_rerank_score:g}"
    reason = (
        "证据 token 预算不足，无法保留支持材料。"
        if state.rejection is KnowledgeAbstention.BUDGET_EXHAUSTED
        else "未找到满足要求的支持证据。"
    )
    return f"{reason}重排相关性阈值={floor:g}；最高重排分数={best}。"


def finish(state: KnowledgeAgentState) -> Command[str]:
    """One owner validates and writes every public terminal field."""
    degraded: list[Literal["rerank"]] = (
        ["rerank"]
        if state.retrieval_result is not None and state.retrieval_result.degradation is not None
        else []
    )
    if state.operation_failure is not None:
        output = KnowledgeAgentOutput(failure=state.operation_failure, degraded_components=degraded)
        status = "failed"
    elif state.rejection is not None:
        output = KnowledgeAgentOutput(
            abstained=True, abstention_reason=refusal_reason(state), degraded_components=degraded
        )
        status = "abstained"
    else:
        output = KnowledgeAgentOutput(evidence=state.packaged, degraded_components=degraded)
        status = "succeeded"
    logger.info("knowledge_completed", status=status, degraded_components=degraded)
    return Command(
        update={name: getattr(output, name) for name in KnowledgeAgentOutput.model_fields}, goto=END
    )


def topology() -> StateGraph[
    KnowledgeAgentState, RuntimeContext, KnowledgeAgentInput, KnowledgeAgentOutput
]:
    """All paths converge on a validated terminal projection, with no generation fallback."""
    graph = StateGraph(
        KnowledgeAgentState,
        context_schema=RuntimeContext,
        input_schema=KnowledgeAgentInput,
        output_schema=KnowledgeAgentOutput,
    )
    graph.add_node(
        "rewrite_query",
        guarded("rewrite_query", rewrite_query, "retrieve"),
        destinations=("retrieve", "finish_knowledge"),
    )
    graph.add_node(
        "retrieve",
        guarded("retrieve", retrieve, "package_evidence"),
        destinations=("package_evidence", "no_evidence", "finish_knowledge"),
    )
    graph.add_node(
        "package_evidence",
        guarded("package_evidence", package, "finish_knowledge"),
        destinations=("no_evidence", "finish_knowledge"),
    )
    graph.add_node("no_evidence", no_evidence, destinations=("finish_knowledge",))
    graph.add_node("finish_knowledge", finish, destinations=(END,))
    graph.add_edge(START, "rewrite_query")
    return graph


def build() -> KnowledgeGraph:
    """Inherit the parent's durable saver and retain a bounded default invocation."""
    return topology().compile(name="knowledge_agent").with_config(recursion_limit=RECURSION_LIMIT)
