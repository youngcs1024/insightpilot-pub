"""Finalize completed PostgreSQL checkpoint evidence without resuming analysis."""

import structlog

from app.agents.contracts import Route
from app.agents.degradation import deadline_synthesis
from app.agents.failures import FailureKind, NodeFailure
from app.agents.nodes.format_answer import format_preference
from app.agents.runtime import RuntimeContext
from app.agents.state import AgentState, GraphOutput
from app.agents.synthesis_answer import synthesis_answer
from app.core.errors import ConflictError, DeadlineExceededError

logger = structlog.get_logger(__name__)


async def finalize_deadline(state: AgentState, ctx: RuntimeContext) -> GraphOutput:
    """Use the existing commit barrier; persistence failure must never become an answer."""
    ctx.finalization_deadline.check("deadline_finalization")
    if state.route is None or state.route.route is not Route.BOTH:
        raise DeadlineExceededError()
    if any(
        failure.node in {"persist_evidence", "synthesize", "format_answer"}
        and failure.kind is not FailureKind.DEADLINE_EXCEEDED
        for failure in state.failures
    ):
        raise ConflictError("failed answer barrier cannot be bypassed")
    previous = await ctx.evidence.read_bundle(ctx.identity)
    if state.evidence_refs is not None and state.evidence_refs != previous.refs:
        raise ConflictError("checkpoint references differ from committed evidence")
    data = previous.data.data if previous.data else state.data_evidence
    knowledge = previous.knowledge.knowledge if previous.knowledge else state.knowledge_evidence
    if data is None and (knowledge is None or not knowledge.chunks):
        raise DeadlineExceededError()
    bundle = await ctx.evidence.commit_bundle(ctx.identity, data, knowledge)
    result = deadline_synthesis(bundle, state.failures)
    answer = synthesis_answer(
        result,
        bundle,
        [*state.degraded_components, "deadline"],
        format_preference(state),
        trace_id=ctx.trace_id,
    )
    ctx.finalization_deadline.check("deadline_answer_ready")
    logger.info(
        "deadline_partial_ready",
        turn_id=str(ctx.identity.turn_id),
        has_data=bundle.data is not None,
        has_knowledge=bundle.knowledge is not None,
    )
    return GraphOutput(
        route=state.route,
        answer=answer,
        evidence_refs=bundle.refs,
        data_evidence=data,
        failures=[
            *state.failures,
            NodeFailure(
                node="deadline_finalization",
                kind=FailureKind.DEADLINE_EXCEEDED,
                detail="Analysis deadline exceeded.",
                retryable=False,
            ),
        ],
        status="degraded",
    )
