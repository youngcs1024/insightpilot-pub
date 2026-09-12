"""Resolve parent-only conversation references before the data specialist."""

import json

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.budget import SUMMARY_TOKENS, bounded_text
from app.agents.contracts import RewrittenQuestion
from app.agents.multiturn import trim_history
from app.agents.nodes.common import failed
from app.agents.prompts import REWRITE_QUESTION
from app.agents.runtime import RuntimeContext
from app.agents.state import AgentState
from app.core.errors import ConflictError, InsightPilotError
from app.core.llm_config import ModelRole
from app.core.observability import TraceMetadata, update_current_observation
from app.schemas.metric_resolution import ClarificationKind, MetricClarification

logger = structlog.get_logger(__name__)


async def rewrite_question(state: AgentState, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """Store interpretation in the checkpoint; export only safe diagnostics."""
    ctx = runtime.context
    try:
        ctx.deadline.check("rewrite_question")
        prepared = state.prepared
        if prepared is None:
            raise ConflictError("missing prepared context")
        if not prepared.has_prior_turns:
            return Command(goto="answer_data")
        if await ctx.evidence.find(ctx.identity) is not None:
            return Command(goto="answer_data")
        rewritten = await ctx.llm.generate_structured(
            ModelRole.ROUTER,
            [
                SystemMessage(content=REWRITE_QUESTION),
                HumanMessage(
                    content=json.dumps(
                        {
                            "question": prepared.question,
                            "summary": bounded_text(prepared.summary, SUMMARY_TOKENS),
                            "history": [
                                item.model_dump() for item in trim_history(prepared.messages)
                            ],
                        },
                        ensure_ascii=False,
                    )
                ),
            ],
            RewrittenQuestion,
            deadline=ctx.deadline,
        )
        ctx.deadline.check("rewrite_question_complete")
        update_current_observation(
            TraceMetadata(
                turn_id=str(ctx.identity.turn_id),
                referenced_prior_turn=rewritten.referenced_prior_turn,
                unresolved_reference_count=len(rewritten.unresolved_references),
            )
        )
        logger.info(
            "question_rewritten",
            turn_id=str(ctx.identity.turn_id),
            referenced_prior_turn=rewritten.referenced_prior_turn,
            unresolved_reference_count=len(rewritten.unresolved_references),
        )
        if rewritten.unresolved_references:
            return Command(
                update={
                    "rewritten": rewritten,
                    "clarification": MetricClarification(
                        kind=ClarificationKind.REFERENCE_UNRESOLVED,
                        message="请补充所指的问题、指标、时间或地区，以便明确本次查询。",
                    ),
                    "status": "succeeded",
                },
                goto=END,
            )
        return Command(update={"rewritten": rewritten}, goto="answer_data")
    except InsightPilotError as exc:
        return failed("rewrite_question", state, exc)
