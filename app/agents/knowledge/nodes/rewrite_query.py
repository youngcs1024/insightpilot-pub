"""Resolve only necessary follow-up references, preserving calendar and query provenance."""

import json

import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.knowledge.state import KnowledgeAgentState
from app.agents.budget import SUMMARY_TOKENS, bounded_text
from app.agents.knowledge.query_context import antecedents, bounded_history, shared_scope
from app.agents.prompts import KNOWLEDGE_REWRITE
from app.agents.runtime import RuntimeContext
from app.core.errors import KnowledgeEvidenceError
from app.core.llm_config import ModelRole
from app.core.observability import TraceMetadata, update_current_observation
from app.schemas.knowledge_query import KnowledgeHistoryTurn, KnowledgeRewrite, KnowledgeTimeResolution
from app.schemas.retrieval import PointTimeScope, RetrievalQuery
from app.services.knowledge_calendar import same_scope, year_of
from app.services.knowledge_time import parse_time, reference_clarification, time_clarification
from app.services.periods import BUSINESS_TZ

logger = structlog.get_logger(__name__)


def _final_time(state: KnowledgeAgentState, ctx: RuntimeContext, selected: list[KnowledgeHistoryTurn]) -> KnowledgeTimeResolution:
    initial = state.time_resolution
    if initial is None:
        raise KnowledgeEvidenceError(reason="knowledge_time_missing")
    if initial.scope is not None and (not selected or not initial.year_inferred):
        return initial.model_copy(deep=True)
    reference = shared_scope(selected)
    if initial.year_inferred and reference is None:
        return KnowledgeTimeResolution(clarification=reference_clarification())
    resolved = parse_time(state.question, now=ctx.now, reference=reference,
                          reference_year=year_of(reference) if reference else None)
    if resolved.clarification is not None:
        return resolved
    if state.time_scope is not None:
        if resolved.scope is not None and not same_scope(resolved.scope, state.time_scope):
            resolved.clarification = time_clarification()
        else:
            resolved.scope = state.time_scope.model_copy(deep=True)
        return resolved
    if resolved.scope is not None:
        return resolved
    if selected:
        if reference is None:
            return KnowledgeTimeResolution(clarification=reference_clarification())
        resolved.scope = reference
        resolved.assumptions.append("当前追问未指定新的时间，沿用所指上文的政策适用时间（Asia/Shanghai）。")
        return resolved
    day = ctx.now.astimezone(BUSINESS_TZ).date()
    resolved.scope = PointTimeScope(as_of=day)
    resolved.assumptions.append(f"未指定日期，按 Asia/Shanghai 的 {day.isoformat()} 查询。")
    return resolved


def _query(state: KnowledgeAgentState, ctx: RuntimeContext, standalone: str,
           selected: list[KnowledgeHistoryTurn], rewritten: KnowledgeRewrite | None = None) -> Command[str]:
    resolved = _final_time(state, ctx, selected)
    if resolved.clarification is not None:
        return Command(update={"query_clarification": resolved.clarification, "rewritten": rewritten}, goto="finish_knowledge")
    scope = resolved.scope
    if scope is None:
        raise KnowledgeEvidenceError(reason="knowledge_time_missing")
    # Model-produced dates and caller-scoped intent cannot override original constraints.
    checked = parse_time(standalone, now=ctx.now, reference=shared_scope(selected), reference_year=year_of(scope))
    if checked.clarification is not None or (checked.scope is not None and not same_scope(checked.scope, scope)):
        return Command(update={"query_clarification": time_clarification(), "rewritten": rewritten}, goto="finish_knowledge")
    return Command(update={"rewritten": rewritten, "query": RetrievalQuery(
        standalone=standalone, original_question=state.question, time_scope=scope,
        assumptions=resolved.assumptions,
    )})


async def rewrite_query(
    state: KnowledgeAgentState, runtime: Runtime[RuntimeContext]
) -> Command[str]:
    """One logical LLM call only when needed; transport retries remain service-owned."""
    ctx = runtime.context
    ctx.deadline.check("knowledge_query")
    resolution = state.time_resolution
    if resolution is None:
        raise KnowledgeEvidenceError(reason="knowledge_time_missing")
    if resolution.clarification is not None:
        return Command(update={"query_clarification": resolution.clarification}, goto="finish_knowledge")
    original = state.knowledge_intent if state.knowledge_intent.strip() else state.question
    if not resolution.needs_history:
        logger.info("knowledge_rewrite_skipped")
        return _query(state, ctx, original, [])
    history = bounded_history(state.knowledge_history, ctx.schema_token_counter)
    if not history:
        return Command(update={"query_clarification": reference_clarification()}, goto="finish_knowledge")
    rewritten = await ctx.llm.generate_structured(
        ModelRole.ROUTER,
        [SystemMessage(content=KNOWLEDGE_REWRITE), HumanMessage(content=json.dumps({
            "question": state.question,
            "knowledge_intent": state.knowledge_intent,
            "history": [turn.model_dump(mode="json") for turn in history],
            "summary": bounded_text(state.conversation_summary, SUMMARY_TOKENS),
            "terminology": [item.model_dump() for item in state.relevant_memories],
            "explicit_time": resolution.scope.model_dump(mode="json") if resolution.scope else None,
        }, ensure_ascii=False))],
        KnowledgeRewrite, deadline=ctx.deadline,
    )
    ctx.deadline.check("knowledge_query_rewritten")
    selected = antecedents(rewritten, history)
    update_current_observation(TraceMetadata(referenced_prior_turn=bool(selected),
        unresolved_reference_count=len(rewritten.unresolved_references)))
    logger.info("knowledge_query_rewritten", referenced_prior_turn=bool(selected),
                unresolved_reference_count=len(rewritten.unresolved_references))
    if not selected:
        return Command(update={"rewritten": rewritten, "query_clarification": reference_clarification()}, goto="finish_knowledge")
    return _query(state, ctx, rewritten.standalone, selected, rewritten)
