"""Resolve current explicit time before the separate history rewrite operation."""

from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.knowledge.state import KnowledgeAgentState
from app.agents.runtime import RuntimeContext
from app.schemas.retrieval import PointTimeScope
from app.services.knowledge_calendar import same_scope, year_of
from app.services.knowledge_time import needs_history, parse_time, time_clarification
from app.services.periods import BUSINESS_TZ


async def resolve_time_scope(state: KnowledgeAgentState, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """Preserve supplied scope unless explicit current text contradicts it."""
    ctx = runtime.context
    ctx.deadline.check("knowledge_time")
    followup = needs_history(state.question) or needs_history(state.knowledge_intent)
    history = state.knowledge_history
    reference = history[-1].time_scope if history and followup else None
    if reference is not None and any(not same_scope(reference, turn.time_scope) for turn in history):
        reference = None
    years = {year_of(turn.time_scope) for turn in history} if followup else set()
    year = next(iter(years)) if len(years) == 1 else None
    parsed = parse_time(state.question, now=ctx.now, reference=reference, reference_year=year)
    parsed.needs_history = followup
    if parsed.clarification is None and state.time_scope is not None:
        if parsed.scope is not None and not same_scope(parsed.scope, state.time_scope):
            parsed.clarification = time_clarification()
        else:
            parsed.scope = state.time_scope.model_copy(deep=True)
    if parsed.scope is None and parsed.clarification is None and not followup:
        if state.knowledge_intent.strip():
            parsed = parse_time(state.knowledge_intent, now=ctx.now)
        if parsed.scope is None and parsed.clarification is None:
            day = ctx.now.astimezone(BUSINESS_TZ).date()
            parsed.scope = PointTimeScope(as_of=day)
            parsed.assumptions.append(f"未指定日期，按 Asia/Shanghai 的 {day.isoformat()} 查询。")
    return Command(update={"time_resolution": parsed}, goto="rewrite_query")
