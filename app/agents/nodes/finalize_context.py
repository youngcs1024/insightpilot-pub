"""Freeze route-scoped context without selecting long-term memories."""

from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.contracts import Route
from app.agents.nodes.common import failed
from app.agents.runtime import RuntimeContext
from app.agents.state import AgentState, TurnContext
from app.core.errors import ConflictError, InsightPilotError
from app.services.knowledge_time import needs_history, parse_time


async def finalize_context(state: AgentState, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """Keep unresolved follow-up scope explicit for the existing knowledge resolver."""
    ctx = runtime.context
    try:
        ctx.deadline.check("finalize_context")
        if state.route is None or state.prepared is None or state.routing_context is None:
            raise ConflictError("missing routing context")
        scope = None
        if state.route.route in {Route.KNOWLEDGE_ONLY, Route.BOTH}:
            parsed = parse_time(state.question, now=ctx.now)
            # A follow-up may inherit an older year/scope after antecedent selection.
            # Never pin it to today's date before the knowledge query resolver runs.
            if not needs_history(state.question) and parsed.clarification is None:
                scope = parsed.scope
        context = TurnContext(
            recent_messages=list(state.messages),
            summary=state.routing_context.summary,
            time_scope=scope,
            prior_sql=list(state.prepared.prior_sql[-3:]),
            knowledge_history=[turn.model_copy(deep=True)
                               for turn in state.prepared.knowledge_history],
            format_preference=state.routing_context.format_preference,
        )
        return Command(update={"context": context})
    except InsightPilotError as exc:
        return failed("finalize_context", state, exc)
