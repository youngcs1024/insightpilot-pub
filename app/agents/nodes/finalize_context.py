"""Freeze a single gated preference selection after current intent interpretation."""

from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.contracts import Route
from app.agents.memory_context import FinalMemoryContext, finalize_memories
from app.core.observability import TraceMetadata, update_current_observation, mark_degraded
from app.schemas.memory import TerminologyContent
from app.schemas.metric_resolution import MetricPatches
from app.agents.nodes.common import failed
from app.agents.presentation import resolve_preference
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
        selected = FinalMemoryContext()
        if ctx.memories is not None:
            selected = await finalize_memories(state, ctx)
        degraded = selected.selection.failed or state.memory_preselection.failed
        if degraded:
            mark_degraded("memory")
        before = {row.id for row in state.memory_preselection.selected
                  if isinstance(row.content, TerminologyContent)}
        after = {row.id for row in selected.selection.selected
                 if isinstance(row.content, TerminologyContent)}
        if not state.memory_disabled and before - after:
            update_current_observation(TraceMetadata(
                memory_restart="read_failed" if degraded else "selection_changed",
            ))
            return Command(update={
                "memory_disabled": True, "memory_restart_pending": True,
                "degraded_components": ["memory"] if degraded and "memory" not in state.degraded_components else [],
            })
        scope = None
        if state.route.route in {Route.KNOWLEDGE_ONLY, Route.BOTH}:
            parsed = parse_time(state.question, now=ctx.now)
            # A follow-up may inherit an older year/scope after antecedent selection.
            # Never pin it to today's date before the knowledge query resolver runs.
            if not needs_history(state.question) and parsed.clarification is None:
                scope = parsed.scope
        context = TurnContext(
            recent_messages=list(state.messages),
            prepared_intent=selected.intent,
            intent_failure=selected.intent_failure,
            memories=list(selected.selection.selected),
            region_scope=selected.region,
            selected_overrides=selected.overrides,
            explicit_patch=selected.intent.explicit_patch.model_copy(deep=True) if selected.intent else MetricPatches(),
            token_accounting={"memories": selected.selection.tokens},
            summary=state.routing_context.summary,
            time_scope=scope,
            prior_sql=list(state.prepared.prior_sql[-3:]),
            knowledge_history=[
                turn.model_copy(deep=True) for turn in state.prepared.knowledge_history
            ],
            format_preference=resolve_preference(
                state.question, selected.format_preference if ctx.memories is not None else state.routing_context.format_preference
            ),
        )
        return Command(update={
            "context": context, "context_clarification": selected.clarification,
            "memory_restart_pending": False,
            "degraded_components": ["memory"] if degraded and "memory" not in state.degraded_components else [],
        })
    except InsightPilotError as exc:
        return failed("finalize_context", state, exc)
