"""Commit both specialist outputs once, before synthesis or formatting."""

from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.contracts import Route
from app.agents.nodes.common import failed
from app.agents.runtime import RuntimeContext
from app.agents.state import AgentState
from app.core.errors import InsightPilotError


async def persist_evidence(state: AgentState, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """No-evidence outcomes retain empty references, never fabricated snapshots."""
    ctx = runtime.context
    try:
        ctx.deadline.check("persist_evidence")
        bundle = await ctx.evidence.commit_bundle(
            ctx.identity, state.data_evidence, state.knowledge_evidence
        )
        has_evidence = bundle.data is not None or bundle.knowledge is not None
        both = state.route is not None and state.route.route is Route.BOTH
        destination = "synthesize" if both and has_evidence else "format_answer"
        if not has_evidence and not state.failures and (
            state.data_clarification is not None or state.knowledge_clarification is not None
        ):
            destination = "clarify"
        return Command(update={"evidence_refs": bundle.refs}, goto=destination)
    except InsightPilotError as exc:
        return failed("persist_evidence", state, exc)
