"""Commit evidence through the injected service before entering the formatter."""

from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.contracts import EvidenceRefs
from app.agents.nodes.common import failed
from app.agents.runtime import RuntimeContext
from app.agents.state import AgentState
from app.core.errors import ConflictError, InsightPilotError


async def persist_evidence(state: AgentState, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """Identical recovery writes reuse the committed immutable snapshot ID."""
    ctx = runtime.context
    try:
        ctx.deadline.check("persist_evidence")
        if state.data_evidence is None:
            raise ConflictError("missing data evidence")
        snapshot = await ctx.evidence.commit(ctx.identity, state.data_evidence)
        return Command(
            update={"evidence_refs": EvidenceRefs(data_snapshot_id=snapshot.id)},
            goto="format_answer",
        )
    except InsightPilotError as exc:
        return failed("persist_evidence", state, exc)
