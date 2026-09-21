"""Source-labelled assembly only; cross-evidence reasoning belongs to Step 4.6."""

from typing import Literal

from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.contracts import SourceSummary
from app.agents.nodes.common import failed
from app.agents.runtime import RuntimeContext
from app.agents.state import AgentState
from app.core.errors import ConflictError, InsightPilotError


async def synthesize(state: AgentState, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """Read back committed references before enabling source-labelled rendering."""
    try:
        ctx = runtime.context
        ctx.deadline.check("synthesize")
        bundle = await ctx.evidence.read_bundle(ctx.identity)
        if bundle.refs != state.evidence_refs:
            raise ConflictError("uncommitted synthesis evidence")
        missing: list[Literal["data", "knowledge"]] = []
        if bundle.data is None:
            missing.append("data")
        if bundle.knowledge is None:
            missing.append("knowledge")
        return Command(
            update={
                "source_summary": SourceSummary(
                    evidence_refs=bundle.refs, missing_components=missing
                )
            },
            goto="format_answer",
        )
    except InsightPilotError as exc:
        return failed("synthesize", state, exc)
