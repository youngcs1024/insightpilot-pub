"""A bounded terminal clarification, finalized by the shared formatter."""

import structlog
from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.clarification_policy import render_clarification
from app.agents.nodes.common import failed
from app.agents.runtime import RuntimeContext
from app.agents.state import AgentState
from app.core.errors import ConflictError, InsightPilotError

logger = structlog.get_logger(__name__)


async def clarify(state: AgentState, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """Call injected reference services; never invoke a model or an analytical source."""
    ctx = runtime.context
    try:
        ctx.deadline.check("clarify")
        if state.failures or state.data_evidence or state.knowledge_evidence:
            raise ConflictError("clarification contains analytical state")
        if ctx.clarification_capabilities is None:
            raise ConflictError("missing clarification capability service")
        capabilities = await ctx.clarification_capabilities.read(deadline=ctx.deadline)
        ctx.deadline.check("clarify_complete")
        result = render_clarification(state, capabilities, now=ctx.now)
        logger.info(
            "clarification_prepared",
            category=result.category.value if result.category else None,
            loop_prevented=result.loop_prevented,
        )
        return Command(update={"route_clarification": result}, goto="format_answer")
    except InsightPilotError as exc:
        return failed("clarify", state, exc)
