"""Load owned history via the injected conversation service."""

from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.nodes.common import failed
from app.agents.runtime import RuntimeContext
from app.agents.state import AgentState
from app.core.errors import InsightPilotError


async def prepare(state: AgentState, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """Prepare only the current turn; previous transient state never enters history."""
    ctx = runtime.context
    try:
        ctx.deadline.check("prepare")
        prepared = await ctx.conversations.prepare(ctx.identity)
        return Command(update={"prepared": prepared}, goto="rewrite_question")
    except InsightPilotError as exc:
        return failed("prepare", state, exc)
