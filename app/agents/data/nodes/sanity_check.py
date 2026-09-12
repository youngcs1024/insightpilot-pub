"""Advisory node for Step 2.11 assembly; no I/O or correction routing."""

from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.data.sanity import check_result
from app.agents.data.state import DataAgentState
from app.agents.runtime import RuntimeContext


async def sanity_check(state: DataAgentState, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """Write only observations, preserving SQL, bindings and the successful result."""
    outcome = check_result(state.query_result, runtime.context.settings.data_agent.sanity)
    return Command(update={"sanity_check_result": outcome})
