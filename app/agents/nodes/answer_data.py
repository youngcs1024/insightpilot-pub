"""Explicit parent/data projection with immutable evidence reuse."""

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_config
from langgraph.graph import END
from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.data.graph import build
from app.agents.data.state import DataAgentInput, DataAgentOutput
from app.agents.nodes.common import failed
from app.agents.runtime import RuntimeContext
from app.agents.state import AgentState
from app.core.errors import ConflictError, InsightPilotError

DATA_GRAPH = build()


async def answer_data(
    state: AgentState, runtime: Runtime[RuntimeContext], config: RunnableConfig
) -> Command[str]:
    """Project only finalized inputs; preserve callbacks and checkpoint namespace."""
    ctx = runtime.context
    try:
        ctx.deadline.check("answer_data")
        if state.prepared is None:
            raise ConflictError("missing prepared context")
        previous = await ctx.evidence.find(ctx.identity)
        if previous is not None:
            return Command(update={"data_evidence": previous.data}, goto="persist_evidence")
        question = state.rewritten.standalone if state.rewritten else state.prepared.question
        inputs = DataAgentInput(
            question=question, data_intent=question, prior_sql=state.prepared.prior_sql
        )
        output = DataAgentOutput.model_validate(
            await DATA_GRAPH.ainvoke(inputs, config, context=ctx)
        )
        if output.clarification is not None:
            return Command(
                update={"clarification": output.clarification, "status": "succeeded"}, goto=END
            )
        if output.failure is not None:
            return Command(
                update={"failures": [output.failure], "status": "failed"},
                goto=END,
            )
        if output.evidence is None:
            raise ConflictError("missing data specialist output")
        return Command(update={"data_evidence": output.evidence}, goto="persist_evidence")
    except InsightPilotError as exc:
        return failed("answer_data", state, exc)


async def answer_data_node(state: AgentState, *, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """Adapt the graph protocol while forwarding the active checkpoint configuration."""
    return await answer_data(state, runtime, get_config())
