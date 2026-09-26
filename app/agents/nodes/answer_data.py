"""Explicit parent/data projection with immutable evidence reuse."""

import asyncio

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_config
from langgraph.graph import END
from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.data.graph import build
from app.agents.data.state import DataAgentOutput
from app.agents.nodes.common import failed, specialist_failed
from app.agents.projections import to_data_input, to_legacy_data_input
from app.agents.runtime import RuntimeContext
from app.agents.state import AgentState
from app.core.errors import ConflictError, InsightPilotError

DATA_GRAPH = build()


async def _invoke(
    state: AgentState, ctx: RuntimeContext, config: RunnableConfig, *, routed: bool
) -> DataAgentOutput:
    ctx.deadline.check("answer_data")
    previous = await ctx.evidence.find(ctx.identity)
    if previous is not None:
        return DataAgentOutput(evidence=previous.data)
    if routed and state.context is not None and state.context.intent_failure is not None:
        return DataAgentOutput(failure=state.context.intent_failure)
    project = to_data_input if routed else to_legacy_data_input
    inputs = project(state, token_counter=ctx.schema_token_counter)
    return DataAgentOutput.model_validate(await DATA_GRAPH.ainvoke(inputs, config, context=ctx))


async def answer_data(
    state: AgentState, runtime: Runtime[RuntimeContext], config: RunnableConfig
) -> Command[str]:
    """Retain the legacy production entrypoint until Step 4.4 wires routing."""
    try:
        output = await _invoke(state, runtime.context, config, routed=False)
        if output.clarification is not None:
            return Command(
                update={"clarification": output.clarification, "status": "succeeded"}, goto=END
            )
        if output.failure is not None:
            return Command(update={"failures": [output.failure], "status": "failed"}, goto=END)
        if output.evidence is None:
            raise ConflictError("missing data specialist output")
        return Command(update={"data_evidence": output.evidence}, goto="persist_evidence")
    except InsightPilotError as exc:
        return failed("answer_data", state, exc)


async def answer_data_routed(
    state: AgentState, runtime: Runtime[RuntimeContext], config: RunnableConfig
) -> Command[str]:
    """Write only data-owned fields and append-only deltas, without parent routing."""
    budget = asyncio.timeout(runtime.context.deadline.remaining())
    try:
        async with budget:
            output = await _invoke(state, runtime.context, config, routed=True)
            runtime.context.deadline.check("answer_data_complete")
        return Command(
            update={
                "data_evidence": output.evidence,
                "data_clarification": output.clarification,
                "assumptions": list(output.assumptions),
                "failures": [output.failure] if output.failure else [],
            }
        )
    except Exception as exc:
        return specialist_failed("answer_data_routed", exc, deadline_expired=budget.expired())


async def answer_data_node(state: AgentState, *, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """Adapt the legacy graph protocol, forwarding the active checkpoint configuration."""
    return await answer_data(state, runtime, get_config())


async def answer_data_routed_node(
    state: AgentState, *, runtime: Runtime[RuntimeContext]
) -> Command[str]:
    """Expose the strict routed entrypoint to a parent graph."""
    return await answer_data_routed(state, runtime, get_config())
