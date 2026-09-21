"""Knowledge wrapper with isolated terminal channels and no persistence side path."""

from langchain_core.runnables import RunnableConfig
from langgraph.config import get_config
from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.knowledge.graph import build
from app.agents.knowledge.state import KnowledgeAgentOutput
from app.agents.nodes.common import specialist_failed
from app.agents.projections import to_knowledge_input
from app.agents.runtime import RuntimeContext
from app.agents.state import AgentState
from app.core.errors import InsightPilotError

KNOWLEDGE_GRAPH = build()


async def answer_knowledge(
    state: AgentState, runtime: Runtime[RuntimeContext], config: RunnableConfig
) -> Command[str]:
    """Return owned outputs and new deltas; parent dispatch owns the next step."""
    ctx = runtime.context
    try:
        ctx.deadline.check("answer_knowledge")
        if state.knowledge_evidence is not None:
            return Command(update={"knowledge_evidence": state.knowledge_evidence})
        bundle = await ctx.evidence.read_bundle(ctx.identity)
        if bundle.knowledge is not None:
            return Command(update={"knowledge_evidence": bundle.knowledge.knowledge})
        inputs = to_knowledge_input(state, token_counter=ctx.schema_token_counter)
        output = KnowledgeAgentOutput.model_validate(
            await KNOWLEDGE_GRAPH.ainvoke(inputs, config, context=ctx)
        )
        return Command(
            update={
                "knowledge_evidence": output.evidence,
                "knowledge_clarification": output.clarification,
                "knowledge_abstention_reason": output.abstention_reason,
                "assumptions": list(output.assumptions),
                "degraded_components": list(output.degraded_components),
                "failures": [output.failure] if output.failure else [],
            }
        )
    except InsightPilotError as exc:
        return specialist_failed("answer_knowledge", exc)


async def answer_knowledge_node(
    state: AgentState, *, runtime: Runtime[RuntimeContext]
) -> Command[str]:
    """Forward the inherited callback and checkpoint configuration unchanged."""
    return await answer_knowledge(state, runtime, get_config())
