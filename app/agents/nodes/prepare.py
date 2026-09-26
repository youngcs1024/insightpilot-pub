"""Load owned history via the injected conversation service."""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.contracts import RoutingContext
from app.agents.nodes.common import failed
from app.agents.runtime import RuntimeContext
from app.agents.state import AgentState
from app.core.errors import InsightPilotError
from app.schemas.memory import FormatPreferenceContent, TerminologyContent
from app.schemas.memory_retrieval import MemoryReadRequest, MemorySelection, MemoryStage


async def prepare(state: AgentState, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """Prepare only the current turn; previous transient state never enters history."""
    ctx = runtime.context
    try:
        ctx.deadline.check("prepare")
        prepared = await ctx.conversations.prepare(ctx.identity)
        memories = MemorySelection()
        if ctx.memories is not None and not state.memory_disabled:
            memories = await ctx.memories.retrieve(
                MemoryReadRequest(user_id=ctx.identity.user_id, question=prepared.question,
                                  stage=MemoryStage.PREPARE),
                deadline=ctx.deadline, counter=ctx.schema_token_counter,
            )
        messages = [
            (HumanMessage if item.role == "user" else AIMessage)(
                content=item.content, id=f"{state.turn_id}:history:{index}"
            )
            for index, item in enumerate(prepared.messages)
        ]
        return Command(
            update={
                "prepared": prepared,
                "memory_preselection": memories,
                "degraded_components": ["memory"] if memories.failed and "memory" not in state.degraded_components else [],
                "question": prepared.question,
                "messages": [
                    *messages,
                    HumanMessage(content=prepared.question, id=f"{state.turn_id}:question"),
                ],
                "routing_context": RoutingContext(
                    summary=prepared.summary, recent_messages=prepared.messages,
                    terminology=[row.content.model_copy(deep=True) for row in memories.selected
                                 if isinstance(row.content, TerminologyContent)],
                    format_preference=next((row.content.model_copy(deep=True)
                        for row in memories.selected if isinstance(row.content, FormatPreferenceContent)), None),
                ),
            },
            goto="route",
        )
    except InsightPilotError as exc:
        return failed("prepare", state, exc)
