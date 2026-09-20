"""Load owned history via the injected conversation service."""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.contracts import RoutingContext
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
        messages = [
            (HumanMessage if item.role == "user" else AIMessage)(
                content=item.content, id=f"{state.turn_id}:history:{index}"
            )
            for index, item in enumerate(prepared.messages)
        ]
        return Command(
            update={
                "prepared": prepared,
                "question": prepared.question,
                "messages": [
                    *messages,
                    HumanMessage(content=prepared.question, id=f"{state.turn_id}:question"),
                ],
                "routing_context": RoutingContext(
                    summary=prepared.summary, recent_messages=prepared.messages
                ),
            },
            goto="rewrite_question",
        )
    except InsightPilotError as exc:
        return failed("prepare", state, exc)
