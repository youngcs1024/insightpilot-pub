"""An honest refusal is a successful terminal outcome, never a service failure."""

from langgraph.types import Command

from app.agents.knowledge.state import KnowledgeAgentState
from app.schemas.knowledge import KnowledgeAbstention


def no_evidence(state: KnowledgeAgentState) -> Command[str]:
    """Keep the reason typed until the terminal projection renders explanatory prose."""
    return Command(
        update={
            "rejection": (
                KnowledgeAbstention.BUDGET_EXHAUSTED
                if state.packaged is not None
                else KnowledgeAbstention.NO_EVIDENCE
            )
        },
        goto="finish_knowledge",
    )
