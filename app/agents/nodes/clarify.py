"""A terminal request for information, finalized by the shared formatter."""

from langgraph.types import Command

from app.agents.nodes.prefilter import CLARIFICATION_QUESTION
from app.agents.state import AgentState
from app.schemas.metric_resolution import ClarificationKind, MetricClarification


def clarify(state: AgentState) -> Command[str]:
    """Preserve the router's bounded question without inventing evidence."""
    question = state.route.clarification_question if state.route else ""
    return Command(
        update={
            "route_clarification": MetricClarification(
                kind=ClarificationKind.REFERENCE_UNRESOLVED,
                message=question or CLARIFICATION_QUESTION,
            )
        }
    )
