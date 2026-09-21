"""Owned clarification history, independent of the prompt's text window."""

from uuid import UUID

from app.agents.budget import bounded_text
from app.agents.contracts import HistoryMessage
from app.db.models import Turn, TurnRole, TurnStatus
from app.repositories.turns import TurnRepository
from app.schemas.clarification import ClarificationHistory
from app.schemas.metric_resolution import MetricClarification


def is_clarification(turn: Turn) -> bool:
    """Include legacy successful clarifications, but never failures or plain abstentions."""
    return turn.clarification is not None and turn.status in {
        TurnStatus.SUCCEEDED,
        TurnStatus.ABSTAINED,
    }


def history_message(turn: Turn) -> HistoryMessage:
    """Keep a compact, durable suggestion usable by the next router invocation."""
    content = turn.content
    if turn.role is TurnRole.ASSISTANT and is_clarification(turn):
        clarification = MetricClarification.model_validate(turn.clarification)
        if clarification.suggested_question:
            content = "建议（尚未执行，等待确认）: " + clarification.suggested_question
    return HistoryMessage(role=turn.role.value, content=content)


async def load_clarification_history(
    repository: TurnRepository, conversation_id: UUID, before_seq: int
) -> ClarificationHistory:
    """Inspect the immediate two assistant turns; failed turns break the sequence."""
    previous = (
        await repository.session.scalars(
            repository.previous_assistants(conversation_id, before_seq)
        )
    ).all()
    consecutive = 0
    for turn in previous:
        if not is_clarification(turn):
            break
        consecutive += 1
    questions = (
        await repository.session.scalars(repository.recent_topics(conversation_id, before_seq))
    ).all()
    topics = list(
        dict.fromkeys(bounded_text(turn.content, 240) for turn in questions if turn.content.strip())
    )
    prior = (
        MetricClarification.model_validate(previous[0].clarification)
        if previous and is_clarification(previous[0])
        else None
    )
    return ClarificationHistory(
        consecutive=consecutive,
        recent_topics=topics[:3],
        previous_suggestion=prior.suggested_question if prior else "",
        previous_metric_keys=list(prior.available_metrics) if prior else [],
    )
