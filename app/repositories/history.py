"""Bounded application history and admission validation for graph invocations."""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.budget import SUMMARY_TOKENS, bounded_text
from app.agents.contracts import DataEvidence, HistoryMessage, PreparedContext, TurnIdentity
from app.agents.multiturn import trim_history
from app.core.errors import ConflictError, NotFoundError
from app.db.models import Conversation, Turn, TurnRole, TurnStatus
from app.db.models.evidence import DataEvidenceRecord, KnowledgeEvidenceRecord
from app.repositories.turns import TurnRepository
from app.schemas.knowledge import KnowledgeEvidence
from app.schemas.knowledge_query import KnowledgeHistoryTurn
from app.schemas.retrieval import PointTimeScope, RangeTimeScope


async def validate_turn(repository: TurnRepository, identity: TurnIdentity) -> Turn:
    """The graph may only operate on an admitted, running assistant message."""
    assistant = await repository.get(identity.conversation_id, identity.turn_id)
    if assistant is None:
        raise NotFoundError()
    if assistant.role != TurnRole.ASSISTANT or assistant.status != TurnStatus.RUNNING:
        raise ConflictError("turn is not a running assistant")
    return assistant


class HistoryRepository:
    """Bounded owned history projection; no transaction ownership."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def prepare(self, identity: TurnIdentity) -> PreparedContext:
        """Load a bounded owned window, excluding the current user message."""
        session = self.session
        repository = TurnRepository(session, identity.user_id)
        assistant = await validate_turn(repository, identity)
        user = (
            await repository.get(identity.conversation_id, assistant.reply_to_turn_id)
            if assistant.reply_to_turn_id
            else None
        )
        if user is None or user.role != TurnRole.USER:
            raise ConflictError("assistant has no associated user message")
        conversation = await session.scalar(
            select(Conversation).where(
                Conversation.id == identity.conversation_id,
                Conversation.user_id == identity.user_id,
            )
        )
        if conversation is None:
            raise NotFoundError()
        # Bound DB materialization as well as prompt tokens.
        turns = (
            await session.scalars(repository.history(identity.conversation_id, user.seq))
        ).all()
        messages = trim_history(
            [HistoryMessage(role=turn.role.value, content=turn.content) for turn in reversed(turns)]
        )
        has_prior_turns = (
            await session.scalar(
                select(Turn.id)
                .join(Conversation, Turn.conversation_id == Conversation.id)
                .where(
                    Conversation.user_id == identity.user_id,
                    Turn.conversation_id == identity.conversation_id,
                    Turn.seq < user.seq,
                )
                .limit(1)
            )
            is not None
        )
        records = (
            await session.scalars(
                select(DataEvidenceRecord)
                .join(Turn, DataEvidenceRecord.assistant_turn_id == Turn.id)
                .where(
                    DataEvidenceRecord.user_id == identity.user_id,
                    Turn.conversation_id == identity.conversation_id,
                    Turn.seq < user.seq,
                )
                .order_by(Turn.seq.desc())
                .limit(3)
            )
        ).all()
        knowledge_records = (
            await session.scalars(
                select(KnowledgeEvidenceRecord)
                .join(Turn, KnowledgeEvidenceRecord.assistant_turn_id == Turn.id)
                .where(
                    KnowledgeEvidenceRecord.user_id == identity.user_id,
                    Turn.conversation_id == identity.conversation_id,
                    Turn.seq < user.seq,
                    Turn.status.in_([TurnStatus.SUCCEEDED, TurnStatus.DEGRADED]),
                )
                .order_by(Turn.seq.desc())
                .limit(3)
            )
        ).all()
        knowledge_history = []
        for record in reversed(knowledge_records):
            evidence = KnowledgeEvidence.model_validate(record.payload)
            scope = evidence.time_scope.model_dump(mode="json")
            knowledge_history.append(
                KnowledgeHistoryTurn(
                    turn_id=record.assistant_turn_id,
                    question=evidence.query_used[:300],
                    time_scope=(
                        PointTimeScope if evidence.time_scope.kind == "point" else RangeTimeScope
                    ).model_validate(scope),
                )
            )
        return PreparedContext(
            has_prior_turns=has_prior_turns,
            question=user.content,
            summary=bounded_text(conversation.summary or "", SUMMARY_TOKENS),
            messages=messages,
            prior_sql=[DataEvidence.model_validate(record.payload).sql for record in records],
            knowledge_history=knowledge_history,
        )
