"""User-scoped chronological summary reads and conditional writes."""

from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import ConflictError, NotFoundError
from app.db.models import Conversation, Turn, TurnRole, TurnStatus
from app.repositories.conversation import ConversationRepository
from app.repositories.turns import TurnRepository
from app.schemas.summary import SummaryWork


class SummaryRepository:
    """No commits or model calls; every read/write is owned by the authenticated user."""

    def __init__(self, session: AsyncSession, user_id: UUID) -> None:
        self.session = session
        self.user_id = user_id

    async def next_work(self, conversation_id: UUID, target_turn_id: UUID) -> SummaryWork | None:
        """Read one assistant pair after the stored cursor, never skip a running turn."""
        conversation = await ConversationRepository(self.session, self.user_id).get(conversation_id)
        if conversation is None:
            raise NotFoundError()
        turns = TurnRepository(self.session, self.user_id)
        target = await turns.get(conversation_id, target_turn_id)
        if target is None or target.role is not TurnRole.ASSISTANT:
            raise NotFoundError()
        row = await self.session.scalar(
            select(Turn)
            .join(Conversation, Turn.conversation_id == Conversation.id)
            .where(
                Conversation.user_id == self.user_id,
                Conversation.id == conversation_id,
                Conversation.archived_at.is_(None),
                Turn.role == TurnRole.ASSISTANT,
                Turn.seq > conversation.summary_through_seq,
                Turn.seq <= target.seq,
            )
            .order_by(Turn.seq)
            .limit(1)
        )
        if row is None or row.status is TurnStatus.RUNNING:
            return None
        source = await turns.get(conversation_id, row.reply_to_turn_id) if row.reply_to_turn_id else None
        if source is None or source.role is not TurnRole.USER or source.seq >= row.seq:
            raise ConflictError("Summary requires an owned preceding user turn.")
        return SummaryWork(
            expected_seq=conversation.summary_through_seq,
            covered_seq=row.seq,
            existing_summary=conversation.summary or "",
            latest_user=source.content,
            latest_answer=row.content,
            status=row.status,
        )

    async def advance(self, conversation_id: UUID, work: SummaryWork, summary: str) -> bool:
        """Reject obsolete completions, foreign conversations and archived resources."""
        if work.covered_seq <= work.expected_seq:
            raise ConflictError("Summary cursor must advance.")
        updated = await self.session.scalar(
            update(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.user_id == self.user_id,
                Conversation.archived_at.is_(None),
                Conversation.summary_through_seq == work.expected_seq,
            )
            .values(summary=summary, summary_through_seq=work.covered_seq)
            .returning(Conversation.id)
        )
        return updated is not None
