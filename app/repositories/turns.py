"""Conversation-locked, user-scoped turn persistence; callers own transactions."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import Select, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import ConflictError, NotFoundError
from app.db.models import Conversation, Turn, TurnRole, TurnStatus


class TurnRepository:
    """Require the user identity for every access, including assistant lookups."""

    def __init__(self, session: AsyncSession, user_id: UUID) -> None:
        self.session = session
        self.user_id = user_id

    def _scoped(self, conversation_id: UUID) -> Select[tuple[Turn]]:
        return (
            select(Turn)
            .join(Conversation, Turn.conversation_id == Conversation.id)
            .where(
                Conversation.user_id == self.user_id,
                Conversation.id == conversation_id,
                Conversation.archived_at.is_(None),
            )
        )

    async def lock_conversation(self, conversation_id: UUID) -> None:
        """Serialize turn admission without revealing another user's conversation."""
        row = await self.session.scalar(
            select(Conversation)
            .where(
                Conversation.id == conversation_id,
                Conversation.user_id == self.user_id,
                Conversation.archived_at.is_(None),
            )
            .with_for_update()
        )
        if row is None:
            raise NotFoundError()

    async def find_key(self, conversation_id: UUID, key: str) -> Turn | None:
        """Look up the stored assistant outcome under the conversation lock."""
        result: Turn | None = await self.session.scalar(
            self._scoped(conversation_id).where(Turn.idempotency_key == key)
        )
        return result

    async def get(self, conversation_id: UUID, turn_id: UUID) -> Turn | None:
        """Resolve a turn only within the owned conversation."""
        result: Turn | None = await self.session.scalar(
            self._scoped(conversation_id).where(Turn.id == turn_id)
        )
        return result

    async def create_pair(self, conversation_id: UUID, content: str, key: str | None) -> Turn:
        """Insert a completed user message and its exclusively running assistant."""
        await self.lock_conversation(conversation_id)
        running = await self.session.scalar(
            self._scoped(conversation_id).where(Turn.status == TurnStatus.RUNNING)
        )
        if running is not None:
            raise ConflictError("conversation already has a running turn")
        seq = await self.session.scalar(
            select(func.max(Turn.seq))
            .select_from(Turn)
            .join(Conversation, Turn.conversation_id == Conversation.id)
            .where(Conversation.user_id == self.user_id, Conversation.id == conversation_id)
        )
        user = Turn(
            conversation_id=conversation_id,
            seq=(seq or 0) + 1,
            role=TurnRole.USER,
            content=content,
            status=TurnStatus.SUCCEEDED,
        )
        self.session.add(user)
        await self.session.flush()
        assistant = Turn(
            conversation_id=conversation_id,
            seq=user.seq + 1,
            role=TurnRole.ASSISTANT,
            content="",
            status=TurnStatus.RUNNING,
            idempotency_key=key,
            reply_to_turn_id=user.id,
        )
        self.session.add(assistant)
        await self.session.flush()
        return assistant

    async def current_time(self) -> datetime:
        """Read database wall time after lock acquisition, not transaction start."""
        value: datetime = (await self.session.execute(select(func.clock_timestamp()))).scalar_one()
        return value

    async def release_key(self, conversation_id: UUID, turn_id: UUID) -> None:
        """Release a terminal expired key without removing any history."""
        turn = await self.get(conversation_id, turn_id)
        if turn is None or turn.status == TurnStatus.RUNNING:
            raise ConflictError()
        turn.idempotency_key = None
        await self.session.flush()

    def history(self, conversation_id: UUID, before_seq: int) -> Select[tuple[Turn]]:
        """Return only owned completed history before the current user question."""
        return (
            self._scoped(conversation_id)
            .where(
                Turn.seq < before_seq, Turn.status.in_([TurnStatus.SUCCEEDED, TurnStatus.DEGRADED])
            )
            .order_by(Turn.seq.desc())
            .limit(100)
        )

    async def page(self, conversation_id: UUID, limit: int, offset: int) -> list[Turn]:
        """Return ordered messages only within an active owned conversation."""
        rows = await self.session.scalars(
            self._scoped(conversation_id).order_by(Turn.seq).limit(limit).offset(offset)
        )
        return list(rows)

    async def has_running(self, conversation_id: UUID) -> bool:
        """Check running state under a caller-owned conversation lock."""
        return (
            await self.session.scalar(
                self._scoped(conversation_id).where(Turn.status == TurnStatus.RUNNING)
            )
            is not None
        )
