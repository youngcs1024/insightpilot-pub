"""User-scoped active conversation lookup."""

from uuid import UUID

from app.db.models import Conversation
from app.repositories.base import UserScopedRepository


class ConversationRepository(UserScopedRepository[Conversation]):
    """Never disclose another identity's conversations."""

    model = Conversation

    async def get(self, conversation_id: UUID) -> Conversation | None:
        """Return an owned, non-archived conversation or nothing."""
        result: Conversation | None = await self._session.scalar(
            self._scoped().where(
                Conversation.id == conversation_id, Conversation.archived_at.is_(None)
            )
        )
        return result

    async def create(self, title: str) -> Conversation:
        """Insert an owned conversation; caller commits."""
        conversation = Conversation(user_id=self._user_id, title=title)
        self._session.add(conversation)
        await self._session.flush()
        return conversation

    async def page(self, limit: int, offset: int) -> list[Conversation]:
        """Return active owned conversations in deterministic creation order."""
        rows = await self._session.scalars(
            self._scoped()
            .where(Conversation.archived_at.is_(None))
            .order_by(Conversation.created_at.desc(), Conversation.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return list(rows)
