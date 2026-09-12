"""Application history loading through a user-scoped repository."""

from uuid import UUID

from app.agents.contracts import PreparedContext, TurnIdentity
from app.core.errors import ConflictError, NotFoundError
from app.db.session import Database
from app.repositories.conversation import ConversationRepository
from app.repositories.history import HistoryRepository
from app.repositories.turns import TurnRepository
from app.schemas.chat import ConversationPage, ConversationResponse, TurnPage
from app.services.turn_results import turn_response


class ConversationService:
    """Own the short history read session; never use checkpoint history."""

    def __init__(self, database: Database) -> None:
        self.database = database

    async def prepare(self, identity: TurnIdentity) -> PreparedContext:
        """Load a validated question and bounded history."""
        async with self.database.session() as session:
            return await HistoryRepository(session).prepare(identity)

    async def create(self, user_id: UUID, title: str) -> ConversationResponse:
        """Create and commit an owned conversation."""
        async with self.database.session() as session, session.begin():
            row = await ConversationRepository(session, user_id).create(title)
            result = ConversationResponse.model_validate(row)
        return result

    async def get(self, user_id: UUID, conversation_id: UUID) -> ConversationResponse:
        """Hide foreign, missing and archived resources."""
        async with self.database.session() as session:
            row = await ConversationRepository(session, user_id).get(conversation_id)
            if row is None:
                raise NotFoundError()
            return ConversationResponse.model_validate(row)

    async def page(self, user_id: UUID, limit: int, offset: int) -> ConversationPage:
        """Read a deterministic page of active conversations."""
        async with self.database.session() as session:
            rows = await ConversationRepository(session, user_id).page(limit, offset)
            return ConversationPage(
                items=[ConversationResponse.model_validate(row) for row in rows],
                limit=limit,
                offset=offset,
            )

    async def archive(self, user_id: UUID, conversation_id: UUID) -> ConversationResponse:
        """Soft-delete only an idle owned conversation, preserving all history."""
        async with self.database.session() as session, session.begin():
            turns = TurnRepository(session, user_id)
            await turns.lock_conversation(conversation_id)
            if await turns.has_running(conversation_id):
                raise ConflictError()
            row = await ConversationRepository(session, user_id).get(conversation_id)
            if row is None:
                raise NotFoundError()
            row.archived_at = await turns.current_time()
            await session.flush()
            await session.refresh(row)
            result = ConversationResponse.model_validate(row)
        return result

    async def turns(
        self, user_id: UUID, conversation_id: UUID, limit: int, offset: int
    ) -> TurnPage:
        """Return persisted turns; no graph or external calls."""
        await self.get(user_id, conversation_id)
        async with self.database.session() as session:
            rows = await TurnRepository(session, user_id).page(conversation_id, limit, offset)
            return TurnPage(items=[turn_response(row) for row in rows], limit=limit, offset=offset)
