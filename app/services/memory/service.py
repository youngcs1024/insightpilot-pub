"""Public owned memory history reads; retrieval eligibility belongs to Step 6.4."""

import asyncio
from uuid import UUID

from app.core.config_models import DatabaseSettings
from app.db.session import Database
from app.repositories.memory import MemoryRepository
from app.schemas.memory import MemoryType
from app.schemas.memory_api import MemoryPage


class MemoryService:
    """Own bounded database reads without exposing sessions to the API."""

    def __init__(self, database: Database, settings: DatabaseSettings) -> None:
        self.database = database
        self.timeout_s = settings.command_timeout_s

    async def page(
        self,
        user_id: UUID,
        *,
        include_superseded: bool,
        memory_type: MemoryType | None,
        limit: int,
        offset: int,
    ) -> MemoryPage:
        """Return only the authenticated user's active or historical memories."""
        async with self.database.session() as session, asyncio.timeout(self.timeout_s):
            rows = await MemoryRepository(session, user_id).page(
                include_superseded=include_superseded,
                memory_type=memory_type,
                limit=limit,
                offset=offset,
            )
            return MemoryPage(items=rows, limit=limit, offset=offset)
