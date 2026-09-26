"""Owned memory history and fail-closed, bounded production retrieval."""

import asyncio
from uuid import UUID

import structlog

from app.core.deadline import Deadline
from app.core.errors import InsightPilotError
from app.core.observability import TraceMetadata, observe
from app.schemas.memory_retrieval import MemoryReadRequest, MemorySelection, MemoryStage
from app.services.memory.retrieve import TokenCounter, select_memories

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

    async def retrieve(
        self, request: MemoryReadRequest, *, deadline: Deadline, counter: TokenCounter,
    ) -> MemorySelection:
        """A failed read is distinguishable from a successful empty selection; no retry."""
        deadline.check("memory_read")
        types = ([MemoryType.TERMINOLOGY, MemoryType.FORMAT_PREFERENCE]
                 if request.stage is MemoryStage.PREPARE else list(MemoryType))
        with observe("memory_retrieve", TraceMetadata(memory_stage=request.stage.value)) as span:
            try:
                async with self.database.session() as session, asyncio.timeout(deadline.budget(self.timeout_s)):
                    rows = await MemoryRepository(session, request.user_id).active_candidates(types)
                deadline.check("memory_read_complete")
                result = select_memories(rows, request, counter)
            except InsightPilotError as exc:
                deadline.check("memory_read_failure")
                structlog.get_logger(__name__).exception("memory_read_failed", code=exc.code, exc_info=False)
                result = MemorySelection(failed=True, failure_code=exc.code)
            metadata = TraceMetadata(
                memory_considered=len(result.decisions), memory_selected=len(result.selected),
                memory_tokens=result.tokens, memory_failed=result.failed,
            )
            if span is not None:
                span.update(metadata)
            for decision in result.decisions:
                structlog.get_logger(__name__).info(
                    "memory_selection_decision", stage=request.stage.value,
                    reason=decision.reason.value, score=decision.score,
                )
            return result
