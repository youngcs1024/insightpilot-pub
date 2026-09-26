"""Typed memory persistence; every lookup is owned and callers control transactions."""

from uuid import UUID

import structlog
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import func, select, text

from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.db.models.conversation import Conversation
from app.db.models.memory import MemoryRecord
from app.db.models.turn import Turn
from app.repositories.base import UserScopedRepository
from app.schemas.memory import MemoryCreate, MemoryType, StoredMemory

logger = structlog.get_logger(__name__)


def validated_input(value: MemoryCreate) -> MemoryCreate:
    """Revalidate even constructed or mutated models without logging their payloads."""
    try:
        return MemoryCreate.model_validate(value.model_dump(mode="json", warnings=False))
    except PydanticValidationError as exc:
        logger.exception(
            "memory_validation_failed",
            validation_types=[error["type"] for error in exc.errors(include_input=False)],
            exc_info=False,
        )
        raise ValidationError("Invalid memory content") from None


def projection(record: MemoryRecord) -> StoredMemory:
    """Never expose an unvalidated JSON object across the repository boundary."""
    try:
        return StoredMemory.model_validate(record, from_attributes=True)
    except PydanticValidationError:
        logger.exception("memory_record_invalid", memory_id=str(record.id), exc_info=False)
        raise ValidationError("Invalid stored memory") from None


class MemoryRepository(UserScopedRepository[MemoryRecord]):
    """Preserve original content and provenance; only explicit supersession mutates rows."""

    model = MemoryRecord

    async def lock_writes(self) -> None:
        """Serialize this user's memory checks/inserts, including an empty active set."""
        await self._session.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
            {"key": f"memory-write:{self._user_id}"},
        )

    async def create(self, value: MemoryCreate) -> StoredMemory:
        """Validate content and owned provenance before inserting any row."""
        value = validated_input(value)
        source = await self._session.scalar(
            select(Turn.id)
            .join(Conversation, Conversation.id == Turn.conversation_id)
            .where(Turn.id == value.source_turn_id, Conversation.user_id == self._user_id)
        )
        if source is None:
            raise NotFoundError()
        record = MemoryRecord(
            user_id=self._user_id,
            source_turn_id=value.source_turn_id,
            memory_type=value.memory_type,
            content=value.content.model_dump(mode="json"),
            summary=value.summary,
            confidence=value.confidence,
        )
        self._session.add(record)
        await self._session.flush()
        return projection(record)

    async def get(self, memory_id: UUID) -> StoredMemory | None:
        """Return an owned active or historical memory, hiding foreign identities."""
        record = await self._session.scalar(self._scoped().where(MemoryRecord.id == memory_id))
        return projection(record) if record is not None else None

    async def list_active(self, memory_type: MemoryType) -> list[StoredMemory]:
        """Read active memories of one closed type in deterministic order."""
        records = await self._session.scalars(
            self._scoped()
            .where(MemoryRecord.memory_type == memory_type, MemoryRecord.is_active.is_(True))
            .order_by(MemoryRecord.created_at, MemoryRecord.id)
        )
        return [projection(record) for record in records]

    async def list_history(self, memory_type: MemoryType) -> list[StoredMemory]:
        """Return preserved rows, including superseded versions, for an owned type."""
        records = await self._session.scalars(
            self._scoped()
            .where(MemoryRecord.memory_type == memory_type)
            .order_by(MemoryRecord.created_at, MemoryRecord.id)
        )
        return [projection(record) for record in records]

    async def touch(self, memory_id: UUID) -> StoredMemory:
        """Update only the timestamp of an active owned duplicate."""
        record = await self._session.scalar(
            self._scoped()
            .where(MemoryRecord.id == memory_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        if record is None:
            raise NotFoundError()
        if not record.is_active:
            raise ConflictError("Cannot touch a superseded memory")
        record.updated_at = (await self._session.execute(select(func.clock_timestamp()))).scalar_one()
        await self._session.flush()
        await self._session.refresh(record)
        return projection(record)

    async def page(
        self,
        *,
        include_superseded: bool,
        memory_type: MemoryType | None,
        limit: int,
        offset: int,
    ) -> list[StoredMemory]:
        """Read a bounded owned history page in stable creation order."""
        statement = self._scoped()
        if not include_superseded:
            statement = statement.where(MemoryRecord.is_active.is_(True))
        if memory_type is not None:
            statement = statement.where(MemoryRecord.memory_type == memory_type)
        records = await self._session.scalars(
            statement.order_by(MemoryRecord.created_at, MemoryRecord.id).limit(limit).offset(offset)
        )
        return [projection(record) for record in records]

    async def supersede(self, memory_id: UUID, *, by: UUID) -> StoredMemory:
        """Lock both versions in ID order and retire only an active owned source."""
        records = list(
            await self._session.scalars(
                self._scoped()
                .where(MemoryRecord.id.in_([memory_id, by]))
                .order_by(MemoryRecord.id)
                .with_for_update()
                .execution_options(populate_existing=True)
            )
        )
        owned = {record.id: record for record in records}
        if memory_id not in owned or by not in owned:
            raise NotFoundError()
        old, replacement = owned[memory_id], owned[by]
        if (
            memory_id == by
            or not old.is_active
            or not replacement.is_active
            or old.memory_type != replacement.memory_type
        ):
            raise ConflictError("Invalid memory supersession")
        old.is_active = False
        old.superseded_by = replacement.id
        old.superseded_at = await self._session.scalar(select(func.clock_timestamp()))
        await self._session.flush()
        await self._session.refresh(old)
        return projection(old)
