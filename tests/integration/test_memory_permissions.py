"""Runtime grants protect memory payloads independently of repository conventions."""

import pytest
from sqlalchemy import delete, text, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.memory import MemoryRecord
from app.repositories.memory import MemoryRepository
from app.schemas.memory import MemoryType
from tests.memory_support import memory_input, memory_owner

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "field", ["content", "summary", "confidence", "source_turn_id", "user_id", "created_at"]
)
async def test_runtime_cannot_rewrite_memory(db_session: AsyncSession, field: str) -> None:
    user_id, turn_id = await memory_owner(db_session)
    repo = MemoryRepository(db_session, user_id)
    original = await repo.create(memory_input(turn_id))
    statement = (
        update(MemoryRecord)
        .where(MemoryRecord.id == original.id)
        .values({field: getattr(MemoryRecord, field)})
    )
    savepoint = await db_session.begin_nested()
    with pytest.raises(DBAPIError) as caught:
        await db_session.execute(statement)
    assert caught.value.orig.sqlstate == "42501"
    await savepoint.rollback()
    assert await repo.get(original.id) == original


async def test_runtime_cannot_delete_memory(db_session: AsyncSession) -> None:
    user_id, turn_id = await memory_owner(db_session)
    repo = MemoryRepository(db_session, user_id)
    original = await repo.create(memory_input(turn_id))
    savepoint = await db_session.begin_nested()
    with pytest.raises(DBAPIError) as caught:
        await db_session.execute(delete(MemoryRecord).where(MemoryRecord.id == original.id))
    assert caught.value.orig.sqlstate == "42501"
    await savepoint.rollback()
    assert await repo.list_history(MemoryType.TERMINOLOGY) == [original]


async def test_runtime_can_only_update_lifecycle_columns(db_session: AsyncSession) -> None:
    assert await db_session.scalar(text("SELECT current_user")) == "app_rw"
    assert not await db_session.scalar(
        text("SELECT has_table_privilege(current_user, 'memories', 'UPDATE')")
    )
    for field in ("is_active", "superseded_by", "superseded_at", "updated_at"):
        assert await db_session.scalar(
            text("SELECT has_column_privilege(current_user, 'memories', :field, 'UPDATE')"),
            {"field": field},
        )
    user_id, turn_id = await memory_owner(db_session)
    repo = MemoryRepository(db_session, user_id)
    original = await repo.create(memory_input(turn_id))
    replacement = await repo.create(memory_input(turn_id))
    await repo.supersede(original.id, by=replacement.id)
    assert await repo.list_active(MemoryType.TERMINOLOGY) == [replacement]
