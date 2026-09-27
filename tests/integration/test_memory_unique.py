"""Deferred uniqueness is enforced at commit without changing append-only history."""

# ruff: noqa: PLR2004 -- fixed acceptance counts and retry/timeout boundaries.

import asyncio
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text

from app.core.config_models import Settings
from app.core.errors import MemoryWriteConflictError
from app.db.session import Database
from app.repositories.memory import MemoryRepository
from app.schemas.memory import MemoryType
from app.schemas.memory_extraction import MemoryExtraction
from app.services.memory.extract import MemoryExtractionService
from app.services.memory.integrity import validate_memory_schema
from tests.fakes.chat_model import FakeChatModel
from tests.memory_extraction_support import candidate, extraction_input, source_pair
from tests.memory_support import memory_input
from tests.shared_database import TestPostgres

pytestmark = pytest.mark.integration


@pytest.fixture
async def memory_db(migrated_db: TestPostgres) -> AsyncIterator[Database]:
    database = Database(migrated_db.app)
    database.start()
    try:
        yield database
    finally:
        await database.aclose()


async def test_duplicate_active_metric_rejected_only_at_commit(memory_db: Database) -> None:
    identity = await source_pair(memory_db, extraction_input())
    value = memory_input(
        identity.turn_id,
        kind=MemoryType.METRIC_OVERRIDE,
        content={"metric_key": "refund_rate", "patch": {"date_field": "o.paid_at"}},
    )
    async with memory_db.session() as session, session.begin():
        original = await MemoryRepository(session, identity.user_id).create(value)
    with pytest.raises(MemoryWriteConflictError):  # noqa: PT012 -- assert INSERT succeeds before COMMIT fails.
        async with memory_db.session() as session, session.begin():
            new = await MemoryRepository(session, identity.user_id).create(value)
            assert new.id != original.id  # INSERT/flush succeeded; COMMIT must reject it.
    async with memory_db.session() as session:
        assert await MemoryRepository(session, identity.user_id).list_history(
            MemoryType.METRIC_OVERRIDE
        ) == [original]


async def test_deferred_replacement_commits_and_historical_versions_survive(
    memory_db: Database,
) -> None:
    identity = await source_pair(memory_db, extraction_input())
    value = memory_input(
        identity.turn_id,
        kind=MemoryType.METRIC_OVERRIDE,
        content={"metric_key": "refund_rate", "patch": {"date_field": "o.paid_at"}},
    )
    async with memory_db.session() as session, session.begin():
        old = await MemoryRepository(session, identity.user_id).create(value)
    async with memory_db.session() as session, session.begin():
        repo = MemoryRepository(session, identity.user_id)
        new = await repo.create(value)
        await repo.supersede(old.id, by=new.id)
    async with memory_db.session() as session:
        repo = MemoryRepository(session, identity.user_id)
        assert [row.id for row in await repo.list_active(MemoryType.METRIC_OVERRIDE)] == [new.id]
        assert (await repo.get(old.id)).superseded_by == new.id
        key = await session.scalar(
            text("SELECT active_metric_key FROM memories WHERE id=:id AND user_id=:user"),
            {"id": old.id, "user": identity.user_id},
        )
        assert key is None
    await validate_memory_schema(memory_db, timeout_s=5)


async def test_concurrent_supersession_has_one_active_override(
    memory_db: Database, settings: Settings
) -> None:
    identity = await source_pair(memory_db, extraction_input())
    other = await source_pair(memory_db, extraction_input(), owner=identity)
    writer = MemoryExtractionService(memory_db, settings, FakeChatModel([]))
    first = MemoryExtraction(candidates=[candidate()])
    second = MemoryExtraction(
        candidates=[
            candidate(content={"metric_key": "refund_rate", "patch": {"date_field": "o.paid_at"}})
        ]
    )
    await asyncio.gather(writer._write(identity, first), writer._write(other, second))
    async with memory_db.session() as session:
        repo = MemoryRepository(session, identity.user_id)
        active = await repo.list_active(MemoryType.METRIC_OVERRIDE)
        rows = await repo.list_history(MemoryType.METRIC_OVERRIDE)
    assert len(active) == 1
    assert len(rows) == 2
    assert next(row for row in rows if not row.is_active).superseded_by == active[0].id
