"""Real owned memory reads, fresh service instances and concurrent supersession."""

import asyncio
from collections.abc import AsyncIterator
from time import monotonic

import pytest

from app.core.deadline import Deadline
from app.core.errors import DatabaseError, DeadlineExceededError
from app.db.session import Database
from app.repositories.memory import MemoryRepository
from app.schemas.memory import MemoryType, StoredMemory
from app.schemas.memory_retrieval import MemoryReadRequest, MemoryStage
from app.services.memory.service import MemoryService
from app.services.schema_tokens import SchemaTokenCounter
from tests.memory_extraction_support import extraction_input, source_pair
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


async def test_owned_active_selection_survives_new_pool(memory_db: Database, migrated_db: TestPostgres) -> None:
    owner = await source_pair(memory_db, extraction_input())
    other = await source_pair(memory_db, extraction_input())
    async with memory_db.session() as session, session.begin():
        mine = await MemoryRepository(session, owner.user_id).create(memory_input(owner.turn_id))
        await MemoryRepository(session, other.user_id).create(memory_input(other.turn_id))
    restarted = Database(migrated_db.app)
    restarted.start()
    try:
        result = await MemoryService(restarted, migrated_db.app).retrieve(
            MemoryReadRequest(user_id=owner.user_id, question="大促政策", stage=MemoryStage.FINALIZE),
            deadline=Deadline(monotonic() + 10), counter=SchemaTokenCounter(),
        )
        assert [r.id for r in result.selected] == [mine.id]
        assert len(result.decisions) == 1
        assert result.selected[0].source_turn_id == owner.turn_id
    finally:
        await restarted.aclose()


async def test_preroute_database_query_excludes_metric_and_region_types(memory_db: Database, migrated_db: TestPostgres, monkeypatch: pytest.MonkeyPatch) -> None:
    owner = await source_pair(memory_db, extraction_input())
    seen = []
    original = MemoryRepository.active_candidates
    async def capture(repo: MemoryRepository, types: list[MemoryType]) -> list[StoredMemory]:
        seen.append(types)
        return await original(repo, types)
    monkeypatch.setattr(MemoryRepository, "active_candidates", capture)
    result = await MemoryService(memory_db, migrated_db.app).retrieve(
        MemoryReadRequest(user_id=owner.user_id, question="大促", stage=MemoryStage.PREPARE),
        deadline=Deadline(monotonic() + 10), counter=SchemaTokenCounter(),
    )
    assert not result.failed
    assert seen == [[MemoryType.TERMINOLOGY, MemoryType.FORMAT_PREFERENCE]]


async def test_supersession_between_stages_never_returns_old_row(memory_db: Database, migrated_db: TestPostgres) -> None:
    owner = await source_pair(memory_db, extraction_input())
    async with memory_db.session() as session, session.begin():
        old = await MemoryRepository(session, owner.user_id).create(memory_input(owner.turn_id))
    service = MemoryService(memory_db, migrated_db.app)
    request = MemoryReadRequest(user_id=owner.user_id, question="大促政策", stage=MemoryStage.PREPARE)
    before = await service.retrieve(request, deadline=Deadline(monotonic() + 10), counter=SchemaTokenCounter())
    async with memory_db.session() as session, session.begin():
        repo = MemoryRepository(session, owner.user_id)
        new = await repo.create(memory_input(owner.turn_id, content={"term": "大促", "means": "双11"}))
        await repo.supersede(old.id, by=new.id)
    after = await service.retrieve(request.model_copy(update={"stage": MemoryStage.FINALIZE}), deadline=Deadline(monotonic() + 10), counter=SchemaTokenCounter())
    assert [r.id for r in before.selected] == [old.id]
    assert [r.id for r in after.selected] == [new.id]


@pytest.mark.parametrize("failure", [DatabaseError(), asyncio.CancelledError()])
async def test_read_failure_or_cancellation(memory_db: Database, migrated_db: TestPostgres, monkeypatch: pytest.MonkeyPatch, failure: BaseException) -> None:
    owner = await source_pair(memory_db, extraction_input())
    calls = 0
    async def fail(repo: MemoryRepository, types: list[MemoryType]) -> list[StoredMemory]:
        nonlocal calls
        calls += 1
        raise failure
    monkeypatch.setattr(MemoryRepository, "active_candidates", fail)
    operation = MemoryService(memory_db, migrated_db.app).retrieve(
        MemoryReadRequest(user_id=owner.user_id, question="大促", stage=MemoryStage.PREPARE),
        deadline=Deadline(monotonic() + 10), counter=SchemaTokenCounter(),
    )
    if isinstance(failure, asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            await operation
    else:
        result = await operation
        assert result.failed and result.selected == []
        assert result.failure_code == failure.code
    assert calls == 1


async def test_command_timeout_is_degraded_but_request_expiry_propagates(memory_db: Database, migrated_db: TestPostgres, monkeypatch: pytest.MonkeyPatch) -> None:
    owner = await source_pair(memory_db, extraction_input())
    async def hang(repo: MemoryRepository, types: list[MemoryType]) -> list[StoredMemory]:
        await asyncio.Event().wait()
        return []
    monkeypatch.setattr(MemoryRepository, "active_candidates", hang)
    service = MemoryService(memory_db, migrated_db.app.model_copy(update={"command_timeout_s": 0.01}))
    request = MemoryReadRequest(user_id=owner.user_id, question="大促", stage=MemoryStage.PREPARE)
    assert (await service.retrieve(request, deadline=Deadline(monotonic() + 10), counter=SchemaTokenCounter())).failed
    with pytest.raises(DeadlineExceededError):
        await service.retrieve(request, deadline=Deadline(0), counter=SchemaTokenCounter())
