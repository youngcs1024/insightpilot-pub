"""Real commits prove provenance, per-user guards and atomic background writes."""

import asyncio
from collections.abc import AsyncIterator

import pytest

from app.core.config_models import Settings
from app.core.errors import DatabaseError, MemoryExtractionError
from app.db.models.turn import TurnStatus
from app.db.session import Database
from app.repositories.memory import MemoryRepository
from app.repositories.turns import TurnRepository
from app.schemas.memory import MemoryCreate, MemoryType, StoredMemory
from app.schemas.memory_extraction import MemoryExtraction
from app.services.memory.extract import MemoryExtractionService
from tests.fakes.chat_model import FakeChatModel
from tests.memory_extraction_support import candidate, extraction_input, source_pair
from tests.shared_database import TestPostgres

pytestmark = pytest.mark.integration


@pytest.fixture
async def extraction_db(migrated_db: TestPostgres) -> AsyncIterator[Database]:
    database = Database(migrated_db.app)
    database.start()
    try:
        yield database
    finally:
        await database.aclose()


async def test_background_persists_owned_provenance(
    extraction_db: Database, settings: Settings
) -> None:
    identity = await source_pair(extraction_db, extraction_input())
    llm = FakeChatModel([MemoryExtraction(candidates=[candidate()])])
    await MemoryExtractionService(extraction_db, settings, llm).run(identity)
    async with extraction_db.session() as session:
        stored = await MemoryRepository(session, identity.user_id).list_active(
            MemoryType.METRIC_OVERRIDE
        )
    assert len(stored) == 1
    assert stored[0].source_turn_id == identity.turn_id
    assert stored[0].content == candidate().content
    assert "以后退款率都按退款申请时间算" in llm.calls[0].messages[1].content


@pytest.mark.parametrize(
    "status", [TurnStatus.FAILED, TurnStatus.ABSTAINED, TurnStatus.DEGRADED, TurnStatus.RUNNING]
)
async def test_persisted_status_gate(
    extraction_db: Database, settings: Settings, status: TurnStatus
) -> None:
    identity = await source_pair(extraction_db, extraction_input(status=status))
    llm = FakeChatModel([])
    await MemoryExtractionService(extraction_db, settings, llm).run(identity)
    assert not llm.calls
    async with extraction_db.session() as session:
        assert not await MemoryRepository(session, identity.user_id).list_active(
            MemoryType.METRIC_OVERRIDE
        )


async def test_duplicate_preserves_content_and_touches_timestamp(
    extraction_db: Database, settings: Settings
) -> None:
    first = await source_pair(extraction_db, extraction_input())
    second = await source_pair(extraction_db, extraction_input(), owner=first)
    llm = FakeChatModel([MemoryExtraction(candidates=[candidate()]) for _ in range(2)])
    service = MemoryExtractionService(extraction_db, settings, llm)
    await service.run(first)
    async with extraction_db.session() as session:
        original = await MemoryRepository(session, first.user_id).list_history(
            MemoryType.METRIC_OVERRIDE
        )
    await service.run(second)
    async with extraction_db.session() as session:
        refreshed = await MemoryRepository(session, first.user_id).list_history(
            MemoryType.METRIC_OVERRIDE
        )
        assert len(refreshed) == 1
        assert refreshed[0].updated_at > original[0].updated_at
        assert refreshed[0].model_dump(exclude={"updated_at"}) == original[0].model_dump(
            exclude={"updated_at"}
        )


async def test_same_batch_last_conflicting_candidate_wins(
    extraction_db: Database, settings: Settings
) -> None:
    identity = await source_pair(extraction_db, extraction_input())
    changed = candidate(content={"metric_key": "refund_rate", "patch": {"date_field": "o.paid_at"}})
    service = MemoryExtractionService(
        extraction_db,
        settings,
        FakeChatModel([MemoryExtraction(candidates=[candidate(), changed])]),
    )
    await service.run(identity)
    async with extraction_db.session() as session:
        stored = await MemoryRepository(session, identity.user_id).list_active(
            MemoryType.METRIC_OVERRIDE
        )
    assert len(stored) == 1
    assert stored[0].content == changed.content


async def test_cross_session_concurrent_writes_have_one_active_memory(
    extraction_db: Database, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = await source_pair(extraction_db, extraction_input())
    second = await source_pair(extraction_db, extraction_input(), owner=first)
    # Both jobs reach the lock before either can write: this also tests the empty-set race.
    entered = 0
    ready = asyncio.Event()
    original = MemoryRepository.lock_writes

    async def lock(repo: MemoryRepository) -> None:
        nonlocal entered
        entered += 1
        if entered == 2:  # noqa: PLR2004 -- both concurrent writers reached the lock.
            ready.set()
        await ready.wait()
        await original(repo)

    monkeypatch.setattr(MemoryRepository, "lock_writes", lock)
    llm = FakeChatModel([MemoryExtraction(candidates=[candidate()]) for _ in range(2)])
    service = MemoryExtractionService(extraction_db, settings, llm)
    async with asyncio.timeout(10):
        await asyncio.gather(service.run(first), service.run(second))
    async with extraction_db.session() as session:
        stored = await MemoryRepository(session, first.user_id).list_history(
            MemoryType.METRIC_OVERRIDE
        )
    assert len(stored) == 1
    assert stored[0].source_turn_id in {first.turn_id, second.turn_id}


async def test_other_users_memory_does_not_block_write(
    extraction_db: Database, settings: Settings
) -> None:
    first = await source_pair(extraction_db, extraction_input())
    other = await source_pair(extraction_db, extraction_input())
    service = MemoryExtractionService(
        extraction_db,
        settings,
        FakeChatModel([MemoryExtraction(candidates=[candidate()]) for _ in range(2)]),
    )
    await service.run(first)
    await service.run(other)
    async with extraction_db.session() as session:
        a = await MemoryRepository(session, first.user_id).list_active(MemoryType.METRIC_OVERRIDE)
        b = await MemoryRepository(session, other.user_id).list_active(MemoryType.METRIC_OVERRIDE)
    assert len(a) == len(b) == 1
    assert a[0].user_id != b[0].user_id
    assert a[0].source_turn_id == first.turn_id
    assert b[0].source_turn_id == other.turn_id


async def test_foreign_source_is_rejected_before_model(
    extraction_db: Database, settings: Settings
) -> None:
    first = await source_pair(extraction_db, extraction_input())
    other = await source_pair(extraction_db, extraction_input())
    llm = FakeChatModel([])
    service = MemoryExtractionService(extraction_db, settings, llm)
    with pytest.raises(MemoryExtractionError):
        await service.run(first.model_copy(update={"user_id": other.user_id}))
    assert not llm.calls


async def test_write_batch_failure_rolls_back_all_candidates(
    extraction_db: Database, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = await source_pair(extraction_db, extraction_input())
    other_type = candidate(memory_type="terminology", content={"term": "退款", "means": "申请退款"})
    original = MemoryRepository.create

    async def fail_second(repo: MemoryRepository, value: MemoryCreate) -> StoredMemory:
        if value.memory_type is MemoryType.TERMINOLOGY:
            raise DatabaseError()
        return await original(repo, value)

    monkeypatch.setattr(MemoryRepository, "create", fail_second)
    service = MemoryExtractionService(
        extraction_db,
        settings,
        FakeChatModel([MemoryExtraction(candidates=[candidate(), other_type])]),
    )
    with pytest.raises(MemoryExtractionError):
        await service.run(identity)
    async with extraction_db.session() as session:
        repo = MemoryRepository(session, identity.user_id)
        assert not await repo.list_history(MemoryType.METRIC_OVERRIDE)
        assert not await repo.list_history(MemoryType.TERMINOLOGY)
        turn = await TurnRepository(session, identity.user_id).get(
            identity.conversation_id, identity.turn_id
        )
        assert turn.status is TurnStatus.SUCCEEDED


async def test_model_call_has_no_open_database_session(
    extraction_db: Database, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = await source_pair(extraction_db, extraction_input())
    llm = FakeChatModel([MemoryExtraction()])
    original = llm.generate_structured

    async def generate(*args: object, **kwargs: object) -> object:
        assert extraction_db.engine.pool.checkedout() == 0
        return await original(*args, **kwargs)

    monkeypatch.setattr(llm, "generate_structured", generate)
    await MemoryExtractionService(extraction_db, settings, llm).run(identity)
