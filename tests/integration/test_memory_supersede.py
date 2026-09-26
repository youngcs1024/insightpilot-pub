"""Real committed transactions prove replacement history and atomic rollback."""

# ruff: noqa: PLR2004 -- fixed threshold, row-count and pagination acceptance values.

import asyncio
from collections.abc import AsyncIterator
from uuid import UUID

import pytest
from structlog.testing import capture_logs

from app.agents.contracts import TurnIdentity
from app.core.config_models import Settings
from app.core.errors import ConflictError, DatabaseError, NotFoundError
from app.db.session import Database
from app.repositories.memory import MemoryRepository
from app.schemas.memory import MemoryCreate, MemoryType, StoredMemory
from app.schemas.memory_extraction import MemoryCandidate, MemoryExtraction
from app.schemas.memory_write import WriteStatus
from app.services.memory.extract import MemoryExtractionService
from tests.fakes.chat_model import FakeChatModel
from tests.memory_extraction_support import candidate, extraction_input, source_pair
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


async def history(database: Database, identity: TurnIdentity) -> list[StoredMemory]:
    async with database.session() as session:
        return await MemoryRepository(session, identity.user_id).list_history(MemoryType.METRIC_OVERRIDE)


def service(database: Database, settings: Settings) -> MemoryExtractionService:
    return MemoryExtractionService(database, settings, FakeChatModel([]))


def changed(date: str = "o.paid_at") -> MemoryCandidate:
    return candidate(content={"metric_key": "refund_rate", "patch": {"date_field": date}})


async def test_duplicate_touches_not_creates(memory_db: Database, settings: Settings) -> None:
    identity = await source_pair(memory_db, extraction_input())
    writer = service(memory_db, settings)
    first = await writer._write(identity, MemoryExtraction(candidates=[candidate()]))
    original = (await history(memory_db, identity))[0]
    second_identity = await source_pair(memory_db, extraction_input(), owner=identity)
    repeated = candidate(summary="new summary", confidence=0.7)
    second = await writer._write(second_identity, MemoryExtraction(candidates=[repeated]))
    rows = await history(memory_db, identity)
    assert first[0].status is WriteStatus.CREATED
    assert second[0].status is WriteStatus.DUPLICATE
    assert first[0].memory_id == second[0].memory_id
    assert len(rows) == 1
    assert rows[0].updated_at > original.updated_at
    assert rows[0].model_dump(exclude={"updated_at"}) == original.model_dump(exclude={"updated_at"})


async def test_conflict_creates_and_supersedes(memory_db: Database, settings: Settings) -> None:
    identity = await source_pair(memory_db, extraction_input())
    writer = service(memory_db, settings)
    first = (await writer._write(identity, MemoryExtraction(candidates=[candidate()])))[0]
    original = (await history(memory_db, identity))[0]
    next_turn = await source_pair(memory_db, extraction_input(), owner=identity)
    result = (await writer._write(next_turn, MemoryExtraction(candidates=[changed()])))[0]
    rows = {row.id: row for row in await history(memory_db, identity)}
    assert result.status is WriteStatus.SUPERSEDED
    assert result.superseded_id == first.memory_id
    old, new = rows[first.memory_id], rows[result.memory_id]
    assert not old.is_active and new.is_active
    assert old.superseded_by == new.id and old.superseded_at is not None
    assert new.source_turn_id == next_turn.turn_id
    assert new.content == changed().content
    assert old.model_dump(exclude={"is_active", "superseded_by", "superseded_at", "updated_at"}) == original.model_dump(exclude={"is_active", "superseded_by", "superseded_at", "updated_at"})


async def test_superseded_row_preserved_with_pointer(memory_db: Database, settings: Settings) -> None:
    identity = await source_pair(memory_db, extraction_input())
    with capture_logs() as logs:
        outcomes = await service(memory_db, settings)._write(
            identity, MemoryExtraction(candidates=[candidate(), changed()])
        )
    rows = {row.id: row for row in await history(memory_db, identity)}
    assert rows[outcomes[0].memory_id].superseded_by == outcomes[1].memory_id
    assert any(row["event"] == "memory_same_turn_conflict" and row["log_level"] == "warning" for row in logs)
    assert candidate().evidence_quote not in str(logs)


async def test_supersession_is_atomic(
    memory_db: Database, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = await source_pair(memory_db, extraction_input())
    writer = service(memory_db, settings)
    await writer._write(identity, MemoryExtraction(candidates=[candidate()]))
    original = await history(memory_db, identity)

    async def fail(repo: MemoryRepository, memory_id: UUID, *, by: UUID) -> StoredMemory:
        # The inserted version is visible inside the transaction before the failure.
        assert len(await repo.list_history(MemoryType.METRIC_OVERRIDE)) == 2
        raise DatabaseError()

    monkeypatch.setattr(MemoryRepository, "supersede", fail)
    with pytest.raises(DatabaseError):
        await writer._write(identity, MemoryExtraction(candidates=[changed()]))
    assert await history(memory_db, identity) == original


async def test_chain_of_three_supersessions_readable(memory_db: Database, settings: Settings) -> None:
    identity = await source_pair(memory_db, extraction_input())
    outcomes = await service(memory_db, settings)._write(
        identity, MemoryExtraction(candidates=[candidate(), changed(), changed("o.created_at"), candidate()])
    )
    rows = {row.id: row for row in await history(memory_db, identity)}
    assert len(rows) == 4
    for old, new in zip(outcomes, outcomes[1:], strict=False):
        assert rows[old.memory_id].superseded_by == new.memory_id
        assert not rows[old.memory_id].is_active
    assert rows[outcomes[-1].memory_id].is_active
    assert rows[outcomes[-1].memory_id].superseded_by is None


async def test_cannot_supersede_already_superseded(memory_db: Database, settings: Settings) -> None:
    identity = await source_pair(memory_db, extraction_input())
    outcomes = await service(memory_db, settings)._write(identity, MemoryExtraction(candidates=[candidate(), changed()]))
    async with memory_db.session() as session, session.begin():
        repo = MemoryRepository(session, identity.user_id)
        with pytest.raises(ConflictError):
            await repo.supersede(outcomes[0].memory_id, by=outcomes[1].memory_id)
        with pytest.raises(ConflictError):
            await repo.supersede(outcomes[1].memory_id, by=outcomes[0].memory_id)
        with pytest.raises(ConflictError):
            await repo.touch(outcomes[0].memory_id)
    other = await source_pair(memory_db, extraction_input())
    async with memory_db.session() as session:
        with pytest.raises(NotFoundError):
            await MemoryRepository(session, other.user_id).touch(outcomes[1].memory_id)


async def test_terminology_fuzzy_duplicate_detected(memory_db: Database, settings: Settings) -> None:
    identity = await source_pair(memory_db, extraction_input())
    a = candidate(memory_type="terminology", content={"term": "ＧＭＶ", "means": "退款申请时间"})
    b = candidate(memory_type="terminology", content={"term": "gmv", "means": "退款申请的时间"})
    results = await service(memory_db, settings)._write(identity, MemoryExtraction(candidates=[a, b]))
    assert results[1].status is WriteStatus.DUPLICATE
    async with memory_db.session() as session:
        rows = await MemoryRepository(session, identity.user_id).list_history(MemoryType.TERMINOLOGY)
    assert len(rows) == 1 and rows[0].content == a.content


async def test_distinct_keys_coexist(memory_db: Database, settings: Settings) -> None:
    identity = await source_pair(memory_db, extraction_input())
    values = [candidate(), candidate(content={"metric_key": "gmv", "patch": {}})]
    results = await service(memory_db, settings)._write(identity, MemoryExtraction(candidates=values))
    assert all(result.status is WriteStatus.CREATED for result in results)
    assert all(row.is_active for row in await history(memory_db, identity))


async def test_late_batch_failure_rolls_back_touch_and_supersession(
    memory_db: Database, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = await source_pair(memory_db, extraction_input())
    writer = service(memory_db, settings)
    await writer._write(identity, MemoryExtraction(candidates=[candidate()]))
    original = await history(memory_db, identity)
    create = MemoryRepository.create

    async def fail(repo: MemoryRepository, value: MemoryCreate) -> StoredMemory:
        if value.memory_type is MemoryType.REGION_FOCUS:
            raise DatabaseError()
        return await create(repo, value)

    monkeypatch.setattr(MemoryRepository, "create", fail)
    region = candidate(memory_type="region_focus", content={"region_ids": [1]})
    with pytest.raises(DatabaseError):
        await writer._write(identity, MemoryExtraction(candidates=[candidate(), changed(), region]))
    assert await history(memory_db, identity) == original


async def test_concurrent_conflicts_have_one_active_version(
    memory_db: Database, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = await source_pair(memory_db, extraction_input())
    second = await source_pair(memory_db, extraction_input(), owner=first)
    ready = asyncio.Event()
    entered = 0
    original = MemoryRepository.lock_writes

    async def lock(repo: MemoryRepository) -> None:
        nonlocal entered
        entered += 1
        if entered == 2:
            ready.set()
        await ready.wait()
        await original(repo)

    monkeypatch.setattr(MemoryRepository, "lock_writes", lock)
    writer = service(memory_db, settings)
    async with asyncio.timeout(10):
        await asyncio.gather(
            writer._write(first, MemoryExtraction(candidates=[candidate()])),
            writer._write(second, MemoryExtraction(candidates=[changed()])),
        )
    rows = await history(memory_db, first)
    assert len(rows) == 2
    active = [row for row in rows if row.is_active]
    assert len(active) == 1
    assert next(row for row in rows if not row.is_active).superseded_by == active[0].id


async def test_ambiguous_active_key_fails_without_partial_writes(
    memory_db: Database, settings: Settings
) -> None:
    identity = await source_pair(memory_db, extraction_input())
    value = MemoryCreate(
        source_turn_id=identity.turn_id,
        memory_type=candidate().memory_type,
        content=candidate().content,
        summary="original",
        confidence=0.9,
    )
    async with memory_db.session() as session, session.begin():
        repo = MemoryRepository(session, identity.user_id)
        await repo.create(value)
        await repo.create(value)
    original = await history(memory_db, identity)
    with pytest.raises(ConflictError):
        await service(memory_db, settings)._write(
            identity, MemoryExtraction(candidates=[changed()])
        )
    assert await history(memory_db, identity) == original


@pytest.mark.parametrize(
    ("kind", "first", "second"),
    [
        ("region_focus", {"region_ids": [1, 2]}, {"region_ids": [3]}),
        ("format_preference", {"prefer": "table", "decimals": 2},
         {"prefer": "prose", "decimals": 2}),
        ("terminology", {"term": "大促", "means": "618"},
         {"term": "大促", "means": "双11"}),
    ],
)
async def test_other_types_replace_and_preserve_original(
    memory_db: Database, settings: Settings,
    kind: str, first: dict[str, object], second: dict[str, object]
) -> None:
    identity = await source_pair(memory_db, extraction_input())
    a, b = candidate(memory_type=kind, content=first), candidate(memory_type=kind, content=second)
    outcomes = await service(memory_db, settings)._write(
        identity, MemoryExtraction(candidates=[a, b])
    )
    assert outcomes[1].status is WriteStatus.SUPERSEDED
    async with memory_db.session() as session:
        repo = MemoryRepository(session, identity.user_id)
        old = await repo.get(outcomes[0].memory_id)
        active = await repo.list_active(MemoryType(kind))
    assert old.content == a.content and not old.is_active
    assert len(active) == 1 and active[0].content == b.content
