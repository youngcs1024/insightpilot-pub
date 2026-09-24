"""Step 6.1: typed writes, mandatory provenance and immutable owned history."""

from uuid import uuid4

import pytest
from sqlalchemy import insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from structlog.testing import capture_logs

from app.core.errors import ConflictError, NotFoundError, ValidationError
from app.db.models.memory import MemoryRecord
from app.repositories.memory import MemoryRepository
from app.schemas.memory import MemoryType
from tests.memory_support import memory_input, memory_owner, raw_memory

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    ("kind", "content"),
    [
        (MemoryType.METRIC_OVERRIDE, {"metric_key": "gmv", "patch": {"date_field": "paid_at"}}),
        (MemoryType.REGION_FOCUS, {"region_ids": [1, 2]}),
        (MemoryType.TERMINOLOGY, {"term": "大促", "means": "618"}),
        (MemoryType.FORMAT_PREFERENCE, {"prefer": "table", "decimals": 2}),
    ],
)
async def test_content_validated_per_type(
    db_session: AsyncSession, kind: MemoryType, content: dict[str, object]
) -> None:
    user_id, turn_id = await memory_owner(db_session)
    repo = MemoryRepository(db_session, user_id)
    value = memory_input(turn_id, kind=kind, content=content)
    saved = await repo.create(value)
    db_session.expunge_all()
    restored = await repo.get(saved.id)
    assert restored == saved
    assert restored.content == value.content
    assert restored.source_turn_id == turn_id
    assert restored.created_at.tzinfo is not None
    assert restored.updated_at == restored.created_at
    assert restored.is_active
    assert restored.superseded_by is None


@pytest.mark.parametrize("source", [None, "orphan"])
async def test_source_turn_required(db_session: AsyncSession, source: str | None) -> None:
    user_id, turn_id = await memory_owner(db_session)
    values = raw_memory(user_id, turn_id)
    values["source_turn_id"] = uuid4() if source else None
    with pytest.raises(IntegrityError):
        await db_session.execute(insert(MemoryRecord).values(**values))


@pytest.mark.parametrize("confidence", [-0.01, 1.01, float("nan"), float("inf")])
async def test_confidence_range_enforced(db_session: AsyncSession, confidence: float) -> None:
    user_id, turn_id = await memory_owner(db_session)
    values = raw_memory(user_id, turn_id)
    values["confidence"] = confidence
    with pytest.raises(IntegrityError):
        await db_session.execute(insert(MemoryRecord).values(**values))


@pytest.mark.parametrize(
    "update",
    [
        {"memory_type": MemoryType.REGION_FOCUS},
        {"memory_type": "world_knowledge"},
        {"summary": "x" * 201},
        {"confidence": -0.01},
        {"confidence": float("nan")},
        {"content": {"region_ids": []}, "memory_type": MemoryType.REGION_FOCUS},
        {"content": {"region_ids": [1] * 6}, "memory_type": MemoryType.REGION_FOCUS},
        {"content": {"region_ids": [True]}, "memory_type": MemoryType.REGION_FOCUS},
        {
            "content": {"prefer": "table", "decimals": 5},
            "memory_type": MemoryType.FORMAT_PREFERENCE,
        },
        {
            "content": {"metric_key": "gmv", "patch": {"expression": "x" * 2001}},
            "memory_type": MemoryType.METRIC_OVERRIDE,
        },
    ],
)
async def test_invalid_write_rejected_without_partial_row(
    db_session: AsyncSession, update: dict[str, object]
) -> None:
    user_id, turn_id = await memory_owner(db_session)
    repo = MemoryRepository(db_session, user_id)
    forged = memory_input(turn_id).model_copy(update=update)
    with capture_logs() as logs, pytest.raises(ValidationError):
        await repo.create(forged)
    assert any(event["event"] == "memory_validation_failed" for event in logs)
    assert await repo.list_history(MemoryType.TERMINOLOGY) == []
    assert await db_session.scalar(
        select(MemoryRecord.id).where(MemoryRecord.user_id == user_id)
    ) is None


@pytest.mark.parametrize(("field", "size"), [("term", 51), ("means", 201)])
async def test_terminology_length_capped(
    db_session: AsyncSession, field: str, size: int
) -> None:
    user_id, turn_id = await memory_owner(db_session)
    value = memory_input(turn_id)
    setattr(value.content, field, "private-memory-payload" * size)
    with capture_logs() as logs, pytest.raises(ValidationError):
        await MemoryRepository(db_session, user_id).create(value)
    assert "private-memory-payload" not in str(logs)
    assert await MemoryRepository(db_session, user_id).list_history(MemoryType.TERMINOLOGY) == []


async def test_superseded_memory_retains_row(db_session: AsyncSession) -> None:
    user_id, turn_id = await memory_owner(db_session)
    repo = MemoryRepository(db_session, user_id)
    old = await repo.create(memory_input(turn_id))
    new = await repo.create(memory_input(turn_id, content={"term": "大促", "means": "双11"}))
    retired = await repo.supersede(old.id, by=new.id)
    db_session.expunge_all()
    assert await repo.get(old.id) == retired
    assert retired.content == old.content
    assert retired.summary == old.summary
    assert retired.confidence == old.confidence
    assert retired.source_turn_id == old.source_turn_id
    assert retired.created_at == old.created_at
    assert retired.superseded_at.tzinfo is not None
    assert not retired.is_active
    assert retired.superseded_by == new.id
    assert await repo.list_active(MemoryType.TERMINOLOGY) == [new]
    assert {row.id for row in await repo.list_history(MemoryType.TERMINOLOGY)} == {old.id, new.id}


async def test_user_scoped_repository_isolates(db_session: AsyncSession) -> None:
    user_a, turn_a = await memory_owner(db_session)
    user_b, turn_b = await memory_owner(db_session)
    repo_a, repo_b = MemoryRepository(db_session, user_a), MemoryRepository(db_session, user_b)
    first = await repo_a.create(memory_input(turn_a))
    other = await repo_b.create(memory_input(turn_b))
    assert await repo_a.get(other.id) is None
    assert await repo_b.get(first.id) is None
    assert await repo_a.list_active(MemoryType.TERMINOLOGY) == [first]
    assert await repo_b.list_history(MemoryType.TERMINOLOGY) == [other]
    assert await repo_a.list_active(MemoryType.REGION_FOCUS) == []
    with pytest.raises(NotFoundError):
        await repo_a.create(memory_input(turn_b))
    with pytest.raises(NotFoundError):
        await repo_a.create(memory_input(uuid4()))
    with pytest.raises(NotFoundError):
        await repo_a.supersede(first.id, by=other.id)
    with pytest.raises(NotFoundError):
        await repo_a.supersede(other.id, by=first.id)
    assert (await repo_a.get(first.id)).is_active
    assert (await repo_b.get(other.id)).is_active


async def test_invalid_supersession_leaves_history_unchanged(db_session: AsyncSession) -> None:
    user_id, turn_id = await memory_owner(db_session)
    repo = MemoryRepository(db_session, user_id)
    first = await repo.create(memory_input(turn_id))
    second = await repo.create(memory_input(turn_id))
    third = await repo.create(memory_input(turn_id))
    region = await repo.create(
        memory_input(turn_id, kind=MemoryType.REGION_FOCUS, content={"region_ids": [1]})
    )
    with pytest.raises(ConflictError):
        await repo.supersede(first.id, by=first.id)
    with pytest.raises(ConflictError):
        await repo.supersede(first.id, by=region.id)
    retired = await repo.supersede(first.id, by=second.id)
    with pytest.raises(ConflictError):
        await repo.supersede(first.id, by=third.id)
    with pytest.raises(ConflictError):
        await repo.supersede(second.id, by=first.id)
    assert await repo.get(first.id) == retired
    assert await repo.get(second.id) == second
    assert await repo.get(third.id) == third


async def test_supersession_rollback_preserves_original(db_session: AsyncSession) -> None:
    user_id, turn_id = await memory_owner(db_session)
    repo = MemoryRepository(db_session, user_id)
    first = await repo.create(memory_input(turn_id))
    transaction = await db_session.begin_nested()
    replacement = await repo.create(memory_input(turn_id))
    await repo.supersede(first.id, by=replacement.id)
    await transaction.rollback()
    db_session.expunge_all()
    assert await repo.get(first.id) == first
    assert await repo.get(replacement.id) is None


async def test_unknown_type_rejected_by_database(db_session: AsyncSession) -> None:
    user_id, turn_id = await memory_owner(db_session)
    values = raw_memory(user_id, turn_id)
    values["memory_type"] = "world_knowledge"
    with pytest.raises(IntegrityError):
        await db_session.execute(insert(MemoryRecord).values(**values))


async def test_invalid_stored_content_is_not_returned(db_session: AsyncSession) -> None:
    user_id, turn_id = await memory_owner(db_session)
    values = raw_memory(user_id, turn_id)
    values["content"] = {"term": "large", "means": "private-memory-payload" * 201}
    await db_session.execute(insert(MemoryRecord).values(**values))
    repo = MemoryRepository(db_session, user_id)
    with capture_logs() as logs, pytest.raises(ValidationError):
        await repo.get(values["id"])
    assert "private-memory-payload" not in str(logs)
    assert any(event["event"] == "memory_record_invalid" for event in logs)
