"""Startup contract comparison and bounded transaction retries without driver prose."""

# ruff: noqa: PLR2004 -- fixed acceptance counts and retry/timeout boundaries.

from unittest.mock import AsyncMock, MagicMock

import pytest
from asyncpg import UniqueViolationError
from sqlalchemy.exc import IntegrityError

from app.core.config_models import Settings
from app.core.errors import ConflictError, MemorySchemaError, MemoryWriteConflictError
from app.db.session import translate_database_error
from app.repositories.memory_schema import MemorySchemaStatus
from app.schemas.memory_extraction import MemoryExtraction
from app.services.memory.extract import MemoryExtractionService
from app.services.memory.integrity import canonical_expression, validate_memory_schema


def test_generated_expression_accepts_postgresql_catalog_casts() -> None:
    authored = "CASE WHEN is_active AND memory_type = 'metric_override' THEN content ->> 'metric_key' ELSE NULL END"
    catalog = "CASE WHEN (is_active AND ((memory_type)::text = 'metric_override'::text)) THEN (content ->> 'metric_key'::text) ELSE NULL::text END"
    assert canonical_expression(authored) == canonical_expression(catalog)
    assert canonical_expression(authored) == canonical_expression(
        f"({catalog})::character varying(64)"
    )
    assert canonical_expression(authored) != canonical_expression(
        authored.replace("is_active", "NOT is_active")
    )


@pytest.mark.parametrize("wrapped", [True, False])
def test_only_named_uniqueness_is_retryable(wrapped: bool) -> None:
    driver = UniqueViolationError("private prose")
    driver.constraint_name = "uq_memories_active_metric"
    if wrapped:
        adapter = Exception("unrelated prose")
        adapter.sqlstate = "23505"
        adapter.__cause__ = driver
        error = IntegrityError("", None, adapter)
    else:
        error = driver
    assert isinstance(translate_database_error(error), MemoryWriteConflictError)
    driver.constraint_name = "some_other_unique_constraint"
    assert type(translate_database_error(error)) is ConflictError


@pytest.mark.parametrize("failures", [1, 3])
async def test_only_known_rolled_back_transactions_retry(settings: Settings, failures: int) -> None:
    service = MemoryExtractionService(AsyncMock(), settings, AsyncMock())
    results = [MemoryWriteConflictError() for _ in range(failures)] + [[]]
    service._write_once = AsyncMock(side_effect=results)
    identity = AsyncMock()
    if failures == 3:
        with pytest.raises(MemoryWriteConflictError):
            await service._write(identity, MemoryExtraction())
    else:
        assert await service._write(identity, MemoryExtraction()) == []
    assert service._write_once.await_count == min(failures + 1, 3)


async def test_ordinary_conflict_is_not_retried(settings: Settings) -> None:
    service = MemoryExtractionService(AsyncMock(), settings, AsyncMock())
    service._write_once = AsyncMock(side_effect=ConflictError())
    with pytest.raises(ConflictError):
        await service._write(AsyncMock(), MemoryExtraction())
    service._write_once.assert_awaited_once()


@pytest.mark.parametrize(
    ("valid", "expression"),
    [
        (False, ""),
        (True, "CASE WHEN is_active THEN 'wrong' ELSE NULL END"),
        (True, "SELECT ("),
    ],
)
async def test_memory_schema_drift_is_typed(
    monkeypatch: pytest.MonkeyPatch, valid: bool, expression: str
) -> None:
    database = MagicMock()
    database.session.return_value.__aenter__ = AsyncMock()
    database.session.return_value.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(
        "app.services.memory.integrity.memory_schema_status",
        AsyncMock(return_value=MemorySchemaStatus(valid=valid, expression=expression)),
    )
    with pytest.raises(MemorySchemaError):
        await validate_memory_schema(database, timeout_s=1)
