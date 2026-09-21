"""Capability reads require published membership and retain typed failures/deadlines."""

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config_models import Settings
from app.core.deadline import Deadline
from app.core.errors import DeadlineExceededError, MetricCatalogError
from app.db.session import Database
from app.schemas.corpus import DocumentType
from app.schemas.ingestion import DocumentStatus
from app.services.clarification_capabilities import ClarificationCapabilityService
from tests.agents.support import context
from tests.clarification_support import published_document, published_manifest
from tests.metric_resolution_support import definition


@pytest.mark.parametrize("publication", ["active", "deleted", "different_version", "empty", "unpublished"])
async def test_only_committed_active_capabilities_are_listed(
    settings: Settings, monkeypatch: pytest.MonkeyPatch, publication: str
) -> None:
    database = MagicMock(spec=Database)
    session = AsyncMock(spec=AsyncSession)
    database.session.return_value.__aenter__.return_value = session
    item = published_document()
    manifest = published_manifest(item)
    if publication == "deleted":
        item.status = DocumentStatus.DELETED
    elif publication == "different_version":
        item.document_version = "d" * 64
    elif publication == "empty":
        item.chunk_count = 0
    metrics = AsyncMock()
    metrics.list_active.return_value = [definition("gmv")]
    documents = AsyncMock()
    documents.manifest.return_value = None if publication == "unpublished" else manifest
    documents.list_documents.return_value = [item]
    monkeypatch.setattr("app.services.clarification_capabilities.MetricRepository", lambda _: metrics)
    monkeypatch.setattr("app.services.clarification_capabilities.DocumentRepository", lambda _: documents)
    output = await ClarificationCapabilityService(database, settings.database).read(deadline=context().deadline)
    assert [item.key for item in output.metrics] == ["gmv"]
    assert output.document_categories == ([DocumentType.POLICY] if publication == "active" else [])
    if publication == "unpublished":
        documents.list_documents.assert_not_awaited()


async def test_catalog_error_is_not_relabelled_as_empty(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = MagicMock(spec=Database)
    database.session.return_value.__aenter__.return_value = AsyncMock(spec=AsyncSession)
    metrics = AsyncMock()
    metrics.list_active.side_effect = MetricCatalogError()
    monkeypatch.setattr("app.services.clarification_capabilities.MetricRepository", lambda _: metrics)
    with pytest.raises(MetricCatalogError):
        await ClarificationCapabilityService(database, settings.database).read(deadline=context().deadline)
    metrics.list_active.assert_awaited_once()


async def test_capability_deadline_closes_session(settings: Settings) -> None:
    closed = False
    database = MagicMock(spec=Database)

    @asynccontextmanager
    async def blocked() -> AsyncIterator[None]:
        nonlocal closed
        try:
            await asyncio.Event().wait()
            yield
        finally:
            closed = True

    database.session.side_effect = blocked
    ctx = context()

    with pytest.raises(DeadlineExceededError):
        await ClarificationCapabilityService(database, settings.database).read(
            deadline=Deadline(time.monotonic() + 0.01)
        )
    assert closed
    assert not ctx.mcp.calls and not ctx.llm.calls
