"""Real storage fixtures and deterministic HTTP encoding for consistency acceptance."""

import time
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import UUID

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import select

from app.core.config_models import ModelRuntimeClientSettings
from app.core.deadline import Deadline
from app.db.models.chunk import Chunk
from app.db.session import Database
from app.retrieval.consistency_store import ConsistencyStore
from app.schemas.ingestion import ActiveManifest, PreparedDocument, VectorRow
from app.services.consistency import ConsistencyService
from app.services.ingestion_config import IngestionSettings
from app.services.ingestion_plan import IngestionPlan, stage
from tests.ingestion_support import Embeddings, Harness, clear_registry, write_sources
from tests.milvus_support import MilvusStack
from tests.shared_database import TestPostgres


class ConsistencyHarness(Harness):
    """Both services share the same real application database and physical index."""

    store: ConsistencyStore

    def checker(self, *, models: bool = True) -> ConsistencyService:
        async def encode(
            prepared: list[PreparedDocument], manifest: ActiveManifest, deadline: Deadline
        ) -> list[VectorRow]:
            rows, _ = await self.service().encode(
                IngestionPlan(changed=prepared), manifest, deadline
            )
            return rows

        return ConsistencyService(
            self.database, self.store, IngestionSettings(), encode if models else None
        )

    async def vectors(self) -> list[VectorRow]:
        """Reproduce current payloads for deliberate corruption inside an isolated index."""
        plan = stage(self.root, [], IngestionSettings())
        rows, _ = await self.service().encode(plan, None, Deadline(time.monotonic() + 30))
        return rows

    async def point_to(self, identifier: UUID, pk: int | None) -> None:
        async with self.database.session() as session, session.begin():
            row = await session.scalar(select(Chunk).where(Chunk.id == identifier))
            row.milvus_pk = pk


@pytest.fixture
async def consistency_harness(
    migrated_db: TestPostgres,
    milvus_stack: MilvusStack,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> AsyncIterator[ConsistencyHarness]:
    database = Database(migrated_db.app)
    database.start()
    embeddings = Embeddings()
    settings = ModelRuntimeClientSettings(auth_token=SecretStr("synthetic-consistency-token"))
    write_sources(tmp_path)
    try:
        await clear_registry(database)
        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(embeddings.handle)) as http,
            milvus_stack.collection("step35", request) as storage,
            ConsistencyStore(storage) as store,
        ):
            yield ConsistencyHarness(database, store, http, embeddings, tmp_path, settings)
    finally:
        await clear_registry(database)
        await database.aclose()
