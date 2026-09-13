"""Reusable ingestion fixtures; imports never start stores or model runtimes."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

import httpx
from sqlalchemy import delete

from app.clients.model_runtime import ModelRuntimeClient
from app.core.config_models import ModelRuntimeClientSettings
from app.db.models.chunk import Chunk
from app.db.models.document import CorpusManifestRecord, Document
from app.db.session import Database
from app.repositories.chunk import ChunkRepository
from app.repositories.document import DocumentRepository
from app.retrieval.ingestion_store import IngestionStore
from app.schemas.ingestion import RegisteredChunk, RegisteredDocument
from app.schemas.model_runtime import EMBED_REVISION, RERANK_REVISION, EmbedResult, ModelMetadata
from app.services.ingestion import IngestionService
from app.services.ingestion_config import IngestionSettings
from tests.corpus_support import entry, markdown_source, write_inventory


def model_metadata() -> ModelMetadata:
    """Use the exact transport identity with deterministic CPU-only embeddings."""
    return ModelMetadata(
        embed_revision=EMBED_REVISION, rerank_revision=RERANK_REVISION,
        precision="fp16", embed_batch=16, rerank_batch=16,
        embed_max_length=512, rerank_max_length=320,
    )


@dataclass
class Embeddings:
    """Real HTTP client contracts with a deterministic transport substitute."""

    calls: list[list[str]] = field(default_factory=list)
    fail_at: int | None = None
    metadata: ModelMetadata = field(default_factory=model_metadata)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["texts"]
        self.calls.append(texts)
        if self.fail_at == len(self.calls):
            raise httpx.ConnectError("synthetic outage", request=request)
        response = EmbedResult(
            request_id="test-ingestion", ms=0, queue_ms=0, inference_ms=0,
            metadata=self.metadata, dense=[[1.0, *([0.0] * 1023)] for _ in texts],
            sparse=[{sum(text.encode()) % 1000 + 1: 1.0} for text in texts],
        )
        return httpx.Response(200, json=response.model_dump(mode="json"))


def write_sources(root: Path, names: list[str] | None = None, *, body: str | None = None) -> None:
    """Only temporary synthetic sources are written; no file deletion is needed."""
    names = names if names is not None else ["rule.md", "other.md"]
    for name in names:
        (root / name).write_text(markdown_source(body=body or "# 政策\n\n## 退款\n\n七天退款规则。\n\n### 条件\n\n完整保留订单证据。"))
    write_inventory(root, [entry(name) for name in names])


async def clear_registry(database: Database) -> None:
    """Reset only registry rows in the fixture-owned PostgreSQL, with real commits."""
    async with database.session() as session, session.begin():
        await session.execute(delete(Chunk))
        await session.execute(delete(Document))
        await session.execute(delete(CorpusManifestRecord))


@dataclass
class Harness:
    """Test resources keep real transactions and separate reader connections."""

    database: Database
    store: IngestionStore
    client: httpx.AsyncClient
    embeddings: Embeddings
    root: Path
    model_settings: ModelRuntimeClientSettings

    def service(self, settings: IngestionSettings | None = None) -> IngestionService:
        return IngestionService(
            self.database, self.store, ModelRuntimeClient(self.model_settings, self.client),
            settings or IngestionSettings(), self.model_settings,
        )

    async def documents(self) -> list[RegisteredDocument]:
        async with self.database.session() as session:
            return await DocumentRepository(session).list_documents()

    async def chunks(self, identifier: UUID) -> list[RegisteredChunk]:
        async with self.database.session() as session:
            return await ChunkRepository(session).list_document(identifier)
