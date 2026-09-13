"""Single-operator ingestion with atomic publication and durable stale cleanup."""

import asyncio
import time
from pathlib import Path

import structlog

from app.clients.model_runtime import ModelRuntimeClient
from app.core.config_models import ModelRuntimeClientSettings
from app.core.deadline import Deadline
from app.core.errors import (
    IngestionAlreadyRunning,
    IngestionConfigurationError,
    InsightPilotError,
)
from app.db.session import Database
from app.repositories.chunk import ChunkRepository
from app.repositories.document import DocumentRepository
from app.retrieval.ingestion_store import IngestionStore
from app.schemas.ingestion import (
    ActiveManifest,
    DocumentStatus,
    EncodingProfile,
    IngestionResult,
    RegisteredChunk,
    RegisteredDocument,
    VectorRow,
    VersionMember,
    canonical,
    digest,
)
from app.schemas.model_runtime import EmbedMode, ModelMetadata
from app.services.ingestion_config import IngestionSettings
from app.services.ingestion_plan import IngestionPlan, stage

logger = structlog.get_logger(__name__)


class IngestionService:
    """Resources are injected; the CLI owns their lifetime and no API route is added."""

    def __init__(
        self,
        database: Database,
        store: IngestionStore,
        model: ModelRuntimeClient,
        settings: IngestionSettings,
        model_settings: ModelRuntimeClientSettings,
    ) -> None:
        self.database = database
        self.store = store
        self.model = model
        self.settings = settings
        self.model_settings = model_settings

    async def ingest(self, root: Path) -> IngestionResult:
        """Keep the guard transaction alive while another connection commits the registry."""
        deadline = Deadline(time.monotonic() + self.settings.timeout_s)
        async with (
            asyncio.timeout(self.settings.timeout_s),
            self.database.session() as guard,
            guard.begin(),
        ):
            if not await DocumentRepository(guard).try_lock():
                raise IngestionAlreadyRunning()
            logger.info("ingestion_started")
            result = await self._ingest(root, deadline)
            logger.info(
                "ingestion_finished",
                changed=result.documents_changed,
                inserted=result.chunks_inserted,
                deleted=result.chunks_deleted,
                failed=len(result.failed_files),
                pending=len(result.cleanup_pending),
            )
            return result

    async def _ingest(self, root: Path, deadline: Deadline) -> IngestionResult:
        async with self.database.session() as session:
            repository = DocumentRepository(session)
            previous = await repository.list_documents()
            manifest = await repository.manifest()
        if manifest is not None and manifest.collection != self.store.settings.collection:
            raise IngestionConfigurationError()
        plan = await asyncio.to_thread(stage, root, previous, self.settings)
        result = IngestionResult(
            failed_files=plan.failed, corpus_version=manifest.corpus_version if manifest else None
        )
        rows, metadata = await self._encode(plan, manifest, deadline)
        if rows:
            await self.store.ensure_collection()
            await self._insert(rows)
        if plan.changed or plan.deleted or plan.refreshed:
            manifest = await self._publish(plan, previous, manifest, metadata)
            result.corpus_version = manifest.corpus_version
            result.documents_changed = len(plan.changed)
            result.documents_deleted = len(plan.deleted)
            result.chunks_inserted = len(rows)
        await self._cleanup(result)
        return result

    async def _encode(
        self,
        plan: IngestionPlan,
        manifest: ActiveManifest | None,
        deadline: Deadline,
    ) -> tuple[list[VectorRow], ModelMetadata | None]:
        candidates = [
            (chunk, item.document.metadata) for item in plan.changed for chunk in item.chunks
        ]
        rows: list[VectorRow] = []
        metadata = manifest.model_metadata if manifest else None
        profile = manifest.encoding if manifest else None
        batch_size = self.model_settings.embed_batch
        for start in range(0, len(candidates), batch_size):
            batch = candidates[start : start + batch_size]
            response = await self.model.embed(
                [chunk.content for chunk, _ in batch], EmbedMode.DOCUMENT, deadline=deadline
            )
            actual = EncodingProfile.from_metadata(response.metadata)
            if profile is not None and actual != profile:
                raise IngestionConfigurationError()
            profile, metadata = actual, response.metadata
            rows.extend(
                VectorRow(chunk=chunk, metadata=business, dense=dense, sparse_learned=sparse)
                for (chunk, business), dense, sparse in zip(
                    batch, response.dense, response.sparse, strict=True
                )
            )
        return rows, metadata

    async def _insert(self, rows: list[VectorRow]) -> None:
        size = self.model_settings.embed_batch
        for start in range(0, len(rows), size):
            batch = rows[start : start + size]
            receipt = await self.store.insert(batch)
            for row, pk in zip(batch, receipt.ids, strict=True):
                row.chunk.milvus_pk = pk

    async def _publish(
        self,
        plan: IngestionPlan,
        previous: list[RegisteredDocument],
        old_manifest: ActiveManifest | None,
        metadata: ModelMetadata | None,
    ) -> ActiveManifest:
        documents = {item.document_id: item for item in previous}
        for value in [*plan.refreshed, *plan.deleted, *(item.document for item in plan.changed)]:
            documents[value.document_id] = value
        members = [
            VersionMember(
                document_id=item.document_id,
                document_version=item.document_version,
                chunking_version=item.chunking_version,
            )
            for item in sorted(documents.values(), key=lambda item: str(item.document_id))
            if item.status is DocumentStatus.ACTIVE
        ]
        configurations = dict(old_manifest.splitter_configs) if old_manifest else {}
        configurations[self.settings.chunking_version()] = self.settings.splitter_config()
        manifest = ActiveManifest(
            corpus_version=digest(
                canonical(
                    [
                        (str(item.document_id), item.document_version, item.chunking_version)
                        for item in members
                    ]
                )
            ),
            members=members,
            splitter_configs={
                key: value
                for key, value in configurations.items()
                if key in {item.chunking_version for item in members}
            },
            encoding=EncodingProfile.from_metadata(metadata) if metadata else None,
            model_metadata=metadata,
            collection=self.store.settings.collection,
        )
        # A partial splitter migration retains each member's own version; the config
        # describes this run, and failed members keep their previous registry version.
        async with self.database.session() as session, session.begin():
            repository, chunks = DocumentRepository(session), ChunkRepository(session)
            for value in [
                *plan.refreshed,
                *plan.deleted,
                *(item.document for item in plan.changed),
            ]:
                await repository.save(value)
            for value in plan.deleted:
                await chunks.replace(value.document_id, [])
            for prepared in plan.changed:
                await chunks.replace(
                    prepared.document.document_id,
                    [
                        RegisteredChunk.model_validate(
                            chunk.model_dump(include=set(RegisteredChunk.model_fields)),
                        )
                        for chunk in prepared.chunks
                    ],
                )
            await repository.save_manifest(manifest)
        logger.info(
            "ingestion_manifest_committed",
            corpus_version=manifest.corpus_version,
            members=len(members),
        )
        return manifest

    async def _cleanup(self, result: IngestionResult) -> None:
        async with self.database.session() as session:
            pending = [
                item
                for item in await DocumentRepository(session).list_documents()
                if item.cleanup_pending
            ]
        for document in pending:
            try:
                async with self.database.session() as session:
                    chunks = await ChunkRepository(session).list_document(document.document_id)
                keep = [chunk.milvus_pk for chunk in chunks if chunk.milvus_pk is not None]
                result.chunks_deleted += await self.store.cleanup(document.document_id, keep)
                async with self.database.session() as session, session.begin():
                    await DocumentRepository(session).finish_cleanup(document.document_id)
            except InsightPilotError:
                logger.exception("ingestion_cleanup_failed", document_id=str(document.document_id))
                result.cleanup_pending.append(document.document_id)
