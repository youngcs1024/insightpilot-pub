"""Locked consistency inspection and crash-safe, identity-preserving index repair."""

import asyncio
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from uuid import UUID

import structlog
from pydantic import ValidationError

from app.core.deadline import Deadline
from app.core.errors import IngestionAlreadyRunning, IngestionRegistryError, InsightPilotError
from app.db.session import Database
from app.repositories.chunk import ChunkRepository
from app.repositories.document import DocumentRepository
from app.retrieval.consistency_store import ConsistencyStore
from app.schemas.consistency import (
    Assessment,
    ConsistencyReport,
    IndexSnapshot,
    RegistrySnapshot,
    RepairBlock,
)
from app.schemas.ingestion import (
    ActiveManifest,
    DocumentStatus,
    PreparedDocument,
    RegisteredChunk,
    VectorRow,
)
from app.services.consistency_check import assess, validate_registry
from app.services.consistency_plan import RepairPlan, stage_repair
from app.services.ingestion import insert_vectors
from app.services.ingestion_config import IngestionSettings

logger = structlog.get_logger(__name__)
EncodeRepair = Callable[
    [list[PreparedDocument], ActiveManifest, Deadline], Awaitable[list[VectorRow]]
]
INSERT_BATCH_SIZE = 16
UUID_ZERO = UUID(int=0)


class ConsistencyService:
    """The caller owns storage lifetimes and supplies a lazy encoding operation."""

    def __init__(
        self,
        database: Database,
        store: ConsistencyStore,
        settings: IngestionSettings,
        encode: EncodeRepair | None = None,
    ) -> None:
        self.database = database
        self.store = store
        self.settings = settings
        self.encode = encode

    async def check(self, *, root: Path | None = None) -> ConsistencyReport:
        """A root opts into repair; checks never touch files or initialize a model."""
        deadline = Deadline(time.monotonic() + self.settings.timeout_s)
        async with (
            asyncio.timeout(self.settings.timeout_s),
            self.database.session() as guard,
            guard.begin(),
        ):
            if not await DocumentRepository(guard).try_lock():
                raise IngestionAlreadyRunning()
            registry = await self._registry()
            index = await self.store.scan()
            assessment = assess(registry, index)
            report = ConsistencyReport(
                corpus_version=registry.manifest.corpus_version if registry.manifest else None,
                collection_exists=index.exists,
                fix_requested=root is not None,
                before=assessment.drift,
                remaining=assessment.drift,
                cleanup_pending=[
                    item.document_id for item in registry.documents if item.cleanup_pending
                ],
            )
            if root is not None and not report.successful:
                await self._repair(root, registry, assessment, report, deadline)
                final_registry = await self._registry()
                final_index = await self.store.scan()
                report.remaining = assess(final_registry, final_index).drift
                report.collection_exists = final_index.exists
                report.cleanup_pending = [
                    item.document_id for item in final_registry.documents if item.cleanup_pending
                ]
            logger.info(
                "consistency_checked",
                fix=report.fix_requested,
                before=len(report.before),
                remaining=len(report.remaining),
                blocked=len(report.blocked),
                inserted=report.chunks_inserted,
                deleted=report.chunks_deleted,
                successful=report.successful,
            )
            return report

    async def _registry(self) -> RegistrySnapshot:
        async with self.database.session() as session, session.begin():
            try:
                registry = RegistrySnapshot(
                    documents=await DocumentRepository(session).list_documents(),
                    chunks=await ChunkRepository(session).list_all(),
                    manifest=await DocumentRepository(session).manifest(),
                )
            except ValidationError as exc:
                raise IngestionRegistryError() from exc
        validate_registry(registry, self.store.settings.collection)
        return registry

    async def _repair(
        self,
        root: Path,
        registry: RegistrySnapshot,
        assessment: Assessment,
        report: ConsistencyReport,
        deadline: Deadline,
    ) -> None:
        active = {
            item.document_id
            for item in registry.documents
            if item.status is DocumentStatus.ACTIVE
        }
        plan = await asyncio.to_thread(
            stage_repair, root, registry, assessment.rebuild & active, self.settings
        )
        report.blocked.extend(plan.blocked)
        if plan.prepared:
            rows = await self._encode(plan, registry, report, deadline)
            if rows is None:
                return
            await self.store.ensure_collection()
            await insert_vectors(self.store, rows, INSERT_BATCH_SIZE)
            await self._publish(plan, registry)
            report.chunks_inserted = len(rows)
            report.documents_repaired = len(plan.prepared)
        await self._retire_damaged_tombstones(registry, assessment)
        current = await self._registry()
        await self._cleanup(current, await self.store.scan(), report)

    async def _encode(
        self,
        plan: RepairPlan,
        registry: RegistrySnapshot,
        report: ConsistencyReport,
        deadline: Deadline,
    ) -> list[VectorRow] | None:
        if self.encode is None or registry.manifest is None:
            report.blocked.extend(
                RepairBlock(document_id=item.document.document_id, code="MODEL_NOT_CONFIGURED")
                for item in plan.prepared
            )
            return None
        try:
            return await self.encode(plan.prepared, registry.manifest, deadline)
        except InsightPilotError as exc:
            logger.exception("consistency_encoding_failed", code=exc.code)
            report.blocked.extend(
                RepairBlock(document_id=item.document.document_id, code=exc.code)
                for item in plan.prepared
            )
            return None

    async def _publish(self, plan: RepairPlan, registry: RegistrySnapshot) -> None:
        """The active manifest is deliberately unchanged, including its frozen configs."""
        async with self.database.session() as session, session.begin():
            documents, chunks = DocumentRepository(session), ChunkRepository(session)
            for item in plan.prepared:
                await documents.save(item.document)
                await chunks.replace(
                    item.document.document_id,
                    [
                        RegisteredChunk.model_validate(
                            chunk.model_dump(include=set(RegisteredChunk.model_fields))
                        )
                        for chunk in item.chunks
                    ],
                )
            if await documents.manifest() != registry.manifest:
                raise IngestionRegistryError()
        logger.info("consistency_registry_committed", documents=len(plan.prepared))

    async def _retire_damaged_tombstones(
        self, registry: RegistrySnapshot, assessment: Assessment
    ) -> None:
        retired = [
            item
            for item in registry.documents
            if item.status is DocumentStatus.DELETED and item.document_id in assessment.rebuild
        ]
        if not retired:
            return
        async with self.database.session() as session, session.begin():
            for document in retired:
                await ChunkRepository(session).replace(document.document_id, [])
                await DocumentRepository(session).save(
                    document.model_copy(update={"chunk_count": 0, "cleanup_pending": True})
                )

    async def _cleanup(
        self, registry: RegistrySnapshot, index: IndexSnapshot, report: ConsistencyReport
    ) -> None:
        keep = {chunk.milvus_pk for chunk in registry.chunks}
        blocked = {item.document_id for item in report.blocked}
        stale = [
            row.milvus_pk
            for row in index.rows
            if row.milvus_pk not in keep and row.document_id not in blocked
        ]
        try:
            report.chunks_deleted += await self.store.delete_observed(stale)
            # Acknowledged deletion is not proof; retain retry flags on silent no-ops.
            remaining = {row.milvus_pk for row in (await self.store.scan()).rows}
            if remaining.intersection(stale):
                return
            async with self.database.session() as session, session.begin():
                for document in registry.documents:
                    if document.cleanup_pending and document.document_id not in blocked:
                        await DocumentRepository(session).finish_cleanup(document.document_id)
        except InsightPilotError as exc:
            logger.exception("consistency_cleanup_failed", code=exc.code)
            # Orphans may have no registry row on which to retain a cleanup flag.
            report.blocked.append(RepairBlock(document_id=UUID_ZERO, code=exc.code))
