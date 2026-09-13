"""Transaction-neutral persistence of the shared corpus and cleanup tombstones."""

from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import IngestionRegistryError
from app.db.models.document import CorpusManifestRecord, Document
from app.schemas.corpus import CorpusMetadata
from app.schemas.ingestion import ActiveManifest, DocumentStatus, RegisteredDocument


class DocumentRepository:
    """No user scope is required for globally published business reference sources."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def try_lock(self) -> bool:
        """Hold a transaction lock on a dedicated session through post-commit cleanup."""
        return bool(
            await self.session.scalar(
                text("SELECT pg_try_advisory_xact_lock(hashtextextended(:key, 0))"),
                {"key": "insightpilot:corpus:ingestion"},
            )
        )

    async def list_documents(self) -> list[RegisteredDocument]:
        """Validate persisted fields before using them for decisions or cleanup."""
        rows = (await self.session.scalars(select(Document).order_by(Document.source_path))).all()
        try:
            return [
                RegisteredDocument(
                    document_id=row.id,
                    source_path=row.source_path,
                    source_fingerprint=row.source_fingerprint,
                    content_sha256=row.content_sha256,
                    document_version=row.document_version,
                    chunking_version=row.chunking_version,
                    metadata=CorpusMetadata.model_validate_json(row.business_metadata),
                    chunk_count=row.chunk_count,
                    status=DocumentStatus(row.status),
                    cleanup_pending=row.cleanup_pending,
                )
                for row in rows
            ]
        except (ValidationError, ValueError) as exc:
            raise IngestionRegistryError() from exc

    async def manifest(self) -> ActiveManifest | None:
        """Read a fully committed pointer; never infer a manifest from vector rows."""
        row = await self.session.get(CorpusManifestRecord, 1)
        if row is None:
            return None
        try:
            manifest = ActiveManifest.model_validate_json(row.payload)
        except ValidationError as exc:
            raise IngestionRegistryError() from exc
        if manifest.corpus_version != row.corpus_version:
            raise IngestionRegistryError()
        return manifest

    async def save(self, value: RegisteredDocument) -> None:
        """Stage a source replacement or tombstone inside the service transaction."""
        row = await self.session.get(Document, value.document_id)
        if row is None:
            row = Document(id=value.document_id)
            self.session.add(row)
        row.source_path = value.source_path
        row.source_fingerprint = value.source_fingerprint
        row.content_sha256 = value.content_sha256
        row.document_version = value.document_version
        row.chunking_version = value.chunking_version
        row.business_metadata = value.metadata.model_dump_json()
        row.chunk_count = value.chunk_count
        row.status = value.status.value
        row.cleanup_pending = value.cleanup_pending
        row.ingested_at = (await self.session.execute(select(func.now()))).scalar_one()
        await self.session.flush()

    async def save_manifest(self, value: ActiveManifest) -> None:
        """Publish the pointer in the same transaction as all registry changes."""
        row = await self.session.get(CorpusManifestRecord, 1)
        if row is None:
            row = CorpusManifestRecord(id=1)
            self.session.add(row)
        row.corpus_version = value.corpus_version
        row.payload = value.model_dump_json()
        await self.session.flush()

    async def finish_cleanup(self, identifier: UUID) -> None:
        """A failure to commit this flag merely causes another idempotent cleanup."""
        await self.session.execute(
            update(Document).where(Document.id == identifier).values(cleanup_pending=False)
        )
