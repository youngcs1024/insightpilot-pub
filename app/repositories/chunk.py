"""Current chunk registration and fail-closed candidate admission."""

from uuid import UUID

from pydantic import ValidationError
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import IngestionRegistryError
from app.db.models.chunk import Chunk
from app.db.models.document import CorpusManifestRecord, Document
from app.schemas.ingestion import ActiveManifest, ChunkIdentity, RegisteredChunk


class ChunkRepository:
    """A service owns commits; answer evidence never has a cascading chunk reference."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def replace(self, identifier: UUID, values: list[RegisteredChunk]) -> None:
        """Replace this document's complete active registry atomically."""
        await self.session.execute(delete(Chunk).where(Chunk.document_id == identifier))
        for value in values:
            self.session.add(
                Chunk(
                    id=value.chunk_uuid,
                    document_id=value.document_id,
                    document_version=value.document_version,
                    chunking_version=value.chunking_version,
                    ordinal=value.ordinal,
                    heading_path=value.heading_path,
                    page=value.page,
                    content_sha256=value.content_sha256,
                    char_len=value.char_len,
                    milvus_pk=value.milvus_pk,
                )
            )
        await self.session.flush()

    async def list_document(self, identifier: UUID) -> list[RegisteredChunk]:
        """Return detached, position-ordered current provenance."""
        return await self._list(identifier)

    async def list_all(self) -> list[RegisteredChunk]:
        """Read the complete shared registry for cross-store maintenance."""
        return await self._list(None)

    async def _list(self, identifier: UUID | None) -> list[RegisteredChunk]:
        query = select(Chunk).order_by(Chunk.document_id, Chunk.ordinal)
        if identifier is not None:
            query = query.where(Chunk.document_id == identifier)
        rows = (await self.session.scalars(query)).all()
        return [
            RegisteredChunk(
                chunk_uuid=row.id,
                document_id=row.document_id,
                document_version=row.document_version,
                chunking_version=row.chunking_version,
                ordinal=row.ordinal,
                heading_path=row.heading_path,
                page=row.page,
                content_sha256=row.content_sha256,
                char_len=row.char_len,
                milvus_pk=row.milvus_pk,
            )
            for row in rows
        ]

    async def admit(self, candidates: list[ChunkIdentity]) -> list[ChunkIdentity]:
        """Read manifest and registry in one snapshot, checking physical and logical IDs."""
        if not candidates:
            return []
        rows = (
            await self.session.execute(
                select(Chunk, CorpusManifestRecord.payload)
                .join(Document, Document.id == Chunk.document_id)
                .join(CorpusManifestRecord, CorpusManifestRecord.id == 1)
                .where(
                    Chunk.id.in_([item.chunk_uuid for item in candidates]),
                    Document.status == "active",
                    Chunk.document_version == Document.document_version,
                    Chunk.chunking_version == Document.chunking_version,
                )
            )
        ).all()
        allowed = set()
        try:
            for row, payload in rows:
                manifest = ActiveManifest.model_validate_json(payload)
                if any(
                    member.document_id == row.document_id
                    and member.document_version == row.document_version
                    and member.chunking_version == row.chunking_version
                    for member in manifest.members
                ):
                    allowed.add(
                        (
                            row.id,
                            row.document_id,
                            row.document_version,
                            row.chunking_version,
                            row.content_sha256,
                            row.milvus_pk,
                        )
                    )
        except ValidationError as exc:
            raise IngestionRegistryError() from exc
        return [
            item
            for item in candidates
            if item.milvus_pk is not None
            and (
                item.chunk_uuid,
                item.document_id,
                item.document_version,
                item.chunking_version,
                item.content_sha256,
                item.milvus_pk,
            )
            in allowed
        ]
