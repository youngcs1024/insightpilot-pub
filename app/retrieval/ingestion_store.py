"""Typed Milvus writes and bounded per-document stale cleanup."""

from datetime import date
from functools import partial
from uuid import UUID

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from app.core.errors import RetrievalUnavailableError
from app.retrieval.milvus_repo import MilvusRepository
from app.schemas.ingestion import ChunkIdentity, InsertReceipt, VectorRow

EPOCH = date(1970, 1, 1)


class SdkInsert(BaseModel):
    """Validate count and ordered auto-generated keys before committing the registry."""

    insert_count: int = Field(strict=True, ge=0)
    ids: list[int]


class PhysicalRow(BaseModel):
    """Only identities cross the storage maintenance boundary."""

    pk: int = Field(strict=True, ge=0)
    chunk_uuid: UUID
    document_id: UUID
    document_version: str
    chunking_version: str
    content_sha256: str


class IngestionStore(MilvusRepository):
    """No retry of inserts whose outcome may be unknown; operator replay is explicit."""

    async def insert(self, values: list[VectorRow]) -> InsertReceipt:
        """Insert a single complete batch, requesting no client-generated BM25 vector."""
        data = []
        for value in values:
            item = value.chunk
            data.append(
                {
                    "chunk_uuid": str(item.chunk_uuid),
                    "document_id": str(item.document_id),
                    "document_version": item.document_version,
                    "chunking_version": item.chunking_version,
                    "content_sha256": item.content_sha256,
                    "heading_path": item.heading_path,
                    "content": item.content,
                    "parent_content": item.parent_content,
                    "doc_type": value.metadata.doc_type.value,
                    "effective_from": (value.metadata.effective_from - EPOCH).days,
                    "effective_to": (value.metadata.effective_to - EPOCH).days
                    if value.metadata.effective_to
                    else -1,
                    "dense": value.dense,
                    "sparse_learned": value.sparse_learned,
                }
            )
        raw = await self._rpc(
            "insert",
            lambda: self._client.insert(
                self.settings.collection,
                data=data,
                timeout=None,
                retry_times=0,
                retry_on_rate_limit=False,
            ),
        )
        try:
            result = SdkInsert.model_validate(raw)
            receipt = InsertReceipt(ids=result.ids)
        except ValidationError as exc:
            raise RetrievalUnavailableError(operation="insert_receipt") from exc
        if (
            result.insert_count != len(values)
            or len(receipt.ids) != len(values)
            or len(set(receipt.ids)) != len(values)
        ):
            raise RetrievalUnavailableError(operation="insert_count")
        return receipt

    async def identities(self, identifier: UUID) -> list[ChunkIdentity]:
        """Strong exclusion pages do not assume server-version-dependent result ordering."""
        result: list[ChunkIdentity] = []
        seen: set[int] = {-1}
        while True:
            raw = await self._rpc(
                "query_document",
                lambda: self._client.query(
                    self.settings.collection,
                    filter="document_id == {document} and pk not in {seen}",
                    filter_params={"document": str(identifier), "seen": sorted(seen)},
                    output_fields=[
                        "pk",
                        "chunk_uuid",
                        "document_id",
                        "document_version",
                        "chunking_version",
                        "content_sha256",
                    ],
                    limit=1000,
                    consistency_level="Strong",
                    timeout=None,
                    retry_times=0,
                    retry_on_rate_limit=False,
                ),
            )
            try:
                rows = TypeAdapter(list[PhysicalRow]).validate_python(raw)
                batch = [
                    ChunkIdentity(
                        chunk_uuid=row.chunk_uuid,
                        document_id=row.document_id,
                        document_version=row.document_version,
                        chunking_version=row.chunking_version,
                        content_sha256=row.content_sha256,
                        milvus_pk=row.pk,
                    )
                    for row in rows
                ]
            except ValidationError as exc:
                raise RetrievalUnavailableError(operation="query_identity") from exc
            if not batch:
                return result
            keys = {row.pk for row in rows}
            if (
                keys & seen
                or len(keys) != len(rows)
                or any(row.document_id != identifier for row in rows)
            ):
                raise RetrievalUnavailableError(operation="query_progress")
            result.extend(batch)
            seen.update(keys)

    async def cleanup(self, identifier: UUID, keep: list[int]) -> int:
        """Delete only observed stale keys; failed calls leave the durable retry marker."""
        rows = await self.identities(identifier)
        current = set(keep)
        stale = [row.milvus_pk for row in rows if row.milvus_pk not in current]
        for offset in range(0, len(stale), 1000):
            await self._rpc(
                "delete_stale",
                partial(
                    self._client.delete,
                    self.settings.collection,
                    ids=stale[offset : offset + 1000],
                    timeout=None,
                    retry_times=0,
                    retry_on_rate_limit=False,
                ),
            )
        return len(stale)
