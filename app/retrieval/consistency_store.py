"""Complete, bounded index inspection and exact-key maintenance operations."""

from functools import partial

from pydantic import Field, TypeAdapter, ValidationError

from app.core.errors import RetrievalUnavailableError
from app.retrieval.ingestion_store import IngestionStore, PhysicalRow
from app.schemas.consistency import IndexSnapshot, StoredChunk
from app.schemas.ingestion import digest

PAGE_SIZE = 1000


class InspectedRow(PhysicalRow):
    """SDK-only text projection; never log or export this object's contents."""

    content: str = Field(repr=False, max_length=65535)


class ConsistencyStore(IngestionStore):
    """Checking never initializes a collection or invents a missing index."""

    async def scan(self) -> IndexSnapshot:
        """Scan unknown documents too, without relying on SDK result ordering."""
        exists = await self._rpc(
            "has_collection",
            lambda: self._client.has_collection(
                self.settings.collection, timeout=None, retry_times=0, retry_on_rate_limit=False
            ),
        )
        if not isinstance(exists, bool):
            raise RetrievalUnavailableError(operation="collection_exists_response")
        if not exists:
            return IndexSnapshot(exists=False)
        await self.validate_collection()
        result = IndexSnapshot(exists=True)
        seen = {-1}
        while True:
            batch = await self._page(seen)
            if not batch:
                return result
            keys = {row.milvus_pk for row in batch}
            if keys & seen or len(keys) != len(batch):
                raise RetrievalUnavailableError(operation="consistency_scan_progress")
            result.rows.extend(batch)
            seen.update(keys)

    async def _page(self, seen: set[int]) -> list[StoredChunk]:
        raw = await self._rpc(
            "scan_consistency",
            lambda: self._client.query(
                self.settings.collection,
                filter="pk not in {seen}",
                filter_params={"seen": sorted(seen)},
                output_fields=[*PhysicalRow.model_fields, "content"],
                limit=PAGE_SIZE,
                consistency_level="Strong",
                timeout=None,
                retry_times=0,
                retry_on_rate_limit=False,
            ),
        )
        try:
            return [
                StoredChunk(
                    chunk_uuid=row.chunk_uuid,
                    document_id=row.document_id,
                    document_version=row.document_version,
                    chunking_version=row.chunking_version,
                    content_sha256=row.content_sha256,
                    milvus_pk=row.pk,
                    actual_sha256=digest(row.content),
                )
                for row in TypeAdapter(list[InspectedRow]).validate_python(raw)
            ]
        except ValidationError as exc:
            raise RetrievalUnavailableError(operation="consistency_scan_response") from exc

    async def delete_observed(self, keys: list[int]) -> int:
        """Delete scanned stale keys only; the caller must rescan to prove success."""
        for start in range(0, len(keys), PAGE_SIZE):
            await self._rpc(
                "consistency_delete",
                partial(
                    self._client.delete,
                    self.settings.collection,
                    ids=keys[start : start + PAGE_SIZE],
                    timeout=None,
                    retry_times=0,
                    retry_on_rate_limit=False,
                ),
            )
        return len(keys)
