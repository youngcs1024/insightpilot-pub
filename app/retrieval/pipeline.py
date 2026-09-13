"""Query encoding and registered candidate pools; reranking belongs to Step 3.7."""

import asyncio
import time

import structlog
from sqlalchemy import text

from app.clients.model_runtime import ModelRuntimeClient
from app.core.deadline import Deadline
from app.core.errors import RetrievalConfigurationError, RetrievalUnavailableError
from app.core.observability import TraceMetadata, observe
from app.db.session import Database
from app.repositories.chunk import ChunkRepository
from app.repositories.document import DocumentRepository
from app.retrieval.config import RetrievalSettings
from app.retrieval.search_store import HybridSearchStore
from app.schemas.ingestion import ChunkIdentity, EncodingProfile, digest
from app.schemas.model_runtime import EmbedMode
from app.schemas.retrieval import (
    Candidate, EncodedQuery, RetrievalQuery, RetrievalResult, RetrievalTimings,
)

logger = structlog.get_logger(__name__)


def elapsed(start: float) -> int:
    """Client-side elapsed milliseconds, never model-server timing alone."""
    return max(0, int((time.monotonic() - start) * 1000))


def identity_key(item: ChunkIdentity) -> tuple[str, ...]:
    """Compare every registered identity field, even for repeated SDK hits."""
    return (
        str(item.chunk_uuid), str(item.document_id), str(item.milvus_pk),
        item.document_version, item.chunking_version, item.content_sha256,
    )


class RetrievalPipeline:
    """Owners inject resources and their lifetime; no model weights or business DB access."""

    def __init__(
        self, database: Database, store: HybridSearchStore,
        model: ModelRuntimeClient | None, settings: RetrievalSettings,
    ) -> None:
        self.database = database
        self.store = store
        self.model = model
        self.settings = settings.model_copy(deep=True)
        if (settings.search.use_dense or settings.search.use_sparse_learned) and model is None:
            raise RetrievalConfigurationError(reason="model_client_required")

    async def retrieve(self, query: RetrievalQuery, *, deadline: Deadline) -> RetrievalResult:
        """One caller-owned deadline bounds models, storage and the admission transaction."""
        deadline.check("retrieval")
        try:
            async with asyncio.timeout(deadline.remaining()):
                return await self._retrieve(query, deadline)
        except TimeoutError as exc:
            raise RetrievalUnavailableError(operation="retrieval_deadline") from exc

    async def _retrieve(self, query: RetrievalQuery, deadline: Deadline) -> RetrievalResult:
        started = time.monotonic()
        timings = RetrievalTimings()
        with observe("retrieval_encode", TraceMetadata()):
            encoded = await self._encode(query, deadline)
        timings.encode_ms = elapsed(started)
        stage = time.monotonic()
        pool = await self.store.hybrid_search(
            encoded, self.settings.search, query.time_scope, deadline=deadline,
            timeout_s=self.settings.search_timeout_s,
        )
        timings.search_ms = elapsed(stage)
        stage = time.monotonic()
        with observe("retrieval_admission", TraceMetadata(row_count=len(pool))):
            result = await self._admit(pool, encoded, query, timings)
        timings.admission_ms = elapsed(stage)
        timings.total_ms = elapsed(started)
        result.timings = timings
        logger.info(
            "retrieval_completed", candidates=len(result.candidates),
            rejected=len(pool) - len(result.candidates), corpus_version=result.corpus_version,
            duration_ms=timings.total_ms,
        )
        return result

    async def _encode(self, query: RetrievalQuery, deadline: Deadline) -> EncodedQuery:
        config = self.settings.search
        if not (config.use_dense or config.use_sparse_learned):
            return EncodedQuery(text=query.standalone)
        if self.model is None:
            raise RetrievalConfigurationError(reason="model_client_required")
        result = await self.model.embed([query.standalone], EmbedMode.QUERY, deadline=deadline)
        return EncodedQuery(
            text=query.standalone, dense=result.dense[0], sparse=result.sparse[0],
            metadata=result.metadata,
        )

    async def _admit(
        self, pool: list[Candidate], encoded: EncodedQuery, query: RetrievalQuery,
        timings: RetrievalTimings,
    ) -> RetrievalResult:
        # The read-only snapshot begins AFTER all network/model calls. No ingestion
        # lock is held; manifest identity and registry admission see the same commit.
        async with self.database.session() as session, session.begin():
            await session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
            manifest = await DocumentRepository(session).manifest()
            if manifest is not None and manifest.collection != self.store.settings.collection:
                raise RetrievalConfigurationError(reason="manifest_collection")
            if (
                manifest is not None and manifest.members and encoded.metadata is not None
                and manifest.encoding != EncodingProfile.from_metadata(encoded.metadata)
            ):
                raise RetrievalConfigurationError(reason="manifest_encoding")
            identities = [
                ChunkIdentity.model_validate(item.model_dump(include=set(ChunkIdentity.model_fields)))
                for item in pool if digest(item.content) == item.content_sha256
            ]
            allowed = await ChunkRepository(session).admit(identities)
        keys = {identity_key(item) for item in allowed}
        accepted = []
        for item in pool:
            key = identity_key(item)
            if key in keys and digest(item.content) == item.content_sha256:
                accepted.append(item)
                keys.remove(key)
        return RetrievalResult(
            query=query.model_copy(deep=True), corpus_version=manifest.corpus_version if manifest else None,
            candidates=accepted, retrieval_config=self.settings.search.model_copy(deep=True),
            model_metadata=encoded.metadata, timings=timings,
        )
