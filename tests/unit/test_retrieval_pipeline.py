"""Encoding and snapshot admission failures with injected services, never live models."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.errors import RetrievalConfigurationError, RetrievalUnavailableError
from app.repositories.chunk import ChunkRepository
from app.repositories.document import DocumentRepository
from app.retrieval.config import RetrievalConfig, RetrievalSettings
from app.retrieval.pipeline import RetrievalPipeline
from app.schemas.ingestion import ChunkIdentity, EncodingProfile
from app.schemas.model_runtime import EmbedMode, EmbedResult
from model_runtime.errors import ModelContractError, ModelError
from tests.ingestion_support import model_metadata
from tests.retrieval_support import candidate, deadline, encoded, fake_store, query

EXPECTED_POOL = 20
SOURCE_CAP = 3


@pytest.fixture
def pipeline(monkeypatch: pytest.MonkeyPatch) -> RetrievalPipeline:
    @asynccontextmanager
    async def transaction() -> AsyncIterator[None]:
        yield None

    session = SimpleNamespace(begin=transaction, execute=AsyncMock())

    @asynccontextmanager
    async def sessions() -> AsyncIterator[SimpleNamespace]:
        yield session

    manifest = SimpleNamespace(
        corpus_version="a" * 64,
        collection="kb_chunks",
        members=[1],
        encoding=EncodingProfile.from_metadata(model_metadata()),
    )
    monkeypatch.setattr(DocumentRepository, "manifest", AsyncMock(return_value=manifest))
    monkeypatch.setattr(ChunkRepository, "admit", AsyncMock(side_effect=lambda values: values))
    monkeypatch.setattr(DocumentRepository, "candidate_sources", AsyncMock(
        side_effect=lambda ids: [SimpleNamespace(document_id=item, source_path="source.md") for item in ids]
    ))
    output = EmbedResult(
        request_id="test",
        ms=0,
        queue_ms=0,
        inference_ms=0,
        metadata=model_metadata(),
        dense=[encoded().dense],
        sparse=[encoded().sparse],
    )
    store = fake_store()
    store.hybrid_search = AsyncMock(return_value=[candidate()])
    return RetrievalPipeline(
        SimpleNamespace(session=sessions),
        store,
        SimpleNamespace(embed=AsyncMock(return_value=output)),
        RetrievalSettings(search=RetrievalConfig(use_rerank=False)),
    )


async def test_vector_query_is_encoded_once_and_metadata_preserved(
    pipeline: RetrievalPipeline,
) -> None:
    result = await pipeline.retrieve(query(), deadline=deadline())
    pipeline.model.embed.assert_awaited_once()
    assert pipeline.model.embed.call_args.args == ([query().standalone], EmbedMode.QUERY)
    assert result.model_metadata == model_metadata()
    assert result.corpus_version == "a" * 64
    assert result.candidates
    assert result.timings.total_ms >= result.timings.encode_ms
    assert result.retrieval_config == pipeline.settings.search
    pipeline.settings.search.pool = 1
    assert result.retrieval_config.pool == EXPECTED_POOL


async def test_bm25_only_does_not_require_or_call_models(pipeline: RetrievalPipeline) -> None:
    pipeline.settings.search = RetrievalConfig(use_rerank=False, use_dense=False, use_sparse_learned=False)
    pipeline.model = None
    result = await pipeline.retrieve(query(), deadline=deadline())
    assert result.model_metadata is None
    assert result.candidates
    assert pipeline.store.hybrid_search.call_args.args[0].dense is None


async def test_empty_pool_is_a_success(pipeline: RetrievalPipeline) -> None:
    pipeline.store.hybrid_search.return_value = []
    assert (await pipeline.retrieve(query(), deadline=deadline())).candidates == []


@pytest.mark.parametrize("failure", [ModelError(), ModelContractError()])
async def test_encoding_failure_stops_before_search(
    pipeline: RetrievalPipeline, failure: Exception
) -> None:
    pipeline.model.embed.side_effect = failure
    with pytest.raises(type(failure)):
        await pipeline.retrieve(query(), deadline=deadline())
    pipeline.store.hybrid_search.assert_not_called()


async def test_manifest_encoding_mismatch_fails_closed(
    pipeline: RetrievalPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = SimpleNamespace(
        corpus_version="a" * 64, collection="kb_chunks", members=[1], encoding=None
    )
    monkeypatch.setattr(DocumentRepository, "manifest", AsyncMock(return_value=manifest))
    with pytest.raises(RetrievalConfigurationError):
        await pipeline.retrieve(query(), deadline=deadline())


async def test_manifest_collection_mismatch_fails_closed(
    pipeline: RetrievalPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        DocumentRepository, "manifest", AsyncMock(return_value=SimpleNamespace(collection="other"))
    )
    with pytest.raises(RetrievalConfigurationError):
        await pipeline.retrieve(query(), deadline=deadline())


async def test_full_identity_and_text_hash_checked_after_admission(
    pipeline: RetrievalPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = candidate()
    wrong_version = good.model_copy(update={"document_version": "f" * 64})
    corrupted = good.model_copy(update={"content": "uncommitted replacement"})
    pipeline.store.hybrid_search.return_value = [wrong_version, corrupted, good, good]
    allowed = ChunkIdentity.model_validate(good.model_dump(include=set(ChunkIdentity.model_fields)))
    monkeypatch.setattr(ChunkRepository, "admit", AsyncMock(return_value=[allowed]))
    result = await pipeline.retrieve(query(), deadline=deadline())
    assert result.candidates == [good]


async def test_admission_reads_repeatable_read_only_snapshot(pipeline: RetrievalPipeline) -> None:
    await pipeline.retrieve(query(), deadline=deadline())
    async with pipeline.database.session() as session:
        assert (
            str(session.execute.call_args.args[0])
            == "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
        )


def test_vector_pipeline_requires_injected_model() -> None:
    with pytest.raises(RetrievalConfigurationError):
        RetrievalPipeline(None, fake_store(), None, RetrievalSettings())


async def test_pipeline_cancellation_is_not_converted(pipeline: RetrievalPipeline) -> None:
    pipeline.model.embed.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await pipeline.retrieve(query(), deadline=deadline())


async def test_outer_deadline_includes_admission(
    pipeline: RetrievalPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def pending() -> object:
        return await asyncio.get_running_loop().create_future()

    monkeypatch.setattr(DocumentRepository, "manifest", AsyncMock(side_effect=pending))
    with pytest.raises(RetrievalUnavailableError):
        await pipeline.retrieve(query(), deadline=deadline(0.01))


async def test_reranking_begins_after_admission_transaction_closes(
    pipeline: RetrievalPipeline
) -> None:
    from tests.rerank_support import model

    active = False

    @asynccontextmanager
    async def transaction() -> AsyncIterator[None]:
        nonlocal active
        active = True
        try:
            yield None
        finally:
            active = False

    async with pipeline.database.session() as session:
        session.begin = transaction
    client = model([0.8])
    response = client.rerank.return_value

    async def rerank(*args: object, **kwargs: object) -> object:
        assert not active
        return response

    pipeline.model.rerank = AsyncMock(side_effect=rerank)
    pipeline.settings.search.use_rerank = True
    result = await pipeline.retrieve(query(), deadline=deadline())
    assert result.reranked
    assert result.retrieval_config.filtering.max_per_doc == SOURCE_CAP
    pipeline.settings.search.filtering.max_per_doc = 2
    assert result.retrieval_config.filtering.max_per_doc == SOURCE_CAP


async def test_missing_registered_source_fails_closed(
    pipeline: RetrievalPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.core.errors import IngestionRegistryError

    monkeypatch.setattr(DocumentRepository, "candidate_sources", AsyncMock(return_value=[]))
    with pytest.raises(IngestionRegistryError):
        await pipeline.retrieve(query(), deadline=deadline())
