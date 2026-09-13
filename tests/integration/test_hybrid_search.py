"""Step 3.6 real storage acceptance with fixed vectors, not live model quality claims."""

from collections.abc import AsyncIterator
from datetime import date
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import update

from app.clients.model_runtime import ModelRuntimeClient
from app.core.config_models import ModelRuntimeClientSettings
from app.db.models.chunk import Chunk
from app.db.models.document import Document
from app.db.session import Database
from app.repositories.chunk import ChunkRepository
from app.repositories.document import DocumentRepository
from app.retrieval.config import MilvusSettings, RetrievalConfig
from app.retrieval.ingestion_store import IngestionStore
from app.retrieval.search_store import HybridSearchStore
from app.schemas.ingestion import ActiveManifest, canonical, digest, document_id
from app.schemas.retrieval import PointTimeScope, PolicyPeriod, RangeTimeScope, RetrievalQuery
from tests.ingestion_support import clear_registry
from tests.milvus_support import MilvusStack, milvus_stack
from tests.retrieval_support import (
    QueryEmbeddings,
    RetrievalHarness,
    corpus,
    deadline,
    encoded,
    query,
)
from tests.shared_database import TestPostgres

pytestmark = [pytest.mark.integration, pytest.mark.storage]
__all__ = ["milvus_stack"]


@pytest.fixture
async def harness(
    migrated_db: TestPostgres, milvus_stack: MilvusStack, tmp_path: Path
) -> AsyncIterator[RetrievalHarness]:
    database = Database(migrated_db.app)
    database.start()
    settings = ModelRuntimeClientSettings(auth_token=SecretStr("synthetic-retrieval-token"))
    storage = MilvusSettings(uri=milvus_stack.uri, collection="step36_" + uuid4().hex, timeout_s=30)
    embeddings = QueryEmbeddings(calls=[])
    corpus(tmp_path)
    try:
        await clear_registry(database)
        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(embeddings.handle)) as http,
            IngestionStore(storage) as ingestion,
            HybridSearchStore(storage) as search,
        ):
            model = ModelRuntimeClient(settings, http)
            value = RetrievalHarness(
                database, ingestion, search, model, settings, embeddings, tmp_path
            )
            await value.ingest()
            await search.ensure_collection()
            yield value
    finally:
        await clear_registry(database)
        await database.aclose()


@pytest.mark.parametrize(
    ("dense", "sparse", "bm25"),
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ],
)
async def test_all_enabled_arm_combinations_return_registered_candidates(
    harness: RetrievalHarness, dense: bool, sparse: bool, bm25: bool
) -> None:
    config = RetrievalConfig(
        use_dense=dense, use_sparse_learned=sparse, use_bm25=bm25, record_arm_scores=True
    )
    result = await harness.pipeline(config).retrieve(query(), deadline=deadline())
    assert result.candidates
    assert all(item.milvus_pk is not None for item in result.candidates)
    assert (result.candidates[0].scores.rrf is not None) == (sum((dense, sparse, bm25)) > 1)
    assert result.retrieval_config == config


async def test_dense_only_returns_candidates(harness: RetrievalHarness) -> None:
    result = await harness.pipeline(
        RetrievalConfig(use_sparse_learned=False, use_bm25=False)
    ).retrieve(query(), deadline=deadline())
    assert result.candidates
    assert result.candidates[0].scores.dense is not None


async def test_three_arm_hybrid_returns_fused_candidates(harness: RetrievalHarness) -> None:
    result = await harness.pipeline().retrieve(query(), deadline=deadline())
    assert result.candidates
    assert all(item.scores.rrf is not None for item in result.candidates)
    assert all(item.scores.dense is None for item in result.candidates)


async def test_parent_content_is_populated(harness: RetrievalHarness) -> None:
    result = await harness.pipeline(
        RetrievalConfig(use_dense=False, use_sparse_learned=False)
    ).retrieve(query(), deadline=deadline())
    assert result.candidates
    assert any(len(item.parent_content) > len(item.content) for item in result.candidates)
    assert "SKU-A1023" in result.candidates[0].parent_content


async def test_bm25_arm_finds_exact_sku_code(harness: RetrievalHarness) -> None:
    # The synthetic embedding directions deliberately give dense a wrong first hit.
    # This demonstrates the independent lexical mechanism, not BGE-M3 model quality.
    dense = RetrievalConfig(use_sparse_learned=False, use_bm25=False, pool=1)
    lexical = RetrievalConfig(use_dense=False, use_sparse_learned=False, pool=1)
    dense_result = await harness.pipeline(dense).retrieve(query(), deadline=deadline())
    lexical_result = await harness.pipeline(lexical).retrieve(query(), deadline=deadline())
    assert dense_result.candidates[0].document_id != document_id("sku.md")
    assert lexical_result.candidates[0].document_id == document_id("sku.md")
    assert "SKU-A1023" in lexical_result.candidates[0].content


async def test_arm_scores_recorded_when_enabled(harness: RetrievalHarness) -> None:
    result = await harness.pipeline(RetrievalConfig(record_arm_scores=True)).retrieve(
        query(), deadline=deadline()
    )
    sku = next(item for item in result.candidates if "SKU-A1023" in item.content)
    assert sku.scores.rrf is not None
    assert sku.scores.dense is not None
    assert sku.scores.sparse_learned is not None
    assert sku.scores.sparse_bm25 is not None
    assert sku.scores.rerank is None


async def test_effective_filter_excludes_superseded_policy(harness: RetrievalHarness) -> None:
    result = await harness.pipeline().retrieve(query(), deadline=deadline())
    ids = {item.document_id for item in result.candidates}
    assert document_id("august.md") in ids
    assert document_id("july.md") not in ids


async def test_comparison_retrieves_july_and_august_versions(harness: RetrievalHarness) -> None:
    request = RetrievalQuery(
        standalone="退款规则",
        time_scope=RangeTimeScope(
            periods=[
                PolicyPeriod(start=date(2026, 7, 1), end=date(2026, 8, 1), label="七月"),
                PolicyPeriod(start=date(2026, 8, 1), end=date(2026, 9, 1), label="八月"),
            ]
        ),
    )
    result = await harness.pipeline().retrieve(request, deadline=deadline())
    assert {document_id("july.md"), document_id("august.md")} <= {
        item.document_id for item in result.candidates
    }
    assert result.query.time_scope == request.time_scope


async def test_effective_to_is_exclusive(harness: RetrievalHarness) -> None:
    request = query().model_copy(update={"time_scope": PointTimeScope(as_of=date(2026, 8, 1))})
    result = await harness.pipeline().retrieve(request, deadline=deadline())
    assert document_id("july.md") not in {item.document_id for item in result.candidates}
    assert document_id("august.md") in {item.document_id for item in result.candidates}


async def test_unbounded_dates_match(harness: RetrievalHarness) -> None:
    # Exercise both sentinel bounds at the storage boundary, independently of authoring.
    async with harness.database.session() as session:
        chunks = await ChunkRepository(session).list_document(document_id("sku.md"))
    pk = chunks[0].milvus_pk
    raw = await harness.search._client.query(
        harness.search.settings.collection,
        filter=f"pk == {pk}",
        output_fields=["*"],
        consistency_level="Strong",
        timeout=10,
    )
    row = raw[0]
    row.pop("sparse_bm25", None)
    row["effective_from"], row["effective_to"] = -1, -1
    row.pop("pk")
    receipt = await harness.search._client.insert(
        harness.search.settings.collection, data=[row], timeout=10
    )
    scope = PointTimeScope(as_of=date(1900, 1, 1))
    result = await harness.search.hybrid_search(
        encoded(), RetrievalConfig(), scope, deadline=deadline(), timeout_s=30
    )
    assert any(
        item.milvus_pk in receipt["ids"]
        and item.effective_from is None
        and item.effective_to is None
        for item in result
    )


async def test_uncommitted_vectors_never_reach_pipeline(harness: RetrievalHarness) -> None:
    async with harness.database.session() as session:
        chunks = await ChunkRepository(session).list_document(document_id("sku.md"))
    pk = chunks[0].milvus_pk
    raw = await harness.search._client.query(
        harness.search.settings.collection,
        filter=f"pk == {pk}",
        output_fields=["*"],
        consistency_level="Strong",
        timeout=10,
    )
    row = raw[0]
    row.pop("pk")
    row.pop("sparse_bm25", None)
    row["document_version"] = "f" * 64
    receipt = await harness.search._client.insert(
        harness.search.settings.collection, data=[row], timeout=10
    )
    result = await harness.pipeline().retrieve(query(), deadline=deadline())
    assert set(receipt["ids"]).isdisjoint(item.milvus_pk for item in result.candidates)
    assert all(item.document_version != "f" * 64 for item in result.candidates)


@pytest.mark.parametrize("damage", ["hash", "pk", "retired"])
async def test_registry_rejects_stale_or_mismatched_hits(
    harness: RetrievalHarness, damage: str
) -> None:
    identifier = document_id("sku.md")
    async with harness.database.session() as session, session.begin():
        if damage == "retired":
            await session.execute(
                update(Document).where(Document.id == identifier).values(status="deleted")
            )
        elif damage == "hash":
            await session.execute(
                update(Chunk).where(Chunk.document_id == identifier).values(content_sha256="f" * 64)
            )
        else:
            await session.execute(
                update(Chunk).where(Chunk.document_id == identifier).values(milvus_pk=None)
            )
    result = await harness.pipeline().retrieve(query(), deadline=deadline())
    assert identifier not in {item.document_id for item in result.candidates}


async def test_returned_manifest_matches_committed_registry(harness: RetrievalHarness) -> None:
    result = await harness.pipeline().retrieve(query(), deadline=deadline())
    async with harness.database.session() as session:
        manifest = await DocumentRepository(session).manifest()
    assert result.corpus_version == manifest.corpus_version
    assert all(
        any(
            member.document_id == item.document_id
            and member.document_version == item.document_version
            and member.chunking_version == item.chunking_version
            for member in manifest.members
        )
        for item in result.candidates
    )


async def test_empty_candidates_are_not_an_error(harness: RetrievalHarness) -> None:
    request = RetrievalQuery(
        standalone="退款规则", time_scope=PointTimeScope(as_of=date(1900, 1, 1))
    )
    result = await harness.pipeline().retrieve(request, deadline=deadline())
    assert result.candidates == []
    assert result.corpus_version is not None


async def test_concurrent_publication_keeps_admission_snapshot_consistent(
    harness: RetrievalHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_manifest = DocumentRepository.manifest
    async with harness.database.session() as session:
        before = await original_manifest(DocumentRepository(session))
    assert before is not None
    identifier = document_id("sku.md")
    members = [item for item in before.members if item.document_id != identifier]
    version = digest(
        canonical(
            [
                (str(item.document_id), item.document_version, item.chunking_version)
                for item in members
            ]
        )
    )
    after = before.model_copy(update={"members": members, "corpus_version": version})

    async def publish_after_snapshot(repository: DocumentRepository) -> ActiveManifest | None:
        snapshot = await original_manifest(repository)
        async with harness.database.session() as writer, writer.begin():
            await writer.execute(
                update(Document).where(Document.id == identifier).values(status="deleted")
            )
            await ChunkRepository(writer).replace(identifier, [])
            await DocumentRepository(writer).save_manifest(after)
        return snapshot

    monkeypatch.setattr(DocumentRepository, "manifest", publish_after_snapshot)
    result = await harness.pipeline().retrieve(query(), deadline=deadline())
    assert result.corpus_version == before.corpus_version
    assert identifier in {item.document_id for item in result.candidates}
    async with harness.database.session() as session:
        current = await original_manifest(DocumentRepository(session))
    assert current == after
    assert current.corpus_version != result.corpus_version
