"""Step 3.4 acceptance with real migrated PostgreSQL, Milvus and HTTP substitutes."""

# ruff: noqa: PLR2004 -- explicit acceptance counts and small synthetic batch sizes.

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from uuid import uuid4

import httpx
import pytest
from sqlalchemy import text

from alembic import command
from scripts.migrate_all import migration_config
from scripts.migration_settings import MigrationSettings, MigrationTarget

from app.core.config_models import ModelRuntimeClientSettings
from app.core.errors import IngestionAlreadyRunning, IngestionConfigurationError, InsightPilotError, RetrievalUnavailableError
from app.db.session import Database
from app.repositories.chunk import ChunkRepository
from app.repositories.document import DocumentRepository
from app.retrieval.config import MilvusSettings
from app.retrieval.ingestion_store import IngestionStore
from app.schemas.ingestion import ActiveManifest, DocumentStatus, document_id
from model_runtime.errors import ModelError
from tests.corpus_support import entry, markdown_source, write_inventory
from tests.ingestion_support import Embeddings, Harness, clear_registry, write_sources
from tests.milvus_support import MilvusStack, milvus_stack
from tests.shared_database import TestPostgres, pg_container

pytestmark = [pytest.mark.integration, pytest.mark.storage]
__all__ = ["milvus_stack"]


@pytest.fixture
async def harness(migrated_db: TestPostgres, milvus_stack: MilvusStack, tmp_path: Path) -> AsyncIterator[Harness]:
    database = Database(migrated_db.app)
    database.start()
    embeddings = Embeddings()
    settings = ModelRuntimeClientSettings(auth_token="synthetic-ingestion-token")
    write_sources(tmp_path)
    try:
        await clear_registry(database)
        async with httpx.AsyncClient(transport=httpx.MockTransport(embeddings.handle)) as http:
            async with IngestionStore(MilvusSettings(uri=milvus_stack.uri, collection="step34_" + uuid4().hex, timeout_s=30)) as store:
                yield Harness(database, store, http, embeddings, tmp_path, settings)
    finally:
        await clear_registry(database)
        await database.aclose()


async def test_first_ingest_creates_documents_and_chunks(harness: Harness) -> None:
    result = await harness.service().ingest(harness.root)
    assert result.successful and result.documents_changed == 2 and result.chunks_inserted > 0
    documents = await harness.documents()
    assert len(documents) == 2
    for document in documents:
        chunks = await harness.chunks(document.document_id)
        indexed = await harness.store.identities(document.document_id)
        assert len(chunks) == document.chunk_count == len(indexed)
        assert {row.milvus_pk for row in chunks} == {row.milvus_pk for row in indexed}
        assert not document.cleanup_pending


async def test_reingest_unchanged_is_noop(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    first = await harness.service().ingest(harness.root)
    calls = len(harness.embeddings.calls)
    async def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("Unchanged ingestion attempted a storage mutation")
    monkeypatch.setattr(harness.store, "insert", forbidden)
    monkeypatch.setattr(harness.store, "cleanup", forbidden)
    second = await harness.service().ingest(harness.root)
    assert second.successful and second.corpus_version == first.corpus_version
    assert (second.documents_changed, second.chunks_inserted, second.chunks_deleted) == (0, 0, 0)
    assert len(harness.embeddings.calls) == calls


async def test_changed_file_replaces_only_its_chunks(harness: Harness) -> None:
    await harness.service().ingest(harness.root)
    other = await harness.chunks(document_id("other.md"))
    old = await harness.chunks(document_id("rule.md"))
    (harness.root / "rule.md").write_text(markdown_source(body="# 政策\n\n更新了规则。"))
    result = await harness.service().ingest(harness.root)
    assert result.documents_changed == 1 and result.chunks_deleted == len(old)
    assert await harness.chunks(document_id("other.md")) == other
    assert {item.chunk_uuid for item in await harness.chunks(document_id("rule.md"))}.isdisjoint(item.chunk_uuid for item in old)


async def test_unchanged_chunks_of_changed_file_survive(harness: Harness) -> None:
    await harness.service().ingest(harness.root)
    before = await harness.chunks(document_id("rule.md"))
    path = harness.root / "rule.md"
    path.write_text(path.read_text() + "\n\n## 新节\n\n附加的新规则。")
    await harness.service().ingest(harness.root)
    after = await harness.chunks(document_id("rule.md"))
    indexed = await harness.store.identities(document_id("rule.md"))
    assert {row.content_sha256 for row in before} <= {row.content_sha256 for row in after}
    assert len(indexed) == len(after)


async def test_deleted_file_chunks_removed(harness: Harness) -> None:
    await harness.service().ingest(harness.root)
    write_inventory(harness.root, [entry("other.md")])
    result = await harness.service().ingest(harness.root)
    assert result.documents_deleted == 1 and result.chunks_deleted > 0
    assert await harness.store.identities(document_id("rule.md")) == []
    assert await harness.chunks(document_id("rule.md")) == []
    assert (harness.root / "rule.md").is_file()  # Unlisted files are never re-discovered.
    tombstone = next(item for item in await harness.documents() if item.source_path == "rule.md")
    assert tombstone.status is DocumentStatus.DELETED and not tombstone.cleanup_pending


async def test_failed_file_does_not_abort_batch(harness: Harness) -> None:
    await harness.service().ingest(harness.root)
    old = await harness.chunks(document_id("rule.md"))
    (harness.root / "rule.md").write_text("broken metadata")
    (harness.root / "other.md").write_text(markdown_source(body="# 新规则\n\n变更。"))
    result = await harness.service().ingest(harness.root)
    assert not result.successful and result.documents_changed == 1
    assert [item.path for item in result.failed_files] == ["rule.md"]
    assert await harness.chunks(document_id("rule.md")) == old


async def test_missing_listed_source_preserves_old_version(harness: Harness) -> None:
    await harness.service().ingest(harness.root)
    old = await harness.chunks(document_id("rule.md"))
    (harness.root / "rule.md").rename(harness.root / "temporarily-unavailable.md")
    result = await harness.service().ingest(harness.root)
    assert result.failed_files and result.documents_deleted == 0
    assert await harness.chunks(document_id("rule.md")) == old


async def test_metadata_change_versions_document(harness: Harness) -> None:
    await harness.service().ingest(harness.root)
    before = await harness.chunks(document_id("rule.md"))
    path = harness.root / "rule.md"
    path.write_text(path.read_text().replace("effective_to: null", "effective_to: 2027-01-01"))
    await harness.service().ingest(harness.root)
    after = await harness.chunks(document_id("rule.md"))
    assert before[0].document_version != after[0].document_version
    assert before[0].content_sha256 == after[0].content_sha256


async def test_repeated_text_preserves_provenance(harness: Harness) -> None:
    await harness.service().ingest(harness.root)
    first, second = await harness.chunks(document_id("rule.md")), await harness.chunks(document_id("other.md"))
    assert first[0].content_sha256 == second[0].content_sha256
    assert {item.chunk_uuid for item in first}.isdisjoint(item.chunk_uuid for item in second)


async def test_concurrent_ingest_blocked_by_advisory_lock(harness: Harness) -> None:
    async with harness.database.session() as session, session.begin():
        assert await DocumentRepository(session).try_lock()
        with pytest.raises(IngestionAlreadyRunning):
            await harness.service().ingest(harness.root)
    assert (await harness.service().ingest(harness.root)).successful


async def test_model_failure_aborts_before_any_write(harness: Harness) -> None:
    harness.embeddings.fail_at = 2
    harness.model_settings.embed_batch = 1
    with pytest.raises(ModelError):
        await harness.service().ingest(harness.root)
    assert await harness.documents() == []
    harness.embeddings.fail_at = None
    assert (await harness.service().ingest(harness.root)).successful


async def test_registry_failure_preserves_previous_active_version(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    initial = await harness.service().ingest(harness.root)
    before = await harness.chunks(document_id("rule.md"))
    (harness.root / "rule.md").write_text(markdown_source(body="# 修改\n\n注册失败。"))
    async def fail(self: DocumentRepository, value: ActiveManifest) -> None:
        await self.session.execute(text("SELECT 1 / 0"))
    with monkeypatch.context() as patch:
        patch.setattr(DocumentRepository, "save_manifest", fail)
        with pytest.raises(InsightPilotError):
            await harness.service().ingest(harness.root)
    assert await harness.chunks(document_id("rule.md")) == before
    async with harness.database.session() as session:
        manifest = await DocumentRepository(session).manifest()
        assert manifest.corpus_version == initial.corpus_version
        candidates = await harness.store.identities(document_id("rule.md"))
        admitted = await ChunkRepository(session).admit(candidates)
        assert {item.chunk_uuid for item in admitted} == {item.chunk_uuid for item in before}


async def test_uncommitted_manifest_not_retrievable(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    observed = False
    original = DocumentRepository.save_manifest
    async def inspect(self: DocumentRepository, value: ActiveManifest) -> None:
        nonlocal observed
        await original(self, value)
        candidates = await harness.store.identities(document_id("rule.md"))
        assert candidates
        async with harness.database.session() as reader:
            assert await DocumentRepository(reader).manifest() is None
            assert await ChunkRepository(reader).admit(candidates) == []
        observed = True
    monkeypatch.setattr(DocumentRepository, "save_manifest", inspect)
    await harness.service().ingest(harness.root)
    assert observed
    async with harness.database.session() as reader:
        candidates = await harness.store.identities(document_id("rule.md"))
        assert await ChunkRepository(reader).admit(candidates) == candidates


async def test_registry_and_active_manifest_commit_atomically(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    original = DocumentRepository.save_manifest
    async def check(self: DocumentRepository, value: ActiveManifest) -> None:
        await original(self, value)
        async with harness.database.session() as reader:
            assert await DocumentRepository(reader).list_documents() == []
            assert await ChunkRepository(reader).list_document(document_id("rule.md")) == []
    monkeypatch.setattr(DocumentRepository, "save_manifest", check)
    await harness.service().ingest(harness.root)
    async with harness.database.session() as session:
        assert await DocumentRepository(session).manifest() is not None
        assert await DocumentRepository(session).list_documents()
        assert await ChunkRepository(session).list_document(document_id("rule.md"))


async def test_crash_after_commit_cleanup_retry(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    await harness.service().ingest(harness.root)
    (harness.root / "rule.md").write_text(markdown_source(body="# 更新\n\n提交后中断。"))
    async def crash(*args: object) -> int:
        raise asyncio.CancelledError()
    with monkeypatch.context() as patch:
        patch.setattr(harness.store, "cleanup", crash)
        with pytest.raises(asyncio.CancelledError):
            await harness.service().ingest(harness.root)
    assert any(item.cleanup_pending for item in await harness.documents())
    calls = len(harness.embeddings.calls)
    retry = await harness.service().ingest(harness.root)
    assert retry.successful and retry.documents_changed == 0 and retry.chunks_inserted == 0
    assert retry.chunks_deleted > 0 and len(harness.embeddings.calls) == calls


async def test_cleanup_failure_returns_pending(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    async def unavailable(*args: object) -> int:
        raise RetrievalUnavailableError()
    with monkeypatch.context() as patch:
        patch.setattr(harness.store, "cleanup", unavailable)
        result = await harness.service().ingest(harness.root)
    assert not result.successful and len(result.cleanup_pending) == 2
    assert (await harness.service().ingest(harness.root)).successful


async def test_incompatible_embedding_profile_refuses_mixed_index(harness: Harness) -> None:
    await harness.service().ingest(harness.root)
    before = await harness.chunks(document_id("rule.md"))
    harness.embeddings.metadata.embed_max_length = 256
    (harness.root / "rule.md").write_text(markdown_source(body="# 新内容\n\n需要嵌入。"))
    with pytest.raises(IngestionConfigurationError):
        await harness.service().ingest(harness.root)
    assert await harness.chunks(document_id("rule.md")) == before


async def test_candidate_physical_identity_must_match(harness: Harness) -> None:
    await harness.service().ingest(harness.root)
    candidate = (await harness.store.identities(document_id("rule.md")))[0]
    invalid = [candidate.model_copy(update={field: value}) for field, value in [
        ("milvus_pk", candidate.milvus_pk + 1), ("content_sha256", "0" * 64),
        ("document_version", "0" * 64), ("chunking_version", "0" * 64),
    ]]
    async with harness.database.session() as reader:
        assert await ChunkRepository(reader).admit(invalid) == []
        assert await ChunkRepository(reader).admit([candidate]) == [candidate]


async def test_complete_authored_corpus_is_idempotent(harness: Harness) -> None:
    root = Path(__file__).resolve().parents[2] / "data/corpus"
    first = await harness.service().ingest(root)
    assert first.successful and first.documents_changed == 60
    assert len(await harness.documents()) == 60
    second = await harness.service().ingest(root)
    assert second.successful and second.corpus_version == first.corpus_version
    assert (second.documents_changed, second.chunks_inserted, second.chunks_deleted) == (0, 0, 0)


async def test_fresh_databases_produce_identical_chunk_ids(
    harness: Harness, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A second bootstrapped PostgreSQL container proves this is not a savepoint or
    # an in-memory UUID comparison. Its named volume remains preserved on teardown.
    await harness.service().ingest(harness.root)
    before = await harness.chunks(document_id("rule.md"))
    provider = pg_container.__wrapped__(tmp_path_factory)
    postgres = await asyncio.to_thread(next, provider)
    database = Database(postgres.app)
    try:
        settings = MigrationSettings(_env_file=None, migration={
            "port": postgres.port, "password": postgres.password,
        })
        with monkeypatch.context() as patch:
            patch.setattr(MigrationSettings, "load", classmethod(lambda cls: settings))
            await asyncio.to_thread(command.upgrade, migration_config(MigrationTarget.APP), "head")
        database.start()
        async with IngestionStore(harness.store.settings.model_copy(update={"collection": "step34_fresh_" + uuid4().hex})) as store:
            other = Harness(database, store, harness.client, harness.embeddings, harness.root, harness.model_settings)
            await other.service().ingest(other.root)
            after = await other.chunks(document_id("rule.md"))
            assert [item.chunk_uuid for item in before] == [item.chunk_uuid for item in after]
            assert [item.document_version for item in before] == [item.document_version for item in after]
    finally:
        await database.aclose()
        await asyncio.to_thread(provider.close)
