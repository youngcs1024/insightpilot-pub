"""Step 3.5 acceptance on real PostgreSQL/Milvus, with deterministic model HTTP."""

# ruff: noqa: PLR2004 -- explicit corruption sizes and acceptance counts.

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from sqlalchemy import delete, text, update

from app.core.errors import IngestionAlreadyRunning, IngestionRegistryError, InsightPilotError
from app.db.models.chunk import Chunk
from app.db.models.document import CorpusManifestRecord, Document
from app.db.models.evidence import DataEvidenceRecord
from app.repositories.chunk import ChunkRepository
from app.repositories.document import DocumentRepository
from app.schemas.consistency import DriftKind
from app.schemas.ingestion import RegisteredChunk, canonical, digest, document_id
from app.services.evidence import EvidenceService
from app.services.ingestion_config import IngestionSettings
from tests.agents.support import context, invoke
from tests.consistency_support import ConsistencyHarness, consistency_harness
from tests.corpus_support import entry, markdown_source, write_inventory
from tests.integration.checkpoint_support import admitted
from tests.milvus_support import milvus_stack

pytestmark = [pytest.mark.integration, pytest.mark.storage]
__all__ = ["consistency_harness", "milvus_stack"]


async def test_no_drift_after_clean_ingest(consistency_harness: ConsistencyHarness) -> None:
    harness = consistency_harness
    await harness.service().ingest(harness.root)
    calls = len(harness.embeddings.calls)
    report = await harness.checker(models=False).check()
    assert report.successful
    assert set(report.counts) == set(DriftKind)
    assert sum(report.counts.values()) == 0
    assert len(harness.embeddings.calls) == calls


async def test_detects_orphaned_milvus_rows(consistency_harness: ConsistencyHarness) -> None:
    harness = consistency_harness
    await harness.service().ingest(harness.root)
    vector = (await harness.vectors())[0]
    vector.chunk.chunk_uuid = uuid4()
    vector.chunk.document_id = uuid4()  # The document is absent from PostgreSQL too.
    receipt = await harness.store.insert([vector])
    report = await harness.checker(models=False).check()
    orphan = [item for item in report.remaining if item.kind is DriftKind.ORPHAN]
    assert len(orphan) == 1
    assert orphan[0].milvus_pk == receipt.ids[0]
    assert not report.successful


@pytest.mark.parametrize("pk", [None, 9_000_000_000_000_000_000])
async def test_detects_missing_milvus_pk(
    consistency_harness: ConsistencyHarness, pk: int | None
) -> None:
    harness = consistency_harness
    await harness.service().ingest(harness.root)
    chunk = (await harness.chunks(document_id("rule.md")))[0]
    await harness.point_to(chunk.chunk_uuid, pk)
    report = await harness.checker().check()
    assert report.counts[DriftKind.MISSING] == 1
    assert (await harness.checker().check(root=harness.root)).successful


@pytest.mark.parametrize("damage", ["metadata", "text"])
async def test_detects_sha_mismatch(
    consistency_harness: ConsistencyHarness, damage: str
) -> None:
    harness = consistency_harness
    await harness.service().ingest(harness.root)
    vector = (await harness.vectors())[0]
    if damage == "metadata":
        vector.chunk.content_sha256 = "0" * 64
    else:
        vector.chunk.content += "corrupt index text"
    receipt = await harness.store.insert([vector])
    await harness.point_to(vector.chunk.chunk_uuid, receipt.ids[0])
    report = await harness.checker().check()
    assert report.counts[DriftKind.SHA] > 0
    repaired = await harness.checker().check(root=harness.root)
    assert repaired.successful
    assert repaired.documents_repaired == 1


async def test_fix_removes_orphans(consistency_harness: ConsistencyHarness) -> None:
    harness = consistency_harness
    await harness.service().ingest(harness.root)
    vector = (await harness.vectors())[0]
    vector.chunk.chunk_uuid = uuid4()
    vector.chunk.document_id = uuid4()
    receipt = await harness.store.insert([vector])
    report = await harness.checker(models=False).check(root=Path("/source-not-required"))
    assert report.successful
    assert report.chunks_inserted == 0
    assert report.chunks_deleted == 1
    assert receipt.ids[0] not in {row.milvus_pk for row in (await harness.store.scan()).rows}


@pytest.mark.parametrize("stale_version", [False, True])
async def test_duplicate_or_stale_vector_cleanup_needs_no_model(
    consistency_harness: ConsistencyHarness, stale_version: bool
) -> None:
    harness = consistency_harness
    await harness.service().ingest(harness.root)
    before = await harness.chunks(document_id("rule.md"))
    vector = (await harness.vectors())[0]
    if stale_version:
        vector.chunk.document_version = "0" * 64
    await harness.store.insert([vector])
    report = await harness.checker(models=False).check(root=Path("/source-not-required"))
    assert report.successful
    assert report.chunks_deleted == 1
    assert report.chunks_inserted == 0
    assert await harness.chunks(document_id("rule.md")) == before


@pytest.mark.parametrize("damage", ["count", "registry_row", "foreign_pk"])
async def test_repairs_counts_and_registry_links(
    consistency_harness: ConsistencyHarness, damage: str
) -> None:
    harness = consistency_harness
    await harness.service().ingest(harness.root)
    identifier = document_id("rule.md")
    before = await harness.chunks(identifier)
    async with harness.database.session() as session, session.begin():
        if damage == "count":
            await session.execute(
                update(Document).where(Document.id == identifier).values(chunk_count=len(before) + 1)
            )
        elif damage == "registry_row":
            await session.execute(delete(Chunk).where(Chunk.id == before[0].chunk_uuid))
        else:
            await session.execute(
                update(Chunk).where(Chunk.id == before[0].chunk_uuid)
                .values(milvus_pk=before[1].milvus_pk)
            )
    report = await harness.checker().check(root=harness.root)
    assert report.successful
    assert {row.chunk_uuid for row in await harness.chunks(identifier)} == {
        row.chunk_uuid for row in before
    }


async def test_reindex_changes_pk_not_chunk_id(consistency_harness: ConsistencyHarness) -> None:
    harness = consistency_harness
    await harness.service().ingest(harness.root)
    before = (await harness.store.scan()).rows
    await harness.store.delete_observed([row.milvus_pk for row in before])
    report = await harness.checker().check(root=harness.root)
    after = (await harness.store.scan()).rows
    assert report.successful
    assert report.chunks_inserted == len(before)
    assert {row.chunk_uuid for row in before} == {row.chunk_uuid for row in after}
    assert {row.milvus_pk for row in before}.isdisjoint(row.milvus_pk for row in after)


async def test_frozen_judgments_resolve_after_rebuild(
    consistency_harness: ConsistencyHarness,
) -> None:
    harness = consistency_harness
    initial = await harness.service().ingest(harness.root)
    rows = (await harness.store.scan()).rows
    # Fixed for this run before corruption; no expected-value refresh during repair.
    frozen = tuple((row.chunk_uuid, 3) for row in rows)
    await harness.store._rpc(
        "test_drop_isolated_collection",
        lambda: harness.store._client.drop_collection(
            harness.store.settings.collection,
            timeout=None, retry_times=0, retry_on_rate_limit=False,
        ),
    )
    absent = await harness.checker(models=False).check()
    assert not absent.collection_exists
    assert absent.counts[DriftKind.MISSING] == len(rows)
    report = await harness.checker().check(root=harness.root)
    assert report.successful
    assert report.corpus_version == initial.corpus_version
    rebuilt = (await harness.store.scan()).rows
    assert {identifier for identifier, grade in frozen if grade >= 2} == {
        row.chunk_uuid for row in rebuilt
    }
    async with harness.database.session() as session:
        assert len(await ChunkRepository(session).admit(rebuilt)) == len(frozen)


async def test_index_cleanup_preserves_evidence_snapshots(
    consistency_harness: ConsistencyHarness,
) -> None:
    harness = consistency_harness
    await harness.service().ingest(harness.root)
    identity = await admitted(harness.database)
    ctx = context()
    await invoke(ctx)
    service = EvidenceService(harness.database)
    snapshot = await service.commit(identity, ctx.evidence.snapshot.data)
    async with harness.database.session() as session:
        record = await session.get(DataEvidenceRecord, snapshot.id)
        before = (record.payload, record.content_sha256)
    rows = (await harness.store.scan()).rows
    await harness.store.delete_observed([row.milvus_pk for row in rows])
    assert (await harness.checker().check(root=harness.root)).successful
    write_inventory(harness.root, [entry("other.md")])
    await harness.service().ingest(harness.root)
    assert (await harness.checker().check()).successful
    async with harness.database.session() as session:
        record = await session.get(DataEvidenceRecord, snapshot.id)
        assert (record.payload, record.content_sha256) == before
    assert await service.find(identity, snapshot.id) == snapshot
    # Knowledge original/generation text snapshots remain Step 4.9 acceptance.


async def test_check_and_repeated_fix_are_zero_write(
    consistency_harness: ConsistencyHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = consistency_harness
    await harness.service().ingest(harness.root)
    forbidden = AsyncMock(side_effect=AssertionError("Clean check must not mutate"))
    monkeypatch.setattr(harness.store, "insert", forbidden)
    monkeypatch.setattr(harness.store, "delete_observed", forbidden)
    monkeypatch.setattr(DocumentRepository, "save", forbidden)
    monkeypatch.setattr(harness.store, "ensure_collection", forbidden)
    for root in (None, Path("/source-not-required")):
        assert (await harness.checker(models=False).check(root=root)).successful
    forbidden.assert_not_awaited()


@pytest.mark.parametrize("damage", ["changed", "missing", "corrupt"])
async def test_unreproducible_source_retains_version_and_other_repairs_continue(
    consistency_harness: ConsistencyHarness, damage: str
) -> None:
    harness = consistency_harness
    initial = await harness.service().ingest(harness.root)
    before = await harness.chunks(document_id("rule.md"))
    rows = (await harness.store.scan()).rows
    await harness.store.delete_observed([row.milvus_pk for row in rows])
    source = harness.root / "rule.md"
    if damage == "changed":
        source.write_text(markdown_source(body="# 新版本\n\n变化后的原文。"))
    elif damage == "missing":
        source.rename(harness.root / "held-source.md")
    else:
        source.write_text("malformed source")
    report = await harness.checker().check(root=harness.root)
    assert not report.successful
    assert report.corpus_version == initial.corpus_version
    assert report.documents_repaired == 1
    assert report.blocked[0].document_id == document_id("rule.md")
    assert await harness.chunks(document_id("rule.md")) == before
    assert await harness.store.identities(document_id("other.md"))


async def test_frozen_splitter_restored_instead_of_current_defaults(
    consistency_harness: ConsistencyHarness,
) -> None:
    harness = consistency_harness
    await harness.service(IngestionSettings(child_size=130, child_overlap=10)).ingest(harness.root)
    before = (await harness.store.scan()).rows
    await harness.store.delete_observed([row.milvus_pk for row in before])
    report = await harness.checker().check(root=harness.root)
    assert report.successful
    assert {row.chunk_uuid for row in (await harness.store.scan()).rows} == {
        row.chunk_uuid for row in before
    }


async def test_unsupported_frozen_splitter_blocks_repair(
    consistency_harness: ConsistencyHarness,
) -> None:
    harness = consistency_harness
    await harness.service().ingest(harness.root)
    async with harness.database.session() as session, session.begin():
        document_repo = DocumentRepository(session)
        manifest = await document_repo.manifest()
        config = canonical({"splitter": "unsupported-v0"})
        version = digest(config)
        members = [member.model_copy(update={"chunking_version": version}) for member in manifest.members]
        corpus_version = digest(canonical([
            (str(item.document_id), item.document_version, item.chunking_version) for item in members
        ]))
        changed = manifest.model_copy(update={
            "members": members, "splitter_configs": {version: config}, "corpus_version": corpus_version
        })
        await session.execute(update(Document).values(chunking_version=version))
        await document_repo.save_manifest(changed)
    report = await harness.checker().check(root=harness.root)
    assert not report.successful
    assert report.blocked
    assert report.chunks_inserted == report.chunks_deleted == 0


@pytest.mark.parametrize("failure", ["unavailable", "profile", "no_config"])
async def test_encoding_failure_preserves_registry_and_index(
    consistency_harness: ConsistencyHarness, failure: str
) -> None:
    harness = consistency_harness
    await harness.service().ingest(harness.root)
    before = await harness.chunks(document_id("rule.md"))
    await harness.point_to(before[0].chunk_uuid, None)
    indexed = (await harness.store.scan()).rows
    if failure == "unavailable":
        harness.embeddings.fail_at = len(harness.embeddings.calls) + 1
    elif failure == "profile":
        harness.embeddings.metadata.embed_max_length = 256
    report = await harness.checker(models=failure != "no_config").check(root=harness.root)
    assert not report.successful
    assert report.blocked
    assert report.chunks_inserted == report.chunks_deleted == 0
    assert (await harness.store.scan()).rows == indexed


async def test_lock_conflict_blocks_check_and_fix(consistency_harness: ConsistencyHarness) -> None:
    harness = consistency_harness
    async with harness.database.session() as session, session.begin():
        assert await DocumentRepository(session).try_lock()
        for root in (None, harness.root):
            with pytest.raises(IngestionAlreadyRunning):
                await harness.checker().check(root=root)


async def test_registry_commit_failure_is_atomic_and_retryable(
    consistency_harness: ConsistencyHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = consistency_harness
    initial = await harness.service().ingest(harness.root)
    before = await harness.chunks(document_id("rule.md"))
    await harness.point_to(before[0].chunk_uuid, None)
    broken = await harness.chunks(document_id("rule.md"))
    original = ChunkRepository.replace

    async def fail(self: ChunkRepository, identifier: object, values: list[RegisteredChunk]) -> None:
        await original(self, identifier, values)
        await self.session.execute(text("SELECT 1 / 0"))

    with monkeypatch.context() as patch:
        patch.setattr(ChunkRepository, "replace", fail)
        with pytest.raises(InsightPilotError):
            await harness.checker().check(root=harness.root)
    assert await harness.chunks(document_id("rule.md")) == broken
    async with harness.database.session() as session:
        assert (await DocumentRepository(session).manifest()).corpus_version == initial.corpus_version
    assert (await harness.checker().check(root=harness.root)).successful


async def test_crash_after_commit_retries_cleanup_without_encoding(
    consistency_harness: ConsistencyHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = consistency_harness
    await harness.service().ingest(harness.root)
    before = await harness.chunks(document_id("rule.md"))
    await harness.point_to(before[0].chunk_uuid, None)
    with monkeypatch.context() as patch:
        patch.setattr(harness.store, "delete_observed", AsyncMock(side_effect=asyncio.CancelledError()))
        with pytest.raises(asyncio.CancelledError):
            await harness.checker().check(root=harness.root)
    assert any(item.cleanup_pending for item in await harness.documents())
    calls = len(harness.embeddings.calls)
    report = await harness.checker(models=False).check(root=Path("/not-needed"))
    assert report.successful
    assert report.chunks_inserted == 0
    assert len(harness.embeddings.calls) == calls
    assert (await harness.checker(models=False).check(root=Path("/not-needed"))).successful


async def test_unacknowledged_cleanup_cannot_pass(
    consistency_harness: ConsistencyHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = consistency_harness
    await harness.service().ingest(harness.root)
    await harness.store.insert([(await harness.vectors())[0]])
    with monkeypatch.context() as patch:
        patch.setattr(harness.store, "delete_observed", AsyncMock(return_value=1))
        report = await harness.checker().check(root=harness.root)
    assert not report.successful
    assert report.remaining
    assert (await harness.checker(models=False).check(root=harness.root)).successful


async def test_damaged_manifest_refuses_check_and_repair(
    consistency_harness: ConsistencyHarness,
) -> None:
    harness = consistency_harness
    await harness.service().ingest(harness.root)
    async with harness.database.session() as session, session.begin():
        await session.execute(update(CorpusManifestRecord).values(corpus_version="0" * 64))
    for root in (None, harness.root):
        with pytest.raises(IngestionRegistryError):
            await harness.checker().check(root=root)


async def test_cli_run_repairs_then_reports_clean_check(
    consistency_harness: ConsistencyHarness,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from app.clients.model_runtime import ModelRuntimeClient
    from scripts import check_consistency as cli

    harness = consistency_harness
    await harness.service().ingest(harness.root)
    chunk = (await harness.chunks(document_id("rule.md")))[0]
    await harness.point_to(chunk.chunk_uuid, None)
    settings = cli.ConsistencyProcessSettings(
        _env_file=None,
        database=harness.database._settings,
        retrieval={"milvus": harness.store.settings},
        model_runtime=harness.model_settings,
    )
    monkeypatch.setattr(cli, "ModelRuntimeClient", lambda cfg: ModelRuntimeClient(cfg, harness.client))
    assert await cli.run(settings, None) == 1
    assert await cli.run(settings, harness.root) == 0
    assert await cli.run(settings, None) == 0
    assert '"successful":true' in capsys.readouterr().out
