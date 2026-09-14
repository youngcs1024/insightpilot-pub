"""Committed real storage plus deterministic HTTP reranking; no live GPU claims."""
# ruff: noqa: PLR2004 -- fixed scores and batch contracts are acceptance expectations.

import pytest
from sqlalchemy import update

from app.db.models.document import Document
from app.repositories.document import DocumentRepository
from app.retrieval.config import RetrievalConfig
from app.schemas.ingestion import ActiveManifest
from app.schemas.retrieval import RetrievalStage, StageStatus
from model_runtime.errors import ModelContractError, ModelError
from tests.retrieval_support import RetrievalHarness, deadline, harness, query

pytestmark = [pytest.mark.integration, pytest.mark.storage]
__all__ = ["harness"]


async def test_rerank_is_single_batched_call(harness: RetrievalHarness) -> None:
    original = await harness.pipeline().retrieve(query(), deadline=deadline())
    result = await harness.pipeline(RetrievalConfig(record_arm_scores=True)).retrieve(
        query(), deadline=deadline()
    )
    calls = harness.embeddings.rerank_calls
    assert len(calls) == 1
    assert calls[0]["passages"] == [item.content for item in original.candidates]
    assert calls[0]["max_length"] == 320
    assert result.reranked
    assert result.meets_floor
    assert result.rerank_metadata == result.model_metadata
    assert result.stages[2].input_count == len(original.candidates)
    assert all(item.source_path for item in result.candidates)
    assert all(
        item.scores.rrf is not None and item.scores.rerank == 0.8 for item in result.candidates
    )
    assert any(item.parent_content != item.content for item in result.candidates)
    assert [item.stage for item in result.stages] == list(RetrievalStage)
    assert result.stages[-1].output_count == len(result.candidates)
    assert result.timings.total_ms >= result.timings.rerank_ms


async def test_degrades_when_model_runtime_down(harness: RetrievalHarness) -> None:
    harness.embeddings.rerank_error = ModelError()
    result = await harness.pipeline(RetrievalConfig()).retrieve(query(), deadline=deadline())
    assert result.candidates
    assert not result.reranked
    assert result.degradation == ModelError.kind
    assert result.meets_floor is None
    assert result.top_rerank_score is None
    assert all(item.scores.rerank is None for item in result.candidates)
    assert result.stages[2].status is StageStatus.DEGRADED


async def test_bad_rerank_contract_fails_closed(harness: RetrievalHarness) -> None:
    harness.embeddings.rerank_scores = []
    with pytest.raises(ModelContractError):
        await harness.pipeline(RetrievalConfig()).retrieve(query(), deadline=deadline())


async def test_below_floor_succeeds_without_candidates(harness: RetrievalHarness) -> None:
    pool = await harness.pipeline().retrieve(query(), deadline=deadline())
    harness.embeddings.rerank_scores = [0.1] * len(pool.candidates)
    result = await harness.pipeline(RetrievalConfig()).retrieve(query(), deadline=deadline())
    assert result.candidates == []
    assert result.reranked
    assert result.meets_floor is False
    assert result.top_rerank_score == 0.1
    assert result.stages[3].output_count == 0


async def test_source_path_uses_same_snapshot_as_manifest(
    harness: RetrievalHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = DocumentRepository.manifest

    async def rename_after_snapshot(repository: DocumentRepository) -> ActiveManifest | None:
        manifest = await original(repository)
        async with harness.database.session() as writer, writer.begin():
            await writer.execute(
                update(Document)
                .where(Document.source_path == "sku.md")
                .values(source_path="renamed.md")
            )
        return manifest

    monkeypatch.setattr(DocumentRepository, "manifest", rename_after_snapshot)
    result = await harness.pipeline(RetrievalConfig()).retrieve(query(), deadline=deadline())
    assert "sku.md" in {item.source_path for item in result.candidates}
    assert "renamed.md" not in {item.source_path for item in result.candidates}
