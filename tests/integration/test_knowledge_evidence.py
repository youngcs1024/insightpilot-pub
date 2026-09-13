"""Real registered metadata and vector text produce detached evidence before graph assembly."""

import pytest
from sqlalchemy import update

from app.db.models.chunk import Chunk
from app.db.models.document import Document
from app.repositories.document import DocumentRepository
from app.retrieval.config import EvidenceConfig, RetrievalConfig
from app.retrieval.evidence import package_evidence
from app.schemas.corpus import CorpusMetadata
from app.schemas.ingestion import ActiveManifest
from app.schemas.knowledge import KnowledgeEvidence
from app.services.knowledge_generation import KnowledgeGenerationService
from app.services.schema_tokens import SchemaTokenCounter
from tests.fakes.chat_model import FakeChatModel
from tests.knowledge_support import draft
from tests.milvus_support import milvus_stack
from tests.retrieval_support import RetrievalHarness, deadline, harness, query

pytestmark = [pytest.mark.integration, pytest.mark.storage]
__all__ = ["harness", "milvus_stack"]


async def test_real_retrieval_packages_scores_metadata_and_generates_citations(harness: RetrievalHarness) -> None:
    result = await harness.pipeline(RetrievalConfig(record_arm_scores=True)).retrieve(query(), deadline=deadline())
    assert result.schema_version == 2
    evidence = package_evidence(result, EvidenceConfig(), SchemaTokenCounter())
    assert evidence.chunks and evidence.reranked and evidence.meets_floor
    assert evidence.corpus_version == result.corpus_version
    assert evidence.chunks[0].scores.rrf is not None
    assert evidence.chunks[0].scores.rerank is not None
    assert all(chunk.document_title == "测试规则" for chunk in evidence.chunks)
    assert all(chunk.page is None for chunk in evidence.chunks)
    assert any(chunk.original_text != next(item.content for item in result.candidates if item.chunk_uuid == chunk.chunk_id) for chunk in evidence.chunks)
    llm = FakeChatModel([draft(evidence.chunks[0].chunk_id)])
    answer = await KnowledgeGenerationService(llm).generate(evidence, deadline=deadline())
    assert answer.citations[0].document_title == "测试规则"
    assert answer.citations[0].source_path == evidence.chunks[0].source_path
    assert len(harness.embeddings.rerank_calls) == 1


async def test_title_page_and_path_share_admission_snapshot(harness: RetrievalHarness, monkeypatch: pytest.MonkeyPatch) -> None:
    original = DocumentRepository.manifest
    async with harness.database.session() as session, session.begin():
        documents = await DocumentRepository(session).list_documents()
        await session.execute(update(Chunk).values(page=2))
    renamed = CorpusMetadata.model_validate(documents[0].metadata.model_dump())
    renamed.title = "并发替换的新标题"

    async def mutate_after_snapshot(repository: DocumentRepository) -> ActiveManifest | None:
        manifest = await original(repository)
        async with harness.database.session() as writer, writer.begin():
            await writer.execute(update(Document).where(Document.id == documents[0].document_id).values(business_metadata=renamed.model_dump_json(), source_path="renamed.md"))
            await writer.execute(update(Chunk).values(page=9))
        return manifest

    monkeypatch.setattr(DocumentRepository, "manifest", mutate_after_snapshot)
    result = await harness.pipeline().retrieve(query(), deadline=deadline())
    evidence = package_evidence(result, EvidenceConfig(), SchemaTokenCounter())
    assert evidence.chunks
    assert all(chunk.document_title == "测试规则" and chunk.page == 2 for chunk in evidence.chunks)
    assert all(chunk.source_path != "renamed.md" for chunk in evidence.chunks)
    assert any(chunk.document_id == documents[0].document_id for chunk in evidence.chunks)
    assert all(source.page == 2 for source in result.provenance)


async def test_reingestion_cannot_change_packaged_text(harness: RetrievalHarness) -> None:
    result = await harness.pipeline().retrieve(query(), deadline=deadline())
    evidence = package_evidence(result, EvidenceConfig(), SchemaTokenCounter())
    serialized = evidence.model_dump_json()
    for path in (harness.root / "sku.md", harness.root / "august.md"):
        path.write_text(path.read_text().replace("退款规则", "替换后的政策"))
    await harness.ingest()
    # The stored payload itself is the read authority; no registry lookup is needed.
    restored = KnowledgeEvidence.model_validate_json(serialized)
    assert restored.model_dump_json() == evidence.model_dump_json()
    assert "替换后的政策" not in restored.generation_block
    llm = FakeChatModel([draft(restored.chunks[0].chunk_id)])
    answer = await KnowledgeGenerationService(llm).generate(restored, deadline=deadline())
    assert answer.citations[0].chunk_id == restored.chunks[0].chunk_id
