"""Shared deterministic evidence fixtures; no test-module imports or import-time I/O."""

from datetime import date
from uuid import UUID

from app.retrieval.config import RetrievalConfig
from app.schemas.corpus import DocumentType
from app.schemas.ingestion import ChunkIdentity
from app.schemas.knowledge import KnowledgeDraft, KnowledgePassage
from app.schemas.retrieval import Candidate, CandidateProvenance, RetrievalResult, RetrievalTimings
from tests.retrieval_support import candidate, query


def provenance(value: ChunkIdentity) -> CandidateProvenance:
    """Unit admission substitutes can project identities without constructing an SDK hit."""
    return CandidateProvenance(
        **value.model_dump(include=set(ChunkIdentity.model_fields)),
        source_path=getattr(value, "source_path", None) or "source.md",
        document_title="退款规则",
        heading_path=getattr(value, "heading_path", "政策"),
        page=None,
        doc_type=getattr(value, "doc_type", DocumentType.POLICY),
        effective_from=getattr(value, "effective_from", None),
        effective_to=getattr(value, "effective_to", None),
    )


def retrieval(values: list[Candidate] | None = None) -> RetrievalResult:
    """Explicitly empty input remains empty, with unknown ranking diagnostics."""
    values = [candidate()] if values is None else values
    return RetrievalResult(
        query=query(),
        corpus_version="a" * 64,
        candidates=values,
        retrieval_config=RetrievalConfig(use_rerank=False),
        model_metadata=None,
        timings=RetrievalTimings(),
        provenance=[provenance(value) for value in values],
    )


def draft(*identifiers: UUID, text: str = "七天内可申请退款。") -> KnowledgeDraft:
    return KnowledgeDraft(passages=(KnowledgePassage(text=text, chunk_ids=identifiers),))


def dated_candidate() -> Candidate:
    value = candidate()
    value.effective_from = date(2026, 7, 1)
    value.effective_to = date(2026, 8, 1)
    return value
