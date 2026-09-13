"""Versioned retrieval contracts, independent of the Milvus SDK and model libraries."""

from datetime import date
from typing import Annotated, Literal, Self

from pydantic import Field, FiniteFloat, model_validator
from pydantic_core import PydanticCustomError

from app.retrieval.config import RetrievalConfig
from app.schemas.corpus import DocumentType
from app.schemas.ingestion import ChunkIdentity, Digest
from app.schemas.mcp import Contract
from app.schemas.model_runtime import Dense, ModelMetadata, Sparse, Text


class RetrievalContract(Contract):
    """Initial released retrieval payload schema."""

    schema_version: Literal[1] = 1


class PolicyPeriod(RetrievalContract):
    """One labelled half-open interval, retained even when search intervals merge."""

    start: date
    end: date
    label: str = Field(default="", max_length=200)

    @model_validator(mode="after")
    def ordered(self) -> Self:
        """Reject explicit invalid intervals rather than widening the search."""
        if self.start >= self.end:
            raise PydanticCustomError("policy_period", "Period start must precede end")
        return self


class PointTimeScope(RetrievalContract):
    """A business calendar date, not a host-local timestamp."""

    kind: Literal["point"] = "point"
    as_of: date


class RangeTimeScope(RetrievalContract):
    """Original labelled intervals; normalization is a search-only projection."""

    kind: Literal["ranges"] = "ranges"
    periods: list[PolicyPeriod] = Field(min_length=1, max_length=24)


KnowledgeTimeScope = Annotated[PointTimeScope | RangeTimeScope, Field(discriminator="kind")]


class RetrievalQuery(RetrievalContract):
    """Time is explicit at this boundary; natural-language resolution is Step 3.11."""

    standalone: Text
    time_scope: KnowledgeTimeScope
    assumptions: list[str] = Field(default_factory=list, max_length=24)


class EncodedQuery(RetrievalContract):
    """BM25-only queries have no fabricated model output or metadata."""

    text: Text
    dense: Dense | None = None
    sparse: Sparse | None = None
    metadata: ModelMetadata | None = None


class RetrievalScores(RetrievalContract):
    """Native scores retain their scales; an absent arm hit is unknown, never zero."""

    dense: FiniteFloat | None = None
    sparse_learned: FiniteFloat | None = None
    sparse_bm25: FiniteFloat | None = None
    rrf: FiniteFloat | None = None
    rerank: FiniteFloat | None = None


class Candidate(ChunkIdentity):
    """Search text and provenance remain attached to a registered physical identity."""

    schema_version: Literal[1] = 1
    milvus_pk: int = Field(strict=True, ge=0)
    content: str = Field(min_length=1, max_length=32_000)
    parent_content: str = Field(min_length=1, max_length=65535)
    heading_path: str = Field(max_length=512)
    doc_type: DocumentType
    effective_from: date | None
    effective_to: date | None
    scores: RetrievalScores = Field(default_factory=RetrievalScores)

    @model_validator(mode="after")
    def valid_interval(self) -> Self:
        """Malformed persisted bounds cannot silently become an unbounded policy."""
        if (
            self.effective_from is not None
            and self.effective_to is not None
            and self.effective_from >= self.effective_to
        ):
            raise PydanticCustomError("candidate_interval", "Invalid candidate validity")
        return self


class RetrievalTimings(RetrievalContract):
    """Client wall times include transport, queueing and candidate admission."""

    encode_ms: int = Field(default=0, ge=0)
    search_ms: int = Field(default=0, ge=0)
    admission_ms: int = Field(default=0, ge=0)
    total_ms: int = Field(default=0, ge=0)


class RetrievalResult(RetrievalContract):
    """A candidate pool, not yet reranked evidence or an answer."""

    query: RetrievalQuery
    corpus_version: Digest | None
    candidates: list[Candidate]
    retrieval_config: RetrievalConfig
    model_metadata: ModelMetadata | None
    timings: RetrievalTimings
