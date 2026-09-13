"""Versioned retrieval contracts, independent of the Milvus SDK and model libraries."""

from datetime import date
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import AfterValidator, Field, FiniteFloat, model_validator
from pydantic_core import PydanticCustomError

from app.retrieval.config import RetrievalConfig
from app.schemas.corpus import DocumentType, relative_source
from app.schemas.ingestion import ChunkIdentity, Digest
from app.schemas.mcp import Contract
from app.schemas.model_runtime import Dense, ModelFailureKind, ModelMetadata, RerankResult, Score, Sparse, Text

SourcePath = Annotated[str, Field(min_length=1, max_length=1024), AfterValidator(relative_source)]


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
    rerank: Score | None = None


class CandidateSource(RetrievalContract):
    """Registered source path, read in the candidate admission snapshot."""

    document_id: UUID
    source_path: SourcePath


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
    source_path: SourcePath | None = None

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
    rerank_ms: int = Field(default=0, ge=0)
    filter_ms: int = Field(default=0, ge=0)
    total_ms: int = Field(default=0, ge=0)


class RetrievalStage(StrEnum):
    """Stable diagnostics; never infer stage state from human-readable prose."""

    SEARCH = "search"
    ADMISSION = "admission"
    RERANK = "rerank"
    THRESHOLD = "threshold"
    DIVERSITY = "diversity"
    TRUNCATION = "truncation"


class StageStatus(StrEnum):
    """Skipped work and failed work are distinct from successfully empty output."""

    COMPLETED = "completed"
    DISABLED = "disabled"
    EMPTY = "empty"
    DEGRADED = "degraded"


class ScoreRange(RetrievalContract):
    """An absent stage score has no fabricated minimum or maximum."""

    minimum: FiniteFloat
    maximum: FiniteFloat


class StageDiagnostic(RetrievalContract):
    """Safe per-stage counts, native score scales and client wall time."""

    stage: RetrievalStage
    status: StageStatus = StageStatus.COMPLETED
    input_count: int = Field(ge=0)
    output_count: int = Field(ge=0)
    elapsed_ms: int = Field(default=0, ge=0)
    score_ranges: dict[Literal["dense", "sparse_learned", "sparse_bm25", "rrf", "rerank"], ScoreRange]


class RankingResult(RetrievalContract):
    """Shared post-search stage, usable by live benchmarks without storage substitutes."""

    candidates: list[Candidate]
    reranked: bool = False
    degradation: ModelFailureKind | None = None
    top_rerank_score: Score | None = None
    meets_floor: bool | None = None
    response: RerankResult | None = None
    stages: list[StageDiagnostic] = Field(default_factory=list)


class RetrievalResult(RetrievalContract):
    """Registered retrieval results and explicit ranking state; evidence packaging is separate."""

    query: RetrievalQuery
    corpus_version: Digest | None
    candidates: list[Candidate]
    retrieval_config: RetrievalConfig
    model_metadata: ModelMetadata | None
    timings: RetrievalTimings
    reranked: bool = False
    degradation: ModelFailureKind | None = None
    top_rerank_score: Score | None = None
    meets_floor: bool | None = None
    rerank_metadata: ModelMetadata | None = None
    stages: list[StageDiagnostic] = Field(default_factory=list)
