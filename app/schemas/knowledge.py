"""Immutable knowledge snapshots and typed generation boundaries; no storage or LLM I/O."""

from datetime import date
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from app.retrieval.config import EvidenceConfig, FilterConfig, RetrievalConfig
from app.schemas.ingestion import Digest
from app.schemas.mcp import Contract
from app.schemas.model_runtime import ModelFailureKind, ModelMetadata, Score, Text
from app.schemas.retrieval import (
    PointTimeScope,
    PolicyPeriod,
    RetrievalScores,
    RetrievalTimings,
    SourcePath,
)


class FrozenContract(Contract):
    """Use tuples and frozen nested projections as well as frozen top-level fields."""

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    schema_version: Literal[1] = 1


class FrozenFilterConfig(FilterConfig):
    """Freeze scalar filter settings without changing runtime configuration mutability."""

    model_config = ConfigDict(frozen=True)


class FrozenRetrievalConfig(RetrievalConfig):
    """Every nested setting is immutable, not just this object's attributes."""

    model_config = ConfigDict(frozen=True)
    filtering: FrozenFilterConfig = Field(default_factory=FrozenFilterConfig)


class FrozenEvidenceConfig(EvidenceConfig):
    """Budget used for this exact snapshot."""

    model_config = ConfigDict(frozen=True)


class FrozenScores(RetrievalScores):
    """Unknown arm scores remain unknown in historical evidence."""

    model_config = ConfigDict(frozen=True)


class FrozenModelMetadata(ModelMetadata):
    """Exact model identities and effective parameters, containing only scalars."""

    model_config = ConfigDict(frozen=True)


class FrozenTimings(RetrievalTimings):
    """Original client-side retrieval timings."""

    model_config = ConfigDict(frozen=True)


class FrozenPointScope(PointTimeScope):
    """An immutable business date."""

    model_config = ConfigDict(frozen=True)


class FrozenPeriod(PolicyPeriod):
    """Preserve each original comparison label and half-open period."""

    model_config = ConfigDict(frozen=True)


class FrozenRangeScope(FrozenContract):
    """No mutable list is retained inside the snapshot."""

    kind: Literal["ranges"] = "ranges"
    periods: tuple[FrozenPeriod, ...] = Field(min_length=1, max_length=24)


FrozenTimeScope = Annotated[FrozenPointScope | FrozenRangeScope, Field(discriminator="kind")]


class InjectionFlag(StrEnum):
    """Heuristic observations are data, never grounds for silently dropping evidence."""

    IGNORE_CHINESE = "ignore_chinese"
    IGNORE_PREVIOUS = "ignore_previous"
    SYSTEM_ROLE = "system_role"
    ROLE_OVERRIDE = "role_override"


class TextSelection(StrEnum):
    """Whole-text selection is independent of optional sentence compression."""

    PARENT = "parent"
    CHILD = "child"
    OMITTED_BUDGET = "omitted_budget"


class EvidenceDecision(FrozenContract):
    """Account for every candidate, including suspicious candidates omitted by budget."""

    chunk_id: UUID
    selection: TextSelection
    injection_flags: tuple[InjectionFlag, ...] = ()


class EvidenceChunk(FrozenContract):
    """Self-contained selected text; future history reads never consult live sources."""

    chunk_id: UUID
    document_id: UUID
    document_title: str = Field(min_length=1, max_length=200)
    source_path: SourcePath
    document_version: Digest
    chunking_version: Digest
    content_sha256: Digest
    heading_path: str = Field(max_length=512)
    page: int | None = Field(ge=1)
    effective_from: date | None
    effective_to: date | None
    original_text: str = Field(min_length=1, max_length=65535, repr=False)
    generation_text: str = Field(min_length=1, max_length=65535, repr=False)
    text_selection: Literal[TextSelection.PARENT, TextSelection.CHILD]
    scores: FrozenScores
    injection_flags: tuple[InjectionFlag, ...] = ()
    compressed: Literal[False] = False

    @model_validator(mode="after")
    def intact_text(self) -> Self:
        """Core packaging cannot claim uncompressed text after silently editing it."""
        if self.original_text != self.generation_text:
            raise PydanticCustomError("knowledge_text", "Uncompressed text must be identical")
        if (
            self.effective_from is not None
            and self.effective_to is not None
            and self.effective_from >= self.effective_to
        ):
            raise PydanticCustomError("knowledge_interval", "Invalid evidence validity")
        return self


class KnowledgeEvidence(FrozenContract):
    """A bounded immutable payload; durable knowledge storage is introduced in Step 4.9."""

    query_used: Text
    time_scope: FrozenTimeScope
    assumptions: tuple[Text, ...] = Field(default=(), max_length=24)
    corpus_version: Digest | None
    chunks: tuple[EvidenceChunk, ...] = Field(max_length=100)
    retrieval_config: FrozenRetrievalConfig
    packaging_config: FrozenEvidenceConfig
    top_rerank_score: Score | None
    meets_floor: bool | None
    reranked: bool
    degradation: ModelFailureKind | None
    model_metadata: FrozenModelMetadata | None
    rerank_metadata: FrozenModelMetadata | None
    timings: FrozenTimings
    compressed: Literal[False] = False
    decisions: tuple[EvidenceDecision, ...] = Field(max_length=100)
    generation_block: str = Field(max_length=256_000, repr=False)
    generation_tokens: int = Field(ge=0, le=6000)
    tokenizer: Literal["cl100k_base"] = "cl100k_base"

    @model_validator(mode="after")
    def consistent_selection(self) -> Self:
        """Reject duplicated or unaccounted citations and impossible snapshot bounds."""
        ids = [item.chunk_id for item in self.chunks]
        decisions = [item.chunk_id for item in self.decisions]
        selected = [
            item.chunk_id
            for item in self.decisions
            if item.selection is not TextSelection.OMITTED_BUDGET
        ]
        if len(ids) != len(set(ids)) or len(decisions) != len(set(decisions)) or ids != selected:
            raise PydanticCustomError("knowledge_ids", "Inconsistent evidence selection")
        if self.chunks and (self.corpus_version is None or self.meets_floor is False):
            raise PydanticCustomError("knowledge_corpus", "Invalid nonempty evidence")
        if self.generation_tokens > self.packaging_config.max_tokens:
            raise PydanticCustomError("knowledge_budget", "Evidence exceeds its budget")
        if bool(self.chunks) != bool(self.generation_block):
            raise PydanticCustomError("knowledge_block", "Inconsistent generation block")
        return self


class KnowledgePassage(FrozenContract):
    """The model writes prose and IDs only, never authoritative citation metadata."""

    text: str = Field(min_length=1, max_length=4000)
    chunk_ids: tuple[UUID, ...] = Field(min_length=1, max_length=100)


class KnowledgeDraft(FrozenContract):
    """Empty passages allow the model to state that available evidence is insufficient."""

    passages: tuple[KnowledgePassage, ...] = Field(max_length=16)


class Citation(FrozenContract):
    """Authoritative display metadata resolved by the program from selected evidence."""

    chunk_id: UUID
    document_id: UUID
    document_title: str = Field(min_length=1, max_length=200)
    source_path: SourcePath
    heading_path: str = Field(max_length=512)
    page: int | None = Field(ge=1)


class KnowledgeAbstention(StrEnum):
    """No evidence, budget exhaustion and invalid generated references are distinct."""

    NO_EVIDENCE = "no_evidence"
    BUDGET_EXHAUSTED = "budget_exhausted"
    UNSUPPORTED = "unsupported"
    FABRICATED_CITATION = "fabricated_citation"


class KnowledgeGeneration(FrozenContract):
    """Internal result only; the production chat envelope remains unchanged."""

    passages: tuple[KnowledgePassage, ...] = ()
    citations: tuple[Citation, ...] = ()
    abstention: KnowledgeAbstention | None = None
    attempts: int = Field(ge=0, le=2)

    @model_validator(mode="after")
    def terminal_shape(self) -> Self:
        """An abstention can never leak a previously rejected draft."""
        if self.abstention is not None:
            if self.passages or self.citations:
                raise PydanticCustomError("knowledge_abstention", "Abstention contains an answer")
        elif not self.passages or not self.citations:
            raise PydanticCustomError("knowledge_answer", "Answer requires cited passages")
        return self
