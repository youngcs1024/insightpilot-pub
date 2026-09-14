"""Isolated, serializable knowledge inputs and mutually exclusive terminal outcomes."""

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from app.agents.failures import NodeFailure
from app.schemas.knowledge import KnowledgeAbstention, KnowledgeEvidence
from app.schemas.mcp import Contract
from app.schemas.metric_resolution import RegionScope
from app.schemas.model_runtime import Text
from app.schemas.retrieval import KnowledgeTimeScope, RetrievalQuery, RetrievalResult


class TerminologyProjection(Contract):
    """Already-finalized terminology, without storage access or other memory types."""

    schema_version: Literal[1] = 1
    type: Literal["terminology"] = "terminology"
    term: str = Field(min_length=1, max_length=50)
    means: str = Field(min_length=1, max_length=200)


class KnowledgeAgentInput(Contract):
    """Callers supply resolved time and a standalone intent until Step 3.11."""

    schema_version: Literal[1] = 1
    question: Text
    knowledge_intent: str = Field(default="", max_length=32_000)
    time_scope: KnowledgeTimeScope
    region_scope: RegionScope | None = None
    relevant_memories: list[TerminologyProjection] = Field(default_factory=list, max_length=5)
    conversation_summary: str = Field(default="", max_length=4000)


class KnowledgeAgentOutput(Contract):
    """Only evidence, refusal or a safe failure crosses back to the caller."""

    schema_version: Literal[1] = 1
    evidence: KnowledgeEvidence | None = None
    failure: NodeFailure | None = None
    abstained: bool = False
    abstention_reason: str | None = Field(default=None, min_length=1, max_length=1000)
    degraded_components: list[Literal["rerank"]] = Field(default_factory=list, max_length=1)

    @model_validator(mode="after")
    def terminal_outcome(self) -> Self:
        """Reject ambiguous outcomes and empty successful evidence."""
        if sum((self.evidence is not None, self.failure is not None, self.abstained)) != 1:
            raise PydanticCustomError("knowledge_outcome", "Expected exactly one terminal outcome")
        if self.abstained != (self.abstention_reason is not None):
            raise PydanticCustomError("knowledge_abstention", "Refusal requires a reason")
        if self.evidence is not None:
            if not self.evidence.chunks:
                raise PydanticCustomError("knowledge_empty", "Successful evidence must be nonempty")
            expected = ["rerank"] if self.evidence.degradation is not None else []
            if self.degraded_components != expected:
                raise PydanticCustomError("knowledge_degradation", "Inconsistent degradation")
        return self


class KnowledgeAgentState(KnowledgeAgentInput):
    """Intermediate channels stay private; only finish writes the terminal projection."""

    query: RetrievalQuery | None = None
    retrieval_result: RetrievalResult | None = None
    packaged: KnowledgeEvidence | None = None
    rejection: KnowledgeAbstention | None = None
    operation_failure: NodeFailure | None = None
    evidence: KnowledgeEvidence | None = None
    failure: NodeFailure | None = None
    abstained: bool = False
    abstention_reason: str | None = None
    degraded_components: list[Literal["rerank"]] = Field(default_factory=list, max_length=1)
