"""Fresh per-assistant-turn state with separate node-owned outputs."""

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from app.agents.contracts import (
    Answer,
    DataEvidence,
    EvidenceRefs,
    PreparedContext,
    RewrittenQuestion,
    TurnIdentity,
)
from app.agents.failures import NodeFailure
from app.schemas.mcp import Contract
from app.schemas.metric_resolution import MetricClarification

GRAPH_VERSION: Literal["phase2-v2"] = "phase2-v2"


class GraphInput(TurnIdentity):
    """A question is loaded from its admitted user message, never caller-overridden."""

    graph_version: Literal["phase2-v2"] = GRAPH_VERSION


class GraphOutput(Contract):
    """Typed success/failure projection for the future ChatService."""

    schema_version: Literal[1] = 1
    data_evidence: DataEvidence | None = None
    answer: Answer | None = None
    clarification: MetricClarification | None = None
    evidence_refs: EvidenceRefs | None = None
    failures: list[NodeFailure] = Field(default_factory=list)
    status: Literal["succeeded", "failed"] = "failed"

    @model_validator(mode="after")
    def separate_clarification(self) -> Self:
        """A clarification must never contain a fabricated analytical answer."""
        if self.clarification is not None and (
            self.answer is not None
            or self.data_evidence is not None
            or self.evidence_refs is not None
        ):
            raise PydanticCustomError(
                "graph_clarification", "Clarification cannot contain analysis"
            )
        return self


class AgentState(GraphInput):
    """No service, credential, runtime object or checkpoint from another turn."""

    graph_version: Literal["phase2-v2"] = GRAPH_VERSION
    prepared: PreparedContext | None = None
    rewritten: RewrittenQuestion | None = None
    data_evidence: DataEvidence | None = None
    evidence_refs: EvidenceRefs | None = None
    answer: Answer | None = None
    clarification: MetricClarification | None = None
    failures: list[NodeFailure] = Field(default_factory=list)
    status: Literal["succeeded", "failed"] = "failed"
