"""Fresh per-assistant-turn state with separate node-owned outputs."""

import operator
from typing import Annotated, Literal, Self

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from app.agents.contracts import (
    Answer,
    DataEvidence,
    EvidenceRefs,
    PreparedContext,
    RewrittenQuestion,
    RouteDecision,
    RoutingContext,
    SynthesisResult,
    TurnIdentity,
)
from app.agents.failures import NodeFailure
from app.schemas.knowledge import KnowledgeEvidence
from app.schemas.knowledge_query import KnowledgeClarification, KnowledgeHistoryTurn
from app.schemas.mcp import Contract
from app.schemas.memory import FormatPreferenceContent, Memory
from app.schemas.memory_retrieval import MemorySelection
from app.schemas.metric_resolution import (
    MetricClarification,
    MetricIntent,
    MetricPatches,
    RegionScope,
    SelectedOverrides,
)
from app.schemas.retrieval import KnowledgeTimeScope
from app.services.periods import Period

GRAPH_VERSION: Literal["phase6-v1"] = "phase6-v1"


class TurnContext(Contract):
    """Written once after routing; downstream projections must copy nested values.

    Freezing prevents field replacement. Lists retain the shared wire contracts;
    their contents are read-only by ownership, not recursively frozen by Pydantic.
    """

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    schema_version: Literal[1, 2] = 2
    prepared_intent: MetricIntent | None = None
    intent_failure: NodeFailure | None = None
    recent_messages: list[AnyMessage] = Field(default_factory=list)
    summary: str = Field(default="", max_length=32_000)
    memories: list[Memory] = Field(default_factory=list, max_length=5)
    time_scope: KnowledgeTimeScope | None
    region_scope: RegionScope | None = None
    selected_overrides: SelectedOverrides = Field(default_factory=SelectedOverrides)
    format_preference: FormatPreferenceContent | None = None
    prior_sql: list[str] = Field(default_factory=list, max_length=3)
    explicit_patch: MetricPatches = Field(default_factory=MetricPatches)
    reference_period: Period | None = None
    knowledge_history: list[KnowledgeHistoryTurn] = Field(default_factory=list, max_length=3)
    token_accounting: dict[str, Annotated[int, Field(ge=0)]] = Field(default_factory=dict)


class GraphInput(TurnIdentity):
    """A question is loaded from its admitted user message, never caller-overridden."""

    graph_version: Literal["phase6-v1"] = GRAPH_VERSION


class GraphOutput(Contract):
    """Typed success/failure projection for the future ChatService."""

    schema_version: Literal[1] = 1
    route: RouteDecision | None = None
    data_evidence: DataEvidence | None = None
    answer: Answer | None = None
    clarification: MetricClarification | None = None
    evidence_refs: EvidenceRefs | None = None
    failures: list[NodeFailure] = Field(default_factory=list)
    status: Literal["succeeded", "degraded", "abstained", "failed"] = "failed"

    @model_validator(mode="after")
    def separate_clarification(self) -> Self:
        """A clarification must never contain a fabricated analytical answer."""
        if self.clarification is not None and (
            (
                self.answer is not None
                and (
                    not self.answer.abstained
                    or self.answer.claims
                    or self.answer.citations
                    or self.answer.sql
                    or self.answer.assumptions
                    or self.answer.evidence_refs != EvidenceRefs()
                )
            )
            or self.data_evidence is not None
            or (
                self.evidence_refs is not None
                and (
                    self.evidence_refs.data_snapshot_id is not None
                    or self.evidence_refs.knowledge_snapshot_id is not None
                )
            )
        ):
            raise PydanticCustomError(
                "graph_clarification", "Clarification cannot contain analysis"
            )
        return self


class AgentState(GraphInput):
    """No service, credential, runtime object or checkpoint from another turn."""

    graph_version: Literal["phase6-v1"] = GRAPH_VERSION
    # prepare owns the loaded question/history. GraphInput cannot supply them.
    question: str = Field(default="", max_length=32_000)
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    routing_context: RoutingContext | None = None
    memory_preselection: MemorySelection = Field(default_factory=MemorySelection)
    memory_disabled: bool = False
    memory_restart_pending: bool = False
    # Reserved owners: finalize_context, router, knowledge wrapper (Steps 4.3-4.4).
    context: TurnContext | None = None
    context_clarification: MetricClarification | None = None
    route: RouteDecision | None = None
    knowledge_evidence: KnowledgeEvidence | None = None
    knowledge_clarification: KnowledgeClarification | None = None
    knowledge_abstention_reason: str | None = Field(default=None, min_length=1, max_length=1000)
    route_clarification: MetricClarification | None = None
    data_clarification: MetricClarification | None = None
    assumptions: Annotated[list[str], operator.add] = Field(default_factory=list)
    degraded_components: Annotated[list[str], operator.add] = Field(default_factory=list)
    abstained: bool = False
    synthesis: SynthesisResult | None = None
    prepared: PreparedContext | None = None
    rewritten: RewrittenQuestion | None = None
    data_evidence: DataEvidence | None = None
    evidence_refs: EvidenceRefs | None = None
    answer: Answer | None = None
    clarification: MetricClarification | None = None
    failures: Annotated[list[NodeFailure], operator.add] = Field(default_factory=list)
    status: Literal["succeeded", "degraded", "abstained", "failed"] = "failed"
