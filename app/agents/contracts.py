"""Versioned graph and evidence boundaries; only JSON-safe values are durable."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator
from pydantic_core import PydanticCustomError

from app.schemas.knowledge import Citation, KnowledgeEvidence, KnowledgePassage
from app.schemas.knowledge_query import KnowledgeHistoryTurn
from app.schemas.mcp import ColumnSpec, Contract, SqlValue
from app.schemas.memory import FormatPreferenceContent, TerminologyContent
from app.schemas.metric_resolution import BindingFieldSource, RegionScope
from app.schemas.metrics import Grain, MetricExample
from app.schemas.sanity import SanityFlag
from app.schemas.synthesis import Claim, SynthesisAbstention, SynthesisOutput

__all__ = [
    "MAX_ANSWER_CHARS",
    "Answer",
    "AnswerDraft",
    "ColumnStatistics",
    "DataEvidence",
    "EvidenceRefs",
    "EvidenceSnapshot",
    "HistoryMessage",
    "MetricBinding",
    "MetricExamplesSnapshot",
    "PhaseOneSqlGeneratorOutput",
    "PreparedContext",
    "ResolvedMetricBinding",
    "ResultSummary",
    "RewrittenQuestion",
    "Route",
    "RouteDecision",
    "RouterInput",
    "RoutingContext",
    "SanityFlag",
    "SqlGeneratorOutput",
    "TurnIdentity",
]

MAX_ANSWER_CHARS = 32_000


class PhaseOneSqlGeneratorOutput(Contract):
    """A concise query rationale followed by SQL and displayed assumptions."""

    thinking: str = Field(max_length=4000)
    sql_query: str = Field(min_length=1, max_length=32000)
    assumptions: list[str] = Field(default_factory=list, max_length=20)


class ColumnStatistics(Contract):
    """Statistics over all returned rows, never just the display sample."""

    name: str
    type: str
    null_count: int = Field(ge=0)
    distinct_count: int = Field(ge=0)
    minimum: SqlValue = None
    maximum: SqlValue = None
    total: str | None = None


class ResultSummary(Contract):
    """Audit statistics/sample; generation_block records the smaller model view.

    sample_truncated describes this audit sample, not the budgeted prompt sample.
    Historical snapshots may contain more than the current 30 statistics columns.
    """

    schema_version: Literal[1] = 1
    returned_row_count: int = Field(ge=0, le=5000)
    statistics_scope: Literal["returned_rows"] = "returned_rows"
    columns: list[ColumnStatistics]
    sample_rows: list[list[SqlValue]] = Field(max_length=200)
    sample_truncated: bool
    result_truncated: bool


class MetricBinding(Contract):
    """Resolved business meaning; Phase 1 produces an empty binding list."""

    metric_key: str
    definition_version: int
    resolved_expression: str
    date_field: str
    period_start: datetime
    period_end: datetime
    filters_applied: list[str]
    override_id: UUID | None = None


class ResolvedMetricBinding(MetricBinding):
    """V2 retains a complete resolved query without changing historical V1 evidence."""

    schema_version: Literal[2] = 2
    period_start: AwareDatetime
    period_end: AwareDatetime
    grain: Grain
    region_scope: RegionScope | None = None
    resolved_description: str
    field_sources: list[BindingFieldSource]

    @model_validator(mode="after")
    def ordered_period(self) -> Self:
        """New bindings require a nonempty interval; historical V1 stays readable."""
        if self.period_start >= self.period_end:
            raise PydanticCustomError("metric_period", "Period start must precede end")
        return self


class DataEvidence(Contract):
    """The immutable, exact evidence supplied to answer generation."""

    schema_version: Literal[1] = 1
    sql: str
    dialect: Literal["postgres"] = "postgres"
    # Phase 2 introduces actual metric bindings, not invented Phase 1 definitions.
    metric_bindings: list[ResolvedMetricBinding | MetricBinding] = Field(default_factory=list)
    assumptions: list[str]
    columns: list[ColumnSpec]
    row_count: int = Field(ge=0, le=5000)
    rows: list[list[SqlValue]] = Field(max_length=200)
    result_summary: ResultSummary
    generation_block: str
    execution_ms: int = Field(ge=0)
    mcp_call_id: str
    limit_applied: bool
    sanity_flags: list[SanityFlag]


class EvidenceRefs(Contract):
    """Committed identifiers; absent specialists remain null."""

    schema_version: Literal[1] = 1
    data_snapshot_id: UUID | None = None
    knowledge_snapshot_id: UUID | None = None


class EvidenceSnapshot(Contract):
    """A persisted document, never a live data-source lookup."""

    schema_version: Literal[1] = 1
    id: UUID
    data: DataEvidence


class KnowledgeSnapshot(Contract):
    """A committed knowledge payload, independent of today's corpus."""

    schema_version: Literal[1] = 1
    id: UUID
    knowledge: KnowledgeEvidence


class EvidenceBundle(Contract):
    """Both immutable snapshots from one owned assistant turn."""

    schema_version: Literal[1] = 1
    data: EvidenceSnapshot | None = None
    knowledge: KnowledgeSnapshot | None = None

    @property
    def refs(self) -> EvidenceRefs:
        """Only committed records can supply identifiers."""
        return EvidenceRefs(
            data_snapshot_id=self.data.id if self.data else None,
            knowledge_snapshot_id=self.knowledge.id if self.knowledge else None,
        )


class SourceSummary(Contract):
    """Deterministic BOTH assembly instructions, not cross-source inference."""

    schema_version: Literal[1] = 1
    evidence_refs: EvidenceRefs
    missing_components: list[Literal["data", "knowledge"]] = Field(default_factory=list)


class AnswerDraft(Contract):
    """Only prose and confidence are generated by the model."""

    markdown: str = Field(min_length=1, max_length=MAX_ANSWER_CHARS)
    confidence: float = Field(ge=0, le=1)


class SynthesisInput(Contract):
    """No history or runtime context crosses the cross-evidence reasoning boundary."""

    schema_version: Literal[1] = 1
    question: str = Field(min_length=1, max_length=32_000)
    data: DataEvidence | None
    knowledge: KnowledgeEvidence | None
    assumptions: list[str] = Field(max_length=100)


class SynthesisResult(SynthesisOutput):
    """Only program-validated content plus committed provenance becomes durable."""

    evidence_refs: EvidenceRefs
    missing_components: list[Literal["data", "knowledge"]] = Field(default_factory=list)
    abstention: SynthesisAbstention | None = None
    attempts: int = Field(ge=0, le=2)


class DataAnswerDraft(Contract):
    """Single-source generation emits references, never final presentation fields."""

    claims: list[Claim] = Field(max_length=16)


class Answer(AnswerDraft):
    """Trusted evidence fields are assembled by the formatter."""

    schema_version: Literal[1, 2, 3] = 3
    claims: list[Claim] = Field(default_factory=list, max_length=16)
    trace_id: str | None = Field(default=None, min_length=1, max_length=128)
    format_preference: FormatPreferenceContent | None = None
    attempted_sources: list[Literal["data", "knowledge"]] = Field(default_factory=list)
    unanswered: list[str] = Field(default_factory=list, max_length=12)
    assumptions: list[str]
    sql: str
    evidence_refs: EvidenceRefs
    citations: list[Citation] = Field(default_factory=list)
    knowledge_passages: list[KnowledgePassage] = Field(default_factory=list)
    degraded_components: list[str] = Field(default_factory=list)
    abstained: bool = False
    synthesis: SynthesisResult | None = None


    @model_validator(mode="after")
    def current_trace(self) -> Self:
        """Historical payloads stay readable without inventing missing provenance."""
        if self.schema_version == 3 and self.trace_id is None:
            raise PydanticCustomError("answer_trace", "Answer v3 requires a trace ID")
        return self


class TurnIdentity(Contract):
    """Trusted caller identity must match persisted turn ownership."""

    schema_version: Literal[1] = 1
    user_id: UUID
    conversation_id: UUID
    turn_id: UUID


class HistoryMessage(Contract):
    """Application history retains roles without graph internal messages."""

    role: Literal["user", "assistant"]
    content: str


class PreparedContext(Contract):
    """Preparation's sole output; future memory remains explicitly empty."""

    schema_version: Literal[1] = 1
    has_prior_turns: bool = False
    question: str
    summary: str
    messages: list[HistoryMessage]
    prior_sql: list[str] = Field(max_length=3)
    memories: list[str] = Field(default_factory=list, max_length=0)
    knowledge_history: list[KnowledgeHistoryTurn] = Field(default_factory=list, max_length=3)


class SqlGeneratorOutput(Contract):
    """A brief design summary precedes the candidate SQL, never hidden reasoning."""

    thinking: str = Field(
        max_length=4000, description="Briefly summarize tables, joins and filters."
    )
    sql: str = Field(max_length=32000)
    tables_used: list[str] = Field(max_length=32)
    notes: str = Field(default="", max_length=4000)


class MetricExamplesSnapshot(Contract):
    """Examples from exactly the catalog version used to resolve a binding."""

    schema_version: Literal[1] = 1
    metric_key: str = Field(min_length=1, max_length=64)
    definition_version: int = Field(gt=0)
    examples: list[MetricExample] = Field(min_length=1, max_length=20)


class RewrittenQuestion(Contract):
    """Durable parent-only interpretation, never exported as trace prose."""

    schema_version: Literal[1] = 1
    standalone: str = Field(min_length=1, max_length=32_000)
    referenced_prior_turn: bool
    unresolved_references: list[str] = Field(default_factory=list, max_length=20)


class Route(StrEnum):
    """Closed evidence-source choices, never provider prose."""

    DATA_ONLY = "data_only"
    KNOWLEDGE_ONLY = "knowledge_only"
    BOTH = "both"
    CLARIFY = "clarify"


class RouteDecision(Contract):
    """A bounded classification and independently scoped specialist tasks."""

    schema_version: Literal[1] = 1
    route: Route
    confidence: float = Field(ge=0, le=1)
    data_intent: str = Field(default="", max_length=32_000)
    knowledge_intent: str = Field(default="", max_length=32_000)
    metric_hints: list[Annotated[str, Field(min_length=1, max_length=64)]] = Field(
        default_factory=list, max_length=6
    )
    reasoning: str = Field(default="", max_length=2000)
    decided_by: Literal["prefilter", "llm"] = "llm"
    clarification_question: str = Field(default="", max_length=1000)

    @model_validator(mode="after")
    def required_intents(self) -> Self:
        """An analytical route cannot silently fall back to the raw question."""
        needs_data = self.route in {Route.DATA_ONLY, Route.BOTH}
        needs_knowledge = self.route in {Route.KNOWLEDGE_ONLY, Route.BOTH}
        if (needs_data and not self.data_intent.strip()) or (
            needs_knowledge and not self.knowledge_intent.strip()
        ):
            raise PydanticCustomError("route_intent", "Selected specialists require scoped intents")
        if any(not hint.strip() for hint in self.metric_hints):
            raise PydanticCustomError("route_hint", "Metric hints must be bounded nonempty keys")
        return self


class RoutingContext(Contract):
    """Already-loaded history, without SQL, evidence or saved metric overrides."""

    schema_version: Literal[1] = 1
    summary: str = Field(default="", max_length=32_000)
    recent_messages: list[HistoryMessage] = Field(default_factory=list, max_length=100)
    terminology: list[TerminologyContent] = Field(default_factory=list, max_length=5)
    format_preference: FormatPreferenceContent | None = None


class RouterInput(Contract):
    """Minimal input independent of the later parent state migration."""

    schema_version: Literal[1] = 1
    question: str = Field(max_length=32_000)
    routing_context: RoutingContext = Field(default_factory=RoutingContext)
