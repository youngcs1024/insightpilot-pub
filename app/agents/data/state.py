"""Serializable state owned by the data specialist."""

from typing import Literal, Self

from langchain_core.messages import AIMessage
from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from app.agents.contracts import DataEvidence, MetricExamplesSnapshot, ResolvedMetricBinding
from app.agents.failures import NodeFailure
from app.schemas.mcp import Contract, QueryResultPayload
from app.schemas.memory import TerminologyProjection
from app.schemas.metric_resolution import (
    MetricClarification,
    MetricKey,
    MetricPatches,
    RegionScope,
    SelectedOverrides,
)
from app.schemas.sanity import SanityCheckResult
from app.schemas.schema_catalog import TableName
from app.schemas.sql_correction import MAX_CORRECTIONS, CorrectionStatus, CorrectionStopReason
from app.services.periods import Period


class DataAgentInput(Contract):
    """Finalized turn inputs; no specialist independently retrieves preferences."""

    schema_version: Literal[1, 2] = 2
    question: str = Field(min_length=1, max_length=32_000)
    data_intent: str = Field(default="", max_length=32_000)
    metric_hints: list[MetricKey] = Field(default_factory=list, max_length=6)
    relevant_memories: list[TerminologyProjection] = Field(default_factory=list, max_length=5)
    selected_overrides: SelectedOverrides = Field(default_factory=SelectedOverrides)
    region_scope: RegionScope | None = None
    explicit_patch: MetricPatches = Field(default_factory=MetricPatches)
    reference_period: Period | None = None
    prior_sql: list[str] = Field(default_factory=list, max_length=3)


class DataAgentOutput(Contract):
    """Only a packaged result crosses back into the parent graph."""

    schema_version: Literal[1, 2] = 2
    evidence: DataEvidence | None = None
    failure: NodeFailure | None = None
    assumptions: list[str] = Field(default_factory=list)
    clarification: MetricClarification | None = None
    correction_stop_reason: CorrectionStopReason | None = None

    @model_validator(mode="after")
    def terminal_outcome(self) -> Self:
        """Require exactly one terminal artifact at the specialist boundary."""
        if (
            sum(value is not None for value in (self.evidence, self.failure, self.clarification))
            != 1
        ):
            raise PydanticCustomError("data_outcome", "Expected exactly one terminal outcome")
        return self


class DataAgentState(DataAgentInput):
    """Schema and metric outputs each have a single node owner."""

    evidence: DataEvidence | None = None
    failure: NodeFailure | None = None
    schema_block: str = ""
    schema_tables: list[TableName] = Field(default_factory=list, max_length=8)
    metric_bindings: list[ResolvedMetricBinding] = Field(default_factory=list, max_length=6)
    assumptions: list[str] = Field(default_factory=list)
    clarification: MetricClarification | None = None
    metric_examples: list[MetricExamplesSnapshot] = Field(default_factory=list, max_length=6)
    generated_sql: str = ""
    tables_used: list[str] = Field(default_factory=list)
    messages: list[AIMessage] = Field(default_factory=list)
    query_result: QueryResultPayload | None = None
    sanity_check_result: SanityCheckResult = Field(default_factory=SanityCheckResult)
    failures: list[NodeFailure] = Field(default_factory=list)
    correction_count: int = Field(default=0, ge=0, le=MAX_CORRECTIONS)
    correction_status: CorrectionStatus = CorrectionStatus.IDLE
    correction_stop_reason: CorrectionStopReason | None = None
