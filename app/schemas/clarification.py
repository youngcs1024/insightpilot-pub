"""Bounded clarification policy inputs, separate from analytical evidence."""

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field

from app.schemas.corpus import DocumentType
from app.schemas.mcp import Contract


class ClarificationCategory(StrEnum):
    """The three product-level reasons to request a new user turn."""

    AMBIGUOUS_REFERENCE = "ambiguous_reference"
    AMBIGUOUS_SCOPE = "ambiguous_scope"
    OUT_OF_SCOPE = "out_of_scope"


class MissingDimension(StrEnum):
    """Missing or unresolved scope, never inferred from error prose."""

    REFERENCE = "reference"
    METRIC = "metric"
    PERIOD = "period"
    REGION = "region"
    GRAIN = "grain"
    DEFINITION = "definition"


class ClarificationIntent(Contract):
    """Router or specialist supplied scope; suggestions are not authorization."""

    schema_version: Literal[1] = 1
    category: ClarificationCategory
    missing_dimensions: list[MissingDimension] = Field(default_factory=list, max_length=6)
    metric_keys: list[Annotated[str, Field(min_length=1, max_length=64)]] = Field(
        default_factory=list, max_length=6
    )
    period_expression: str = Field(default="", max_length=500)
    subject: str = Field(default="", max_length=1000)


class ClarificationHistory(Contract):
    """Counts survive text trimming; topics are owned, real prior user messages."""

    schema_version: Literal[1] = 1
    consecutive: int = Field(default=0, ge=0, le=2)
    previous_suggestion: str = Field(default="", max_length=2000)
    previous_metric_keys: list[Annotated[str, Field(min_length=1, max_length=64)]] = Field(
        default_factory=list, max_length=64
    )
    recent_topics: list[Annotated[str, Field(min_length=1, max_length=240)]] = Field(
        default_factory=list, max_length=3
    )


class AvailableMetric(Contract):
    """Only the published name and key are needed for a capability menu."""

    key: str = Field(min_length=1, max_length=64)
    display_name: str = Field(min_length=1, max_length=100)


class ClarificationCapabilities(Contract):
    """Application database projection; no query execution or retrieval results."""

    schema_version: Literal[1] = 1
    metrics: list[AvailableMetric] = Field(default_factory=list, max_length=64)
    document_categories: list[DocumentType] = Field(default_factory=list, max_length=6)
