"""Typed memory projections reserved for Phase 6; no storage or retrieval behavior."""

from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator
from pydantic_core import PydanticCustomError

from app.schemas.mcp import Contract
from app.schemas.metric_resolution import MetricKey, MetricPatch


class MemoryType(StrEnum):
    """The closed set of supported user preferences."""

    METRIC_OVERRIDE = "metric_override"
    REGION_FOCUS = "region_focus"
    TERMINOLOGY = "terminology"
    FORMAT_PREFERENCE = "format_preference"


class MetricOverrideContent(Contract):
    """A saved patch still requires current-intent eligibility checks."""

    metric_key: MetricKey
    patch: MetricPatch


class RegionFocusContent(Contract):
    """A preference, never an unconditional specialist filter."""

    region_ids: list[Annotated[int, Field(strict=True, gt=0)]] = Field(min_length=1, max_length=5)


class TerminologyContent(Contract):
    """Bounded user terminology, treated as untrusted reference data."""

    term: str = Field(min_length=1, max_length=50)
    means: str = Field(min_length=1, max_length=200)


class FormatPreferenceContent(Contract):
    """Presentation is applied by the formatter, never by a specialist."""

    prefer: Literal["table", "prose"]
    decimals: int = Field(ge=0, le=4)


class Memory(Contract):
    """A provenance-carrying Pydantic projection, not an ORM object or memory store."""

    schema_version: Literal[1] = 1
    id: UUID
    user_id: UUID
    source_turn_id: UUID
    memory_type: MemoryType
    content: MetricOverrideContent | RegionFocusContent | TerminologyContent | FormatPreferenceContent
    summary: str = Field(max_length=200)
    confidence: float = Field(ge=0, le=1)
    is_active: bool = True
    superseded_by: UUID | None = None
    superseded_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def content_matches_type(self) -> Self:
        """Reject a valid payload belonging to a different preference category."""
        expected = {
            MemoryType.METRIC_OVERRIDE: MetricOverrideContent,
            MemoryType.REGION_FOCUS: RegionFocusContent,
            MemoryType.TERMINOLOGY: TerminologyContent,
            MemoryType.FORMAT_PREFERENCE: FormatPreferenceContent,
        }
        if not isinstance(self.content, expected[self.memory_type]):
            raise PydanticCustomError("memory_content_type", "Memory content does not match its type")
        return self
