"""Typed memory content, write inputs and durable read projections."""

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


class TerminologyProjection(TerminologyContent):
    """Finalized terminology only; no user identity or unrelated preference payload."""

    schema_version: Literal[1] = 1
    type: Literal["terminology"] = "terminology"


class FormatPreferenceContent(Contract):
    """Presentation is applied by the formatter, never by a specialist."""

    prefer: Literal["table", "prose"]
    decimals: int = Field(ge=0, le=4)


class MemoryPayload(Contract):
    """Shared typed content validation for extraction and repository writes."""

    schema_version: Literal[1] = 1
    memory_type: MemoryType
    content: (
        MetricOverrideContent | RegionFocusContent | TerminologyContent | FormatPreferenceContent
    )
    summary: str = Field(max_length=200)
    confidence: float = Field(ge=0, le=1)

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
            raise PydanticCustomError(
                "memory_content_type", "Memory content does not match its type"
            )
        return self


class MemoryCreate(MemoryPayload):
    """Validated new content; identity and lifecycle fields belong to the repository."""

    source_turn_id: UUID


class Memory(MemoryCreate):
    """A provenance-carrying projection compatible with existing graph checkpoints."""

    id: UUID
    user_id: UUID
    is_active: bool = True
    superseded_by: UUID | None = None
    superseded_at: AwareDatetime | None = None


class StoredMemory(Memory):
    """Repository result with database-generated audit timestamps."""

    created_at: AwareDatetime
    updated_at: AwareDatetime
