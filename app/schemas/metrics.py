"""Typed global metric definitions and explicit, non-natural-language rendering inputs."""

from datetime import datetime
from enum import StrEnum
from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator
from pydantic_core import PydanticCustomError

from app.schemas.mcp import Contract


class Grain(StrEnum):
    """Closed grouping choices; SQL identifiers never come from free text."""

    TOTAL = "total"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    REGION = "region"
    CATEGORY = "category"


class MetricDateField(StrEnum):
    """The two catalog-owned time bases in v1."""

    PAID = "o.paid_at"
    REQUESTED = "r.requested_at"


class MetricExample(Contract):
    """A question, explanatory answer and executable PostgreSQL example."""

    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field(min_length=1, max_length=4000)
    sql: str = Field(min_length=1, max_length=32000)


class MetricDefinition(Contract):
    """A detached catalog version, distinct from its ORM storage record."""

    schema_version: Literal[1] = 1
    key: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    version: int = Field(gt=0, strict=True)
    display_name: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=8000)
    expression_template: str = Field(min_length=1, max_length=32000)
    base_tables: list[str] = Field(min_length=1, max_length=8)
    default_date_field: MetricDateField
    required_filters: list[str] = Field(min_length=1, max_length=16)
    supported_grains: list[Grain] = Field(min_length=1, max_length=6)
    examples: list[MetricExample] = Field(min_length=1, max_length=20)
    is_active: bool = Field(strict=True)

    @model_validator(mode="after")
    def unique_lists(self) -> Self:
        """Ambiguous duplicate declarations fail instead of silently collapsing."""
        for values in (self.base_tables, self.required_filters, self.supported_grains):
            if len(values) != len(set(values)):
                raise PydanticCustomError("metric_duplicate", "Duplicate metric declaration")
        return self


class MetricCatalog(Contract):
    """Migration authoring envelope; runtime reads definitions from PostgreSQL."""

    schema_version: Literal[1] = 1
    definitions: list[MetricDefinition] = Field(min_length=1)


class MetricRenderContext(Contract):
    """Explicit instants only; period interpretation belongs to Step 2.4."""

    period_start: AwareDatetime
    period_end: AwareDatetime
    grain: Grain = Grain.TOTAL

    @model_validator(mode="after")
    def ordered_period(self) -> Self:
        """Require a nonempty half-open interval of aware instants."""
        if self.period_start >= self.period_end:
            raise PydanticCustomError("metric_period", "Period start must precede end")
        return self


class MetricTemplateContext(Contract):
    """Internal template slots, constructed exclusively from validated fields."""

    period_start: str
    period_end: str
    grain: Grain
    date_field: MetricDateField
    bucket: str
    paid_bucket: str


def example_period() -> tuple[datetime, datetime]:
    """Fixed seed comparison period; never infer a year from the wall clock."""
    return (
        datetime.fromisoformat("2026-08-01T00:00:00+08:00"),
        datetime.fromisoformat("2026-09-01T00:00:00+08:00"),
    )
