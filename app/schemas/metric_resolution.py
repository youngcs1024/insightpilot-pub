"""Versioned intent, preference and clarification boundaries for metric resolution."""

from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import AwareDatetime, Field, model_validator
from pydantic_core import PydanticCustomError

from app.schemas.mcp import Contract

MetricKey = Annotated[str, Field(min_length=1, max_length=64)]
SqlFragment = Annotated[str, Field(min_length=1, max_length=2000)]


class MetricPatch(Contract):
    """Absent fields preserve lower-priority values; fragments still require AST validation."""

    schema_version: Literal[1] = 1
    date_field: str | None = Field(default=None, min_length=1, max_length=127)
    add_filters: list[SqlFragment] = Field(default_factory=list, max_length=16)
    remove_filters: list[SqlFragment] = Field(default_factory=list, max_length=16)
    expression: SqlFragment | None = None


class MetricPatchEntry(Contract):
    """An explicit patch belongs to exactly one identified metric."""

    metric_key: MetricKey
    patch: MetricPatch


class MetricPatches(Contract):
    """No ambiguous duplicate patches cross the specialist boundary."""

    schema_version: Literal[1] = 1
    items: list[MetricPatchEntry] = Field(default_factory=list, max_length=6)

    @model_validator(mode="after")
    def unique_keys(self) -> Self:
        """Reject duplicate keys rather than silently choosing one."""
        keys = [item.metric_key for item in self.items]
        if len(keys) != len(set(keys)):
            raise PydanticCustomError("duplicate_metric_patch", "Duplicate metric patches")
        return self

    def for_metric(self, key: str) -> MetricPatch:
        """Return a detached patch, or an explicitly empty one."""
        return next(
            (item.patch.model_copy(deep=True) for item in self.items if item.metric_key == key),
            MetricPatch(),
        )


class SelectedMetricOverride(MetricPatchEntry):
    """An upstream-selected, user-owned memory; specialists never retrieve it."""

    id: UUID
    user_id: UUID
    created_at: AwareDatetime
    confidence: float = Field(ge=0, le=1)


class SelectedOverrides(Contract):
    """Phase 2 production inputs are empty; Phase 6 supplies gated memories."""

    schema_version: Literal[1] = 1
    items: list[SelectedMetricOverride] = Field(default_factory=list, max_length=5)

    @model_validator(mode="after")
    def unique_keys(self) -> Self:
        """One selected preference per metric is required."""
        keys = [item.metric_key for item in self.items]
        if len(keys) != len(set(keys)):
            raise PydanticCustomError("duplicate_metric_override", "Duplicate selected overrides")
        return self

    def for_metric(self, key: str) -> SelectedMetricOverride | None:
        """Consume the selected slot without storage access."""
        return next(
            (item.model_copy(deep=True) for item in self.items if item.metric_key == key), None
        )


class RegionScope(Contract):
    """Finalized business region IDs; an explicit all-regions scope has an empty list."""

    schema_version: Literal[1] = 1
    region_ids: list[Annotated[int, Field(gt=0, strict=True)]] = Field(
        default_factory=list, max_length=5
    )

    @model_validator(mode="after")
    def unique_ids(self) -> Self:
        """Duplicate region IDs are invalid upstream input."""
        if len(self.region_ids) != len(set(self.region_ids)):
            raise PydanticCustomError("duplicate_region", "Duplicate region IDs")
        return self


class RegionReference(Contract):
    """Explicit names are catalog lookup data, never model-selected region IDs."""

    schema_version: Literal[1] = 1
    names: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(
        default_factory=list, max_length=5
    )
    all_regions: bool = False


class MetricIntent(Contract):
    """Extract intent only; unfamiliar values survive until typed clarification."""

    schema_version: Literal[1] = 1
    metric_keys: list[MetricKey] = Field(default_factory=list, max_length=6)
    period_expression: str = Field(default="", max_length=500)
    grain: str = Field(default="total", max_length=64)
    dimensions: list[Annotated[str, Field(min_length=1, max_length=64)]] = Field(
        default_factory=list, max_length=6
    )
    explicit_patch: MetricPatches = Field(default_factory=MetricPatches)
    region_mentioned: bool = False
    region: RegionReference = Field(default_factory=RegionReference)


class ClarificationKind(StrEnum):
    """Business ambiguity is distinct from operational failure."""

    REFERENCE_UNRESOLVED = "reference_unresolved"
    METRIC_NOT_IDENTIFIED = "metric_not_identified"
    METRIC_NOT_FOUND = "metric_not_found"
    PERIOD_UNRESOLVED = "period_unresolved"
    UNSUPPORTED_GRAIN = "unsupported_grain"
    INVALID_EXPLICIT_PATCH = "invalid_explicit_patch"
    REGION_UNRESOLVED = "region_unresolved"


class MetricClarification(Contract):
    """Safe prose plus typed choices; never raw exception text."""

    schema_version: Literal[1] = 1
    kind: ClarificationKind
    message: str
    metric_key: str | None = None
    available_metrics: list[str] = Field(default_factory=list)
    supported_grains: list[str] = Field(default_factory=list)


class BindingSource(StrEnum):
    """Auditable precedence, including finalized region input."""

    COMPANY = "company"
    SAVED = "saved"
    EXPLICIT = "explicit"
    REGION = "region"


class BindingField(StrEnum):
    """Fields whose final ownership is retained in the binding."""

    DATE = "date_field"
    EXPRESSION = "expression"
    FILTER = "filter"


class BindingFieldSource(Contract):
    """Filter removals are retained as applied=False."""

    field: BindingField
    value: str
    source: BindingSource
    applied: bool = True
