"""Typed boundary contract for validating an application-resolved metric query."""

from typing import Literal

from pydantic import AwareDatetime, Field

from app.schemas.mcp import Contract
from app.schemas.metrics import Grain
from app.schemas.schema_tools import RequestedTable


class ResolveMetricArgs(Contract):
    """The application owns meaning; the server checks the complete rendered query."""

    metric_key: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    expression: str = Field(min_length=1, max_length=32000)
    resolved_sql: str = Field(min_length=1, max_length=32000)
    base_tables: list[RequestedTable] = Field(min_length=1, max_length=8)
    date_field: str = Field(min_length=1, max_length=127)
    period_start: AwareDatetime
    period_end: AwareDatetime
    filters: list[str] = Field(max_length=16)
    grain: Grain


class MetricFragment(Contract):
    """Display fragments and the authoritative normalized complete query."""

    schema_version: Literal[1] = 1
    select_fragment: str
    from_fragment: str
    where_fragment: str
    group_by_fragment: str
    normalized_sql: str = Field(min_length=1, max_length=32000)
    normalized: bool
    warnings: list[str] = Field(default_factory=list)
