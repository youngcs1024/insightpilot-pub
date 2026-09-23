"""Published bindings adapted to the MCP validation contract for tests."""

import sqlglot

from app.schemas.metric_tools import ResolveMetricArgs
from app.schemas.metric_resolution import MetricPatch
from app.services.metric_binding import build_binding
from tests.metric_resolution_support import definition, request, schema


def metric_args(key: str = "gmv", *, patch: MetricPatch | None = None) -> ResolveMetricArgs:
    """Render one real published metric, including its complete query and provenance."""
    binding = build_binding(request(key, patch=patch), schema()).binding
    query = sqlglot.parse_one(binding.resolved_expression, read="postgres")
    return ResolveMetricArgs(
        metric_key=key,
        expression=query.expressions[1].this.sql(dialect="postgres"),
        resolved_sql=binding.resolved_expression,
        base_tables=definition(key).base_tables,
        date_field=binding.date_field,
        period_start=binding.period_start,
        period_end=binding.period_end,
        filters=binding.filters_applied,
        grain=binding.grain,
    )
