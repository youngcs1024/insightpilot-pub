"""Pure policy checks against the frozen, physically scripted schema artifact."""

from datetime import datetime

import pytest

from app.core.errors import McpPolicyRejected
from app.schemas.mcp import PolicyReason
from app.schemas.metric_resolution import MetricPatch
from mcp_server.config import MetricPolicySettings
from mcp_server.tools.get_schema import SchemaTool
from mcp_server.tools.resolve_metric import MetricResolver
from tests.metric_tool_support import metric_args
from tests.schema_tool_support import ARTIFACT, ScriptedSchemaReader


def resolver() -> MetricResolver:
    return MetricResolver(SchemaTool(ScriptedSchemaReader(), ARTIFACT), MetricPolicySettings())


@pytest.mark.parametrize(
    "key", ["gmv", "order_count", "aov", "active_customer", "refund_rate", "refund_count"]
)
async def test_all_published_queries_validate_without_semantic_rewrite(key: str) -> None:
    args = metric_args(key)
    result = await resolver().resolve(args)
    assert result.normalized_sql
    assert result.select_fragment
    assert "TIMESTAMPTZ" in result.where_fragment
    if key == "refund_rate":
        assert "WITH numerator" in result.normalized_sql
        assert "FULL OUTER JOIN" in result.normalized_sql
        assert result.warnings == ["complex_query_fragments_partial"]
    else:
        assert not result.warnings


async def test_unknown_column_reports_safe_name() -> None:
    args = metric_args()
    args.resolved_sql = args.resolved_sql.replace("o.gross_amount", "o.imagined_amount")
    args.expression = args.expression.replace("o.gross_amount", "o.imagined_amount")
    with pytest.raises(McpPolicyRejected) as caught:
        await resolver().resolve(args)
    assert caught.value.reasons == [PolicyReason.UNKNOWN_COLUMN]
    assert caught.value.column_name == "imagined_amount"


async def test_unknown_using_column_is_rejected_before_execution() -> None:
    args = metric_args()
    args.resolved_sql = args.resolved_sql.replace("USING (customer_id)", "USING (imagined_id)")
    with pytest.raises(McpPolicyRejected) as caught:
        await resolver().resolve(args)
    assert caught.value.reasons == [PolicyReason.UNKNOWN_COLUMN]
    assert caught.value.column_name == "imagined_id"


@pytest.mark.parametrize(
    ("key", "patch"),
    [
        ("gmv", MetricPatch(date_field="o.created_at", add_filters=["o.gross_amount > 100"])),
        ("refund_rate", MetricPatch(date_field="o.paid_at")),
    ],
)
async def test_structured_patches_preserve_validated_complete_sql(
    key: str, patch: MetricPatch
) -> None:
    args = metric_args(key, patch=patch)
    result = await resolver().resolve(args)
    assert result.normalized_sql
    assert "TIMESTAMPTZ" in result.where_fragment


async def test_exact_five_year_limit_allowed_and_excess_rejected() -> None:
    args = metric_args()
    args.period_end = args.period_start.replace(year=args.period_start.year + 5)
    args.resolved_sql = args.resolved_sql.replace("2026-09-01", "2031-08-01")
    await resolver().resolve(args)
    args.period_end = datetime.fromisoformat("2031-08-01T00:00:01+08:00")
    with pytest.raises(McpPolicyRejected) as caught:
        await resolver().resolve(args)
    assert caught.value.reasons == [PolicyReason.PERIOD_TOO_LONG]
