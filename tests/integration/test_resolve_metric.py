"""Step 5.2 acceptance through the authenticated independent MCP process."""

import time

import pytest

from app.clients.mcp_client import McpClient
from app.core.deadline import Deadline
from app.core.errors import McpPolicyRejected
from app.schemas.mcp import PolicyReason
from tests.integration.catalog_support import catalog_migrated, client
from tests.integration.mcp_support import mcp_endpoint as server_endpoint
from tests.metric_tool_support import metric_args
from tests.seed_support import seed_database_stack

pytestmark = pytest.mark.integration
__all__ = ["catalog_migrated", "client", "server_endpoint"]
database_stack = seed_database_stack


def budget() -> Deadline:
    return Deadline(time.monotonic() + 20)


async def test_valid_binding_returns_fragments(client: McpClient) -> None:
    args = metric_args()
    result = await client.resolve_metric(args, deadline=budget())
    assert "SUM" in result.select_fragment
    assert "FROM biz.orders" in result.from_fragment
    assert "o.status <> 'cancelled'" in result.where_fragment
    assert result.normalized_sql


async def test_unknown_column_rejected(client: McpClient) -> None:
    args = metric_args()
    args.resolved_sql = args.resolved_sql.replace("o.gross_amount", "o.imagined_amount")
    args.expression = args.expression.replace("o.gross_amount", "o.imagined_amount")
    with pytest.raises(McpPolicyRejected) as caught:
        await client.resolve_metric(args, deadline=budget())
    assert caught.value.reasons == [PolicyReason.UNKNOWN_COLUMN]
    assert caught.value.column_name == "imagined_amount"
    assert client.breaker.failures == 0


async def test_non_allowlisted_table_rejected(client: McpClient) -> None:
    args = metric_args()
    args.base_tables.append("pg_catalog.pg_user")
    with pytest.raises(McpPolicyRejected) as caught:
        await client.resolve_metric(args, deadline=budget())
    assert caught.value.reasons == [PolicyReason.TABLE_NOT_ALLOWED]


async def test_inverted_period_rejected(client: McpClient) -> None:
    args = metric_args()
    args.period_start, args.period_end = args.period_end, args.period_start
    with pytest.raises(McpPolicyRejected) as caught:
        await client.resolve_metric(args, deadline=budget())
    assert caught.value.reasons == [PolicyReason.INVALID_PERIOD]


async def test_excessive_period_rejected(client: McpClient) -> None:
    args = metric_args()
    args.period_end = args.period_end.replace(year=args.period_end.year + 5)
    with pytest.raises(McpPolicyRejected) as caught:
        await client.resolve_metric(args, deadline=budget())
    assert caught.value.reasons == [PolicyReason.PERIOD_TOO_LONG]


async def test_subquery_against_pg_catalog_rejected(client: McpClient) -> None:
    args = metric_args()
    args.resolved_sql = args.resolved_sql.replace(
        "o.status <> 'cancelled'",
        "o.status <> 'cancelled' AND o.order_id IN (SELECT oid FROM pg_catalog.pg_class)",
    )
    with pytest.raises(McpPolicyRejected) as caught:
        await client.resolve_metric(args, deadline=budget())
    assert caught.value.reasons == [PolicyReason.TABLE_NOT_ALLOWED]


@pytest.mark.parametrize("key", ["gmv", "refund_rate"])
async def test_fragments_use_timestamptz_literals(client: McpClient, key: str) -> None:
    result = await client.resolve_metric(metric_args(key), deadline=budget())
    assert "TIMESTAMPTZ" in result.where_fragment
    assert "EXTRACT" not in result.where_fragment


async def test_refund_rate_preserves_independent_aggregates(client: McpClient) -> None:
    result = await client.resolve_metric(metric_args("refund_rate"), deadline=budget())
    assert "WITH numerator" in result.normalized_sql
    assert "denominator" in result.normalized_sql
    assert "FULL OUTER JOIN" in result.normalized_sql
    assert result.warnings == ["complex_query_fragments_partial"]
