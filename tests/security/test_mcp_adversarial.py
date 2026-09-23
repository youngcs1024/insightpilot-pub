"""Authored SQL probes through real MCP, with independent database controls where applicable."""

import pytest

from app.clients.mcp_client import McpClient
from app.core.errors import McpPolicyRejected, SqlExecutionError
from app.core.sql_policy.sql_validator import SQLValidator
from app.schemas.mcp import PolicyReason, ValidationStatus
from evals.harness.adversarial import AdversarialCase, load_adversaries
from mcp_server.db import BusinessDatabase
from mcp_server.tools.execute_query import QueryExecutor
from tests.integration.mcp_support import business, client, mcp_endpoint, query

pytestmark = pytest.mark.integration
__all__ = ["business", "client", "mcp_endpoint"]

CASES = load_adversaries()
REJECTIONS = [case for case in CASES if case.expected_reasons]
REWRITES = [case for case in CASES if case.expected_limit is not None]


@pytest.mark.parametrize("case", REJECTIONS, ids=lambda case: case.id)
async def test_adversarial_sql_blocked_over_mcp(client: McpClient, case: AdversarialCase) -> None:
    """Exactly the authored typed rejection occurs without opening the breaker."""
    local = SQLValidator().validate(case.sql, result_cap=case.max_rows)
    assert local.status is not ValidationStatus.VALID
    assert local.reasons == case.expected_reasons
    with pytest.raises(McpPolicyRejected) as caught:
        await query(client, case.sql, case.max_rows)
    assert caught.value.status is local.status
    assert caught.value.reasons == case.expected_reasons
    assert client.breaker.failures == 0


@pytest.mark.parametrize(
    "case", [case for case in REJECTIONS if case.database_read_only], ids=lambda case: case.id
)
async def test_database_rejects_write_after_validator_bypass(
    business: BusinessDatabase, case: AdversarialCase
) -> None:
    """The independent role prevents writes even when no AST policy is called."""
    executor = QueryExecutor(business)
    with pytest.raises(SqlExecutionError):
        await executor.execute(case.sql)


@pytest.mark.parametrize("case", REWRITES, ids=lambda case: case.id)
async def test_safe_rewrites_cap_real_mcp_result(client: McpClient, case: AdversarialCase) -> None:
    result = await query(client, case.sql, case.max_rows)
    assert result.limit_applied
    assert result.row_count <= case.max_rows
    assert result.executed_sql.endswith(f"LIMIT {case.expected_limit}")


async def test_qualified_catalog_cte_shadow_rejected_over_mcp(client: McpClient) -> None:
    sql = "WITH pg_user AS (SELECT 1) SELECT * FROM pg_catalog.pg_user"
    with pytest.raises(McpPolicyRejected) as caught:
        await query(client, sql)
    assert caught.value.reasons == [PolicyReason.TABLE_NOT_ALLOWED]


async def test_union_limit_enforced_over_mcp(client: McpClient) -> None:
    sql = "SELECT n FROM generate_series(1, 21) AS t(n) UNION ALL SELECT 22 ORDER BY n"
    cap = 10
    result = await query(client, sql, cap)
    assert result.rows == [[number] for number in range(1, cap + 1)]
    assert result.row_count == cap
    assert result.result_truncated
    assert result.limit_applied
    assert result.executed_sql.endswith(f"LIMIT {cap + 1}")


async def test_aggregate_input_not_capped_over_mcp(client: McpClient) -> None:
    result = await query(client, "SELECT count(*) AS n FROM generate_series(1, 7000)", 10)
    assert result.rows == [[7000]]
    assert result.row_count == 1
    assert not result.result_truncated
    assert result.limit_applied
