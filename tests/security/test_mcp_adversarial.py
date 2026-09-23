"""Step 5.8 unsafe SQL proof through the real MCP process and PostgreSQL boundary."""

import pytest

from app.clients.mcp_client import McpClient
from app.core.errors import McpPolicyRejected
from app.core.sql_policy.sql_validator import SQLValidator
from app.schemas.mcp import PolicyReason, ValidationStatus
from tests.integration.mcp_support import client, mcp_endpoint, query

pytestmark = pytest.mark.integration
__all__ = ["client", "mcp_endpoint"]

ADVERSARIAL_SQL = [
    pytest.param(
        "INSERT INTO biz.orders(order_id) VALUES (99)",
        PolicyReason.WRITE_OPERATION,
        id="insert",
    ),
    pytest.param(
        "UPDATE biz.orders SET status = 'paid'",
        PolicyReason.WRITE_OPERATION,
        id="update",
    ),
    pytest.param("DELETE FROM biz.orders", PolicyReason.WRITE_OPERATION, id="delete"),
    pytest.param("DROP TABLE biz.orders", PolicyReason.WRITE_OPERATION, id="drop"),
    pytest.param("CREATE TABLE biz.owned(i int)", PolicyReason.WRITE_OPERATION, id="create"),
    pytest.param(
        "ALTER TABLE biz.orders ADD COLUMN owned int",
        PolicyReason.WRITE_OPERATION,
        id="alter",
    ),
    pytest.param("TRUNCATE biz.orders", PolicyReason.WRITE_OPERATION, id="truncate"),
    pytest.param(
        "GRANT SELECT ON biz.orders TO public",
        PolicyReason.WRITE_OPERATION,
        id="grant",
    ),
    pytest.param(
        "SELECT 1; DROP TABLE biz.orders",
        PolicyReason.MULTIPLE_STATEMENTS,
        id="multiple_statements",
    ),
    pytest.param(
        "SELECT * FROM pg_catalog.pg_user",
        PolicyReason.TABLE_NOT_ALLOWED,
        id="catalog_table",
    ),
    pytest.param(
        "SELECT * FROM information_schema.columns",
        PolicyReason.TABLE_NOT_ALLOWED,
        id="information_schema",
    ),
    pytest.param(
        "SELECT * FROM biz.orders UNION SELECT * FROM pg_catalog.pg_shadow",
        PolicyReason.TABLE_NOT_ALLOWED,
        id="union_catalog_arm",
    ),
    pytest.param(
        "WITH x AS (DELETE FROM biz.orders RETURNING *) SELECT * FROM x",
        PolicyReason.WRITE_OPERATION,
        id="cte_delete",
    ),
    pytest.param(
        "SELECT pg_read_file('/etc/passwd')",
        PolicyReason.BLOCKED_FUNCTION,
        id="read_file",
    ),
    pytest.param("SELECT pg_sleep(60)", PolicyReason.BLOCKED_FUNCTION, id="sleep"),
    pytest.param(
        "SELECT set_config('search_path', 'public', false)",
        PolicyReason.BLOCKED_FUNCTION,
        id="set_config",
    ),
    pytest.param(
        "SELECT * FROM dblink('remote', 'SELECT 1') AS t(n int)",
        PolicyReason.BLOCKED_FUNCTION,
        id="dblink",
    ),
    pytest.param(
        "SELECT * FROM biz.orders FOR UPDATE",
        PolicyReason.WRITE_OPERATION,
        id="row_lock",
    ),
]


@pytest.mark.parametrize(("sql", "reason"), ADVERSARIAL_SQL)
async def test_adversarial_sql_blocked_over_mcp(
    client: McpClient, sql: str, reason: PolicyReason
) -> None:
    """All eighteen cases are actual rejections, not successful capped queries."""
    local = SQLValidator().validate(sql)
    assert local.status is not ValidationStatus.VALID
    assert local.reasons == [reason]
    with pytest.raises(McpPolicyRejected) as caught:
        await query(client, sql)
    assert caught.value.status is local.status
    assert caught.value.reasons == [reason]
    assert isinstance(caught.value.reasons[0], PolicyReason)
    assert client.breaker.failures == 0


async def test_qualified_catalog_cte_shadow_rejected_over_mcp(client: McpClient) -> None:
    sql = "WITH pg_user AS (SELECT 1) SELECT * FROM pg_catalog.pg_user"
    with pytest.raises(McpPolicyRejected) as caught:
        await query(client, sql)
    assert caught.value.reasons == [PolicyReason.TABLE_NOT_ALLOWED]


async def test_union_limit_enforced_over_mcp(client: McpClient) -> None:
    sql = "SELECT n FROM generate_series(1, 21) AS t(n) UNION ALL SELECT 22 ORDER BY n"
    result = await query(client, sql, 10)
    assert result.rows == [[number] for number in range(1, 11)]
    assert result.row_count == 10
    assert result.result_truncated
    assert result.limit_applied
    assert result.executed_sql.endswith("LIMIT 11")


async def test_aggregate_input_not_capped_over_mcp(client: McpClient) -> None:
    result = await query(client, "SELECT count(*) AS n FROM generate_series(1, 7000)", 10)
    assert result.rows == [[7000]]
    assert result.row_count == 1
    assert not result.result_truncated
    assert result.limit_applied
