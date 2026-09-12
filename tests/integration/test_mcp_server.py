"""Real HTTP MCP and PostgreSQL proofs, using an independent server environment."""

import httpx2
import psycopg
import pytest

from app.agents.summarize import SAMPLE_CAP, package_result
from app.clients.mcp_client import McpClient
from app.core.config_models import MCPSettings
from app.core.errors import McpPolicyRejected, McpResultError, SqlExecutionError, SqlTimeoutError
from app.schemas.mcp import RESULT_CEILING, PolicyReason, SqlErrorKind, ValidationStatus
from mcp_server.db import BusinessDatabase
from mcp_server.tools.execute_query import QueryExecutor
from tests.integration.mcp_support import business, business_tables, client, mcp_endpoint, query

pytestmark = pytest.mark.integration
__all__ = ["business", "business_tables", "client", "mcp_endpoint"]
HTTP_UNAUTHORIZED = 401
MCP_SCHEMA_VERSION = 2


async def test_select_returns_typed_columns(client: McpClient) -> None:
    result = await query(
        client, "SELECT 12.340::numeric AS amount, DATE '2026-08-01' AS day, 7::int AS n"
    )
    assert [column.name for column in result.columns] == ["amount", "day", "n"]
    assert [column.type for column in result.columns] == ["numeric", "date", "int4"]
    assert result.rows == [["12.340", "2026-08-01", 7]]


async def test_insert_rejected_by_policy(client: McpClient) -> None:
    with pytest.raises(McpPolicyRejected):
        await query(client, "INSERT INTO biz.regions(region_id) VALUES(99)")
    assert client.breaker.failures == 0


async def test_insert_rejected_by_role_when_policy_bypassed(business: BusinessDatabase) -> None:
    with pytest.raises(SqlExecutionError) as caught:
        await QueryExecutor(business).execute("INSERT INTO biz.regions(region_id) VALUES(99)")
    assert isinstance(caught.value.__cause__, psycopg.errors.ReadOnlySqlTransaction)


async def test_statement_timeout_enforced(client: McpClient) -> None:
    # Deliberately expensive aggregate; an outer LIMIT must not cap aggregate input.
    with pytest.raises(SqlTimeoutError):
        await query(
            client,
            "SELECT sum(a.i * b.i) FROM generate_series(1,100000) a(i) CROSS JOIN generate_series(1,100000) b(i)",
        )
    assert client.breaker.failures == 0
    assert (await query(client, "SELECT 1")).rows == [[1]]


@pytest.mark.parametrize(("size", "truncated"), [(9, False), (10, False), (11, True)])
async def test_row_cap_truncates_and_flags(client: McpClient, size: int, truncated: bool) -> None:
    result = await query(client, f"SELECT * FROM generate_series(1,{size}) n", 10)  # noqa: S608 -- integer test parameter.
    assert result.row_count == min(10, size)
    assert result.result_truncated is truncated


async def test_union_and_aggregate_preserve_inputs(client: McpClient) -> None:
    result = await query(
        client, "SELECT count(*) AS n FROM generate_series(1,7000) UNION ALL SELECT 42", 1
    )
    assert result.rows == [[7000]]
    assert result.result_truncated


async def test_empty_result_has_columns(client: McpClient) -> None:
    result = await query(client, "SELECT 1 AS n WHERE false")
    assert result.row_count == 0
    assert result.columns[0].name == "n"
    assert not result.result_truncated


async def test_unauthenticated_request_rejected(mcp_endpoint: MCPSettings) -> None:
    async with httpx2.AsyncClient(timeout=3, trust_env=False) as http:
        for headers in ({}, {"Authorization": "Bearer incorrect"}):
            response = await http.get(str(mcp_endpoint.base_url), headers=headers)
            assert response.status_code == HTTP_UNAUTHORIZED


async def test_transaction_settings_do_not_leak(business: BusinessDatabase) -> None:
    async with business.connection() as conn:
        before = await (await conn.execute("SHOW search_path")).fetchone()
    await QueryExecutor(business).execute("SELECT 1")
    async with business.connection() as conn:
        after = await (await conn.execute("SHOW search_path")).fetchone()
    assert before == after


async def test_server_ceiling_enforced(client: McpClient) -> None:
    result = await query(client, "SELECT * FROM generate_series(1,5001)", 9999)
    assert result.row_count == RESULT_CEILING
    assert result.result_truncated


async def test_unsupported_values_fail_typed(client: McpClient) -> None:
    with pytest.raises(McpResultError):
        await query(client, "SELECT ARRAY[1,2] AS unsupported")
    assert client.breaker.failures == 0


async def test_execution_metadata_matches_rewritten_sql(client: McpClient) -> None:
    result = await query(client, "SELECT 42 AS n", 10)
    assert result.schema_version == MCP_SCHEMA_VERSION
    assert result.executed_sql == "SELECT 42 AS n LIMIT 11"
    assert result.limit_applied
    assert result.execution_ms >= 0
    assert result.mcp_call_id


@pytest.mark.parametrize(
    ("sql", "kind"),
    [
        ("SELECT missing_column", SqlErrorKind.UNDEFINED_COLUMN),
        ("SELECT CASE WHEN true THEN 1 ELSE false END", SqlErrorKind.TYPE_MISMATCH),
    ],
)
async def test_safe_sql_error_category(client: McpClient, sql: str, kind: SqlErrorKind) -> None:
    with pytest.raises(SqlExecutionError) as caught:
        await query(client, sql)
    assert caught.value.kind == kind
    assert str(caught.value) == SqlExecutionError.user_message


async def test_undefined_table_category_when_policy_bypassed(business: BusinessDatabase) -> None:
    with pytest.raises(SqlExecutionError) as caught:
        await QueryExecutor(business).execute("SELECT * FROM biz.missing_table")
    assert caught.value.kind is SqlErrorKind.UNDEFINED_TABLE
    assert str(caught.value) == SqlExecutionError.user_message


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM biz.missing_table",
        "SELECT * FROM pg_catalog.pg_user",
        "SELECT * FROM information_schema.tables",
        "WITH pg_user AS (SELECT 1) SELECT * FROM pg_catalog.pg_user",
        "SELECT * FROM (WITH pg_user AS (SELECT 1) SELECT * FROM pg_user) x JOIN pg_user ON true",
    ],
)
async def test_table_policy_rejection_over_mcp(client: McpClient, sql: str) -> None:
    with pytest.raises(McpPolicyRejected) as caught:
        await query(client, sql)
    assert caught.value.status is ValidationStatus.UNSAFE
    assert caught.value.reasons == [PolicyReason.TABLE_NOT_ALLOWED]
    assert isinstance(caught.value.reasons[0], PolicyReason)
    assert client.breaker.failures == 0


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT count(*) AS n FROM biz.regions",
        "SELECT count(*) AS n FROM regions",
        "WITH x AS (SELECT * FROM biz.regions) SELECT count(*) AS n FROM x",
        "WITH regions AS (SELECT 1) SELECT count(*) AS n FROM biz.regions",
    ],
)
async def test_allowed_business_queries(
    client: McpClient, business: BusinessDatabase, sql: str
) -> None:
    expected = await QueryExecutor(business).execute("SELECT count(*) AS n FROM biz.regions")
    result = await query(client, sql)
    assert result.rows == expected.rows


async def test_fixed_search_path(business: BusinessDatabase) -> None:
    # Operator-side executor inspection; this function is forbidden over ordinary MCP.
    result = await QueryExecutor(business).execute("SELECT current_setting('search_path')")
    assert result.rows == [["biz, pg_catalog"]]


async def test_current_setting_rejected_over_mcp(client: McpClient) -> None:
    with pytest.raises(McpPolicyRejected) as caught:
        await query(client, "SELECT current_setting('search_path')")
    assert caught.value.reasons == [PolicyReason.BLOCKED_FUNCTION]


async def test_exact_cap_not_truncated(client: McpClient) -> None:
    result = await query(client, "SELECT * FROM generate_series(1, 10)", 10)
    assert result.row_count == 10  # noqa: PLR2004 -- exact boundary regression.
    assert not result.result_truncated


async def test_cap_plus_one_detected(client: McpClient) -> None:
    result = await query(client, "SELECT * FROM generate_series(1, 11)", 10)
    assert result.rows == [[n] for n in range(1, 11)]
    assert result.result_truncated
    summary = package_result(result, []).result_summary
    assert summary.columns[0].total == "55"
    assert summary.columns[0].maximum == "10"
    assert summary.statistics_scope == "returned_rows"
    assert summary.result_truncated


async def test_parenthesized_user_limit_preserved(client: McpClient) -> None:
    result = await query(client, "(SELECT * FROM generate_series(1, 20) LIMIT 2)", 10)
    assert result.rows == [[1], [2]]
    assert not result.limit_applied
    assert not result.result_truncated


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        (
            "SELECT n FROM generate_series(1,5001) n ORDER BY n",
            (5000, "12502500", "5000", True, True),
        ),
        (
            "SELECT n FROM generate_series(1,5000) n ORDER BY n",
            (5000, "12502500", "5000", False, True),
        ),
        (
            "SELECT n FROM generate_series(1,5001) n ORDER BY n DESC LIMIT 3",
            (3, "15000", "5001", False, False),
        ),
    ],
)
async def test_summary_excludes_sentinel(
    client: McpClient, sql: str, expected: tuple[int, str, str, bool, bool]
) -> None:
    row_count, total, maximum, truncated, limited = expected
    result = await query(client, sql, RESULT_CEILING)
    evidence = package_result(result, [])
    assert result.row_count == row_count
    assert evidence.result_summary.columns[0].total == total
    assert evidence.result_summary.columns[0].maximum == maximum
    assert evidence.result_summary.result_truncated is truncated
    assert evidence.result_summary.sample_truncated is (row_count > SAMPLE_CAP)
    assert evidence.limit_applied is limited
    assert evidence.sql == result.executed_sql
