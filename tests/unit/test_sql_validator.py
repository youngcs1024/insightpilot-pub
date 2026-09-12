"""Pure policy regression: complete AST checks and outer-only row limits."""

# ruff: noqa: S608 -- deliberate SQL strings exercise the validator without executing SQL.

import pytest
import sqlglot
from sqlglot import exp

from app.core.errors import McpPolicyRejected
from app.schemas.mcp import RESULT_CEILING, PolicyReason, ValidationOutcome, ValidationStatus
from mcp_server.policy.allowlist import ALLOWED_TABLES, MAX_QUERY_DEPTH
from mcp_server.policy.sql_validator import SQLValidator, clamp_result_cap


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "WITH x AS (SELECT 1) SELECT * FROM x",
        "SELECT 1 UNION SELECT 2",
        "SELECT 1 INTERSECT SELECT 2",
        "SELECT 1 EXCEPT SELECT 2",
        "(SELECT 1)",
    ],
)
def test_read_queries_pass(sql: str) -> None:
    assert SQLValidator().validate(sql).status is ValidationStatus.VALID


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO x VALUES (1)",
        "UPDATE x SET a=1",
        "DELETE FROM x",
        "DROP TABLE x",
        "CREATE TABLE x (a int)",
        "ALTER TABLE x ADD a int",
        "TRUNCATE x",
        "GRANT SELECT ON x TO y",
        "SELECT 1; DROP TABLE x",
        "WITH x AS (DELETE FROM t RETURNING *) SELECT * FROM x",
        "SELECT * INTO x FROM y",
        "SELECT * FROM x FOR UPDATE",
        "COPY x TO '/tmp/x'",
    ],
)
def test_writes_rejected(sql: str) -> None:
    assert SQLValidator().validate(sql).status is not ValidationStatus.VALID


@pytest.mark.parametrize(
    "name",
    [
        "pg_sleep",
        "pg_catalog.pg_sleep",
        'pg_catalog."pg_sleep"',
        "pg_read_file",
        "lo_import",
        "dblink",
        "dblink_exec",
        "set_config",
    ],
)
def test_blocked_functions_rejected(name: str) -> None:
    outcome = SQLValidator().validate(f"SELECT {name}('x')")
    assert outcome.reasons == [PolicyReason.BLOCKED_FUNCTION]


@pytest.mark.parametrize(
    "sql", ["SELECT 1 LIMIT $1", "SELECT 1 LIMIT -1", "SELECT 1 FETCH FIRST 2 ROWS WITH TIES"]
)
def test_unsupported_limit_rejected(sql: str) -> None:
    assert SQLValidator().validate(sql).status is not ValidationStatus.VALID


@pytest.mark.parametrize("sql", ["SELECT 1", "SELECT 1 LIMIT 99999", "SELECT 1 LIMIT ALL"])
def test_outer_limit_uses_sentinel(sql: str) -> None:
    outcome = SQLValidator().validate(sql, result_cap=10)
    assert outcome.status is ValidationStatus.VALID
    assert outcome.limit_applied
    assert sqlglot.parse_one(outcome.rewritten_sql).args["limit"].expression.this == "11"


@pytest.mark.parametrize("limit", [0, 1, 10])
def test_smaller_limit_preserved(limit: int) -> None:
    outcome = SQLValidator().validate(f"SELECT 1 LIMIT {limit}", result_cap=10)
    assert not outcome.limit_applied
    assert outcome.rewritten_sql.endswith(f"LIMIT {limit}")


def test_union_cte_aggregate_inputs_unchanged() -> None:
    sql = (
        "WITH x AS (SELECT * FROM biz.orders LIMIT 77) "
        "SELECT count(*) FROM x UNION SELECT count(*) FROM biz.orders"
    )
    outcome = SQLValidator().validate(sql, result_cap=10)
    ast = sqlglot.parse_one(outcome.rewritten_sql)
    assert isinstance(ast, exp.Union)
    assert sorted(int(item.expression.this) for item in ast.find_all(exp.Limit)) == [11, 77]
    assert all(item.args.get("limit") is None for item in (ast.left, ast.right))


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "10"])
def test_invalid_cap_rejected(value: object) -> None:
    with pytest.raises(McpPolicyRejected):
        clamp_result_cap(value)  # type: ignore[arg-type]


def test_server_ceiling() -> None:
    assert clamp_result_cap(9000) == RESULT_CEILING


def test_limit_all_normalization_preserves_literals() -> None:
    outcome = SQLValidator().validate("SELECT 'LIMIT ALL' AS text LIMIT ALL", result_cap=1)
    assert "'LIMIT ALL'" in outcome.rewritten_sql
    assert outcome.status is ValidationStatus.VALID


def test_limit_null_is_uncapped() -> None:
    outcome = SQLValidator().validate("SELECT 1 LIMIT NULL", result_cap=1)
    assert outcome.rewritten_sql.endswith("LIMIT 2")


@pytest.mark.parametrize("table", sorted(ALLOWED_TABLES))
@pytest.mark.parametrize("qualified", [False, True])
def test_business_table_allowed(table: str, qualified: bool) -> None:
    name = table if qualified else table.removeprefix("biz.")
    assert SQLValidator().validate(f"SELECT * FROM {name}").status is ValidationStatus.VALID


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM BIZ.ORDERS",
        'SELECT * FROM "biz"."orders"',
        "WITH ORDERS AS (SELECT 1) SELECT * FROM orders",
        'WITH "X" AS (SELECT 1) SELECT * FROM "X"',
        "SELECT * FROM generate_series(1, 3)",
        "SELECT * FROM pg_catalog.generate_series(1, 3)",
        "SELECT * FROM (SELECT * FROM biz.orders) x",
        "WITH orders AS (SELECT 1) SELECT * FROM biz.orders",
        "WITH x AS (SELECT 1), y AS (SELECT * FROM x) SELECT * FROM y",
        "WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM x WHERE n<3) SELECT * FROM x",
    ],
)
def test_cte_query_allowed(sql: str) -> None:
    assert SQLValidator().validate(sql).status is ValidationStatus.VALID


def test_cte_name_not_treated_as_table() -> None:
    outcome = SQLValidator().validate("WITH pg_user AS (SELECT 1) SELECT * FROM pg_user")
    assert outcome.status is ValidationStatus.VALID


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM pg_catalog.pg_user",
        "SELECT * FROM pg_user",
        "SELECT * FROM biz.step17_probe",
        "SELECT * FROM ops.seed_manifest",
        "SELECT * FROM otherdb.biz.orders",
        'SELECT * FROM "BIZ".orders',
        'SELECT * FROM biz."ORDERS"',
        'WITH "X" AS (SELECT 1) SELECT * FROM x',
        "WITH x AS (SELECT * FROM y), y AS (SELECT 1) SELECT * FROM x",
    ],
)
def test_non_allowlisted_table_rejected(sql: str) -> None:
    outcome = SQLValidator().validate(sql)
    assert outcome.status is ValidationStatus.UNSAFE
    assert outcome.reasons == [PolicyReason.TABLE_NOT_ALLOWED]
    assert outcome.rewritten_sql == ""
    assert not outcome.limit_applied


def test_information_schema_rejected() -> None:
    outcome = SQLValidator().validate("SELECT * FROM information_schema.tables")
    assert outcome.reasons == [PolicyReason.TABLE_NOT_ALLOWED]


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM (SELECT * FROM pg_catalog.pg_user) x",
        "SELECT 1 WHERE EXISTS (SELECT 1 FROM pg_catalog.pg_user)",
        "WITH x AS (SELECT * FROM pg_catalog.pg_user) SELECT 1",
        "SELECT 1 UNION SELECT 1 FROM pg_catalog.pg_user",
        "SELECT * FROM biz.orders o JOIN pg_catalog.pg_user u ON true",
    ],
)
def test_subquery_table_checked(sql: str) -> None:
    assert SQLValidator().validate(sql).reasons == [PolicyReason.TABLE_NOT_ALLOWED]


@pytest.mark.parametrize("table", ["pg_catalog.pg_user", "biz.pg_user", "other.biz.orders"])
def test_schema_qualified_cte_shadow_attack_rejected(table: str) -> None:
    sql = f"WITH pg_user AS (SELECT 1) SELECT * FROM {table}"
    assert SQLValidator().validate(sql).reasons == [PolicyReason.TABLE_NOT_ALLOWED]


def test_cte_name_outside_scope_not_exempt() -> None:
    sql = "SELECT * FROM (WITH pg_user AS (SELECT 1) SELECT * FROM pg_user) x JOIN pg_user ON true"
    assert SQLValidator().validate(sql).reasons == [PolicyReason.TABLE_NOT_ALLOWED]


def test_duplicate_alias_rejected_typed() -> None:
    sql = "SELECT * FROM biz.orders x JOIN biz.customers x ON true"
    assert SQLValidator().validate(sql).reasons == [PolicyReason.UNSUPPORTED_SCOPE]


@pytest.mark.parametrize("kind", ["derived", "scalar", "cte"])
@pytest.mark.parametrize("depth", [10, 11])
def test_deep_nesting_rejected(kind: str, depth: int) -> None:
    sql = "SELECT 1"
    for index in range(depth - 1):
        if kind == "derived":
            sql = f"SELECT * FROM ({sql}) x{index}"
        elif kind == "scalar":
            sql = f"SELECT ({sql})"
        else:
            sql = f"WITH x{index} AS ({sql}) SELECT * FROM x{index}"
    outcome = SQLValidator().validate(sql)
    if depth == MAX_QUERY_DEPTH:
        assert outcome.status is ValidationStatus.VALID
    else:
        assert outcome.reasons == [PolicyReason.NESTING_TOO_DEEP]


def test_set_arms_and_parentheses_do_not_increase_depth() -> None:
    sql = " UNION ALL ".join(["SELECT 1"] * 15)
    sql = "(" * 15 + sql + ")" * 15
    assert SQLValidator().validate(sql).status is ValidationStatus.VALID


@pytest.mark.parametrize(
    "name",
    [
        "current_setting",
        "set_config",
        "pg_terminate_backend",
        "pg_cancel_backend",
        "pg_read_file",
        "pg_read_binary_file",
        "lo_import",
        "lo_export",
        "dblink",
        "dblink_get_result",
        "dblink_get_connections",
        "dblink_is_busy",
        "dblink_cancel_query",
        "dblink_error_message",
        "dblink_open",
        "dblink_fetch",
        "dblink_close",
        "dblink_get_pkey",
        "dblink_build_sql_insert",
        "dblink_build_sql_delete",
        "dblink_build_sql_update",
        "postgres_fdw_handler",
        "postgres_fdw_validator",
        "postgres_fdw_get_connections",
        "postgres_fdw_disconnect",
        "postgres_fdw_disconnect_all",
    ],
)
@pytest.mark.parametrize("position", ["projection", "source"])
def test_all_blocked_functions_in_entire_ast(name: str, position: str) -> None:
    call = f"pg_catalog.\"{name}\"('x')"
    sql = f"SELECT {call}" if position == "projection" else f"SELECT * FROM {call} AS x"
    assert SQLValidator().validate(sql).status is not ValidationStatus.VALID


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM biz.orders INTO OUTFILE '/tmp/out'",
        "COPY biz.orders TO PROGRAM 'id'",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT * FROM dblink('remote', 'SELECT 1') AS t(n int)",
        "SELECT 1; DROP TABLE biz.orders",
        "SELECT * FROM biz.orders /*",
        "WITH x AS (SELECT current_setting('server_version')) SELECT * FROM x",
        "SELECT 1 WHERE EXISTS (SELECT pg_cancel_backend(1))",
        "SELECT 1 UNION SELECT set_config('search_path', 'public', false)",
    ],
)
def test_adversarial_subset_blocked(sql: str) -> None:
    outcome = SQLValidator().validate(sql)
    assert outcome.status is not ValidationStatus.VALID
    assert outcome.rewritten_sql == ""


def test_multiple_statements_rejected() -> None:
    outcome = SQLValidator().validate("SELECT 1; SELECT 2")
    assert outcome.reasons == [PolicyReason.MULTIPLE_STATEMENTS]


def test_reasons_are_enum_members_not_strings() -> None:
    outcome = SQLValidator().validate("SELECT * FROM pg_catalog.pg_user")
    assert all(isinstance(reason, PolicyReason) for reason in outcome.reasons)
    restored = ValidationOutcome.model_validate_json(outcome.model_dump_json())
    assert restored.reasons == [PolicyReason.TABLE_NOT_ALLOWED]
    assert isinstance(restored.reasons[0], PolicyReason)


@pytest.mark.parametrize("operator", ["UNION", "INTERSECT", "EXCEPT"])
def test_union_gets_limit(operator: str) -> None:
    outcome = SQLValidator().validate(f"SELECT 1 {operator} SELECT 2", result_cap=10)
    assert outcome.status is ValidationStatus.VALID
    assert outcome.rewritten_sql.endswith("LIMIT 11")
    assert outcome.limit_applied


def test_union_with_oversized_limit_is_clamped() -> None:
    outcome = SQLValidator().validate("SELECT 1 UNION SELECT 2 LIMIT 9999", result_cap=10)
    assert outcome.rewritten_sql.endswith("LIMIT 11")
    assert outcome.limit_applied


@pytest.mark.parametrize("sql", ["(SELECT 1 LIMIT 1)", "((SELECT 1 LIMIT 1))"])
def test_parenthesized_smaller_limit_preserved(sql: str) -> None:
    outcome = SQLValidator().validate(sql, result_cap=10)
    assert outcome.status is ValidationStatus.VALID
    assert not outcome.limit_applied
    ast = sqlglot.parse_one(outcome.rewritten_sql, dialect="postgres")
    assert [item.expression.this for item in ast.find_all(exp.Limit)] == ["1"]


@pytest.mark.parametrize("sql", ["(SELECT 1)", "(SELECT 1 UNION SELECT 2 LIMIT 99)"])
def test_parenthesized_result_capped(sql: str) -> None:
    outcome = SQLValidator().validate(sql, result_cap=10)
    ast = sqlglot.parse_one(outcome.rewritten_sql, dialect="postgres")
    assert outcome.limit_applied
    assert [item.expression.this for item in ast.find_all(exp.Limit)] == ["11"]


@pytest.mark.parametrize("sql", ["SELECT 1 LIMIT 1", "SELECT 1 LIMIT 10", "SELECT 1 LIMIT 99"])
def test_limit_applied_flag_is_accurate(sql: str) -> None:
    original = sqlglot.parse_one(sql, dialect="postgres")
    snapshot = original.copy()
    outcome = SQLValidator._limit(original, 10)
    assert original == snapshot
    assert outcome.limit_applied is sql.endswith("99")


def test_inner_aggregate_input_not_capped() -> None:
    sql = "SELECT count(*) FROM (SELECT * FROM biz.orders) x"
    outcome = SQLValidator().validate(sql, result_cap=1)
    ast = sqlglot.parse_one(outcome.rewritten_sql, dialect="postgres")
    assert [item.expression.this for item in ast.find_all(exp.Limit)] == ["2"]
    assert ast.find(exp.Subquery).this.args.get("limit") is None


def test_union_arms_unchanged() -> None:
    sql = "(SELECT * FROM biz.orders LIMIT 7) UNION ALL (SELECT * FROM biz.orders LIMIT 8)"
    original = sqlglot.parse_one(sql, dialect="postgres")
    outcome = SQLValidator().validate(sql, result_cap=1)
    result = sqlglot.parse_one(outcome.rewritten_sql, dialect="postgres")
    assert result.this == original.this
    assert result.expression == original.expression


def test_limit_all_handled() -> None:
    outcome = SQLValidator().validate("SELECT 1 UNION SELECT 2 LIMIT ALL", result_cap=1)
    assert outcome.limit_applied
    assert outcome.rewritten_sql.endswith("LIMIT 2")


@pytest.mark.parametrize("limit", ["$1", "(SELECT 1)", "1+1"])
def test_dynamic_limit_rejected_typed(limit: str) -> None:
    outcome = SQLValidator().validate(f"(SELECT 1 LIMIT {limit})")
    assert outcome.reasons == [PolicyReason.UNSUPPORTED_LIMIT]
