"""Single-statement PostgreSQL policy with semantics-preserving outer limits."""

from itertools import pairwise

import sqlglot
from sqlglot import exp
from sqlglot.dialects.postgres import Postgres
from sqlglot.errors import SqlglotError
from sqlglot.tokenizer_core import TokenType

from app.core.errors import McpPolicyRejected
from app.core.sql_policy.allowlist import validate_sources
from app.schemas.mcp import (
    RESULT_CEILING,
    PolicyReason,
    ValidationOutcome,
    ValidationStatus,
)

BLOCKED_FUNCTIONS = frozenset(
    {
        "pg_sleep",
        "pg_sleep_for",
        "pg_sleep_until",
        "pg_read_file",
        "pg_read_binary_file",
        "pg_ls_dir",
        "pg_stat_file",
        "lo_import",
        "lo_export",
        "dblink",
        "dblink_exec",
        "dblink_connect",
        "dblink_connect_u",
        "dblink_disconnect",
        "dblink_send_query",
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
        "current_setting",
        "pg_advisory_lock",
        "pg_advisory_xact_lock",
        "pg_try_advisory_lock",
        "set_config",
        "nextval",
        "setval",
        "pg_terminate_backend",
        "pg_cancel_backend",
        "sleep",
        "benchmark",
        "load_file",
        "into_outfile",
        "into_dumpfile",
        "xp_cmdshell",
        "xp_fileexist",
        "xp_dirtree",
        "xp_regread",
        "xp_regwrite",
        "sp_oacreate",
        "sp_oamethod",
        "openrowset",
        "opendatasource",
        "bulk",
        "waitfor",
        "session_user",
        "reflect",
        "java_method",
        "exec",
        "execute",
        "system",
        "shell",
    }
)
QUERY_TYPES = (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.Subquery)
WRITE_TYPES = (
    exp.DML,
    exp.DDL,
    exp.Into,
    exp.Lock,
    exp.Copy,
    exp.Command,
    exp.Grant,
    exp.Revoke,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
)


def clamp_result_cap(value: int) -> int:
    """Reject booleans/coercion and cap positive input at the server ceiling."""
    if type(value) is not int or value <= 0:
        raise McpPolicyRejected(ValidationStatus.INVALID, [PolicyReason.INVALID_ARGUMENTS])
    return min(value, RESULT_CEILING)


def rejected(reason: PolicyReason, *, invalid: bool = False) -> ValidationOutcome:
    """Build a typed policy outcome without retaining raw SQL diagnostics."""
    return ValidationOutcome(
        status=ValidationStatus.INVALID if invalid else ValidationStatus.UNSAFE,
        reasons=[reason],
    )


def function_name(node: exp.Func) -> str:
    """Handle built-in and anonymous/qualified PostgreSQL functions uniformly."""
    return (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()


def normalize_limit_all(sql: str) -> str:
    """Work around the locked compiled parser: PostgreSQL LIMIT NULL equals ALL.

    Token offsets keep strings, comments and identifiers untouched, including nested SQL.
    """
    tokens = Postgres().tokenize(sql)
    positions = [
        token
        for previous, token in pairwise(tokens)
        if previous.token_type is TokenType.LIMIT and token.token_type is TokenType.ALL
    ]
    for token in reversed(positions):
        sql = sql[: token.start] + "NULL" + sql[token.end + 1 :]
    return sql


class SQLValidator:
    """Validate the complete AST; never add a limit to an input relation."""

    def validate(self, sql: str, *, result_cap: int = 1000) -> ValidationOutcome:
        """Return a safe outer cap+1 query or a typed rejection."""
        cap = clamp_result_cap(result_cap)
        try:
            statements = sqlglot.parse(normalize_limit_all(sql), dialect="postgres")
        except RecursionError:
            return rejected(PolicyReason.NESTING_TOO_DEEP)
        except SqlglotError:
            return rejected(PolicyReason.INVALID_SQL, invalid=True)
        queries = [item for item in statements if item is not None]
        if len(queries) != 1:
            return rejected(PolicyReason.MULTIPLE_STATEMENTS, invalid=True)
        return self._validate_query(queries[0], cap)

    def _validate_query(self, query: exp.Expr, cap: int) -> ValidationOutcome:
        if not isinstance(query, QUERY_TYPES) or any(
            isinstance(node, WRITE_TYPES) for node in query.walk()
        ):
            return rejected(PolicyReason.WRITE_OPERATION)
        if any(function_name(node) in BLOCKED_FUNCTIONS for node in query.find_all(exp.Func)):
            return rejected(PolicyReason.BLOCKED_FUNCTION)
        reason = validate_sources(query)
        if reason is not None:
            return rejected(reason)
        return self._limit(query, cap)

    @staticmethod
    def _limit(query: exp.Expression, cap: int) -> ValidationOutcome:
        query = query.copy()
        outer = query
        # A bare parenthesis wrapper is not a separate result operation. Respect the
        # enclosed result's user LIMIT instead of adding an ineffective wrapper cap.
        while isinstance(outer, exp.Subquery) and not any(
            outer.args.get(key) is not None for key in ("limit", "offset", "order")
        ):
            outer = outer.this
        limit = outer.args.get("limit")
        expression = limit.expression if isinstance(limit, exp.Limit) else None
        if limit is not None:
            if not isinstance(limit, exp.Limit) or limit.args.get("limit_options"):
                return rejected(PolicyReason.UNSUPPORTED_LIMIT)
            if isinstance(expression, exp.Literal) and expression.is_int:
                value = int(expression.this)
                if value <= cap:
                    return ValidationOutcome(
                        status=ValidationStatus.VALID, rewritten_sql=query.sql(dialect="postgres")
                    )
            elif not isinstance(expression, exp.Null):
                return rejected(PolicyReason.UNSUPPORTED_LIMIT)
        outer.set("limit", exp.Limit(expression=exp.Literal.number(cap + 1)))
        return ValidationOutcome(
            status=ValidationStatus.VALID,
            rewritten_sql=query.sql(dialect="postgres"),
            limit_applied=True,
        )
