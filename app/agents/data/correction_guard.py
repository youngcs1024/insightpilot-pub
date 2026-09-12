"""Conservative AST proof against immutable resolved query snapshots.

This deliberately is not a general SQL equivalence solver or the MCP policy.
Unsupported rewrites fail closed. In particular, combining multiple metrics and
changing scope/table aliases requires a future proof strategy, not an LLM vote.
"""

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from sqlglot.optimizer.scope import traverse_scope

from app.agents.contracts import ResolvedMetricBinding
from app.schemas.schema_catalog import BUSINESS_TABLES
from app.services.metric_patch_sql import predicates


def _query(sql: str) -> exp.Select | None:
    parsed = sqlglot.parse(sql, read="postgres")
    if len(parsed) != 1 or not isinstance(parsed[0], exp.Select):
        return None
    query = normalize_identifiers(parsed[0], dialect="postgres")
    # Removing comments affects proof identity only, never executable SQL.
    for node in query.walk():
        node.comments = None
    return query


def same_statement(previous_sql: str, candidate_sql: str) -> bool:
    """Formatting, fencing (removed by caller) and comments are not an attempt."""
    try:
        previous = _query(previous_sql)
        candidate = _query(candidate_sql)
        return previous is not None and candidate is not None and previous == candidate
    except (SqlglotError, RecursionError):
        return False


def _qualify_business_tables(query: exp.Select) -> None:
    # An omitted business schema is a provable name-resolution repair. Resolve
    # physical sources per scope so a CTE named "orders" is never mistaken for one.
    for scope in traverse_scope(query):
        for _node, source in scope.selected_sources.values():
            if (
                isinstance(source, exp.Table)
                and not source.db
                and not source.catalog
                and f"biz.{source.name}" in BUSINESS_TABLES
            ):
                source.set("db", exp.to_identifier("biz"))


def _where(query: exp.Select) -> set[str]:
    where = query.args.get("where")
    return {
        part.sql(dialect="postgres")
        for part in predicates(where.this if isinstance(where, exp.Where) else None)
    }


def _shape(query: exp.Select) -> str:
    result = query.copy()
    result.set("where", None)
    # Only outer output names may change. Nested aliases define column lineage
    # and must remain exact, as must ORDER BY/GROUP BY alias references.
    result.set(
        "expressions",
        [
            item.this.copy() if isinstance(item, exp.Alias) else item.copy()
            for item in result.expressions
        ],
    )
    return result.sql(dialect="postgres")


def preserves_semantics(
    previous_sql: str, candidate_sql: str, bindings: list[ResolvedMetricBinding]
) -> bool:
    """Prove a narrow correction without losing any previous WHERE condition.

    A resolved expression is a full query, including joins, grain, period, region
    and per-scope filters. Exact tree comparison preserves nested scope semantics;
    only the outer WHERE permits extra conjuncts, which must survive unchanged.
    OR/NOT are atomic predicates, never flattened into allegedly required terms.
    Unparseable original SQL and multi-metric compositions are unproven, not safe.
    """
    if len(bindings) != 1:
        return False
    try:
        previous = _query(previous_sql)
        candidate = _query(candidate_sql)
        required = _query(bindings[0].resolved_expression)
        if previous is None or candidate is None or required is None:
            return False
        for query in (previous, candidate, required):
            _qualify_business_tables(query)
        return (
            _shape(previous) == _shape(candidate) == _shape(required)
            and _where(previous) == _where(candidate)
            and _where(required) <= _where(candidate)
        )
    except (SqlglotError, RecursionError):
        return False
