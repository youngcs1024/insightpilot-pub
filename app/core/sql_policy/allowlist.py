"""Scope-aware physical relation policy for the eight business tables."""

from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from sqlglot.optimizer.scope import Scope, traverse_scope

from app.schemas.mcp import PolicyReason

ALLOWED_TABLES = frozenset(
    {
        "biz.regions",
        "biz.customers",
        "biz.products",
        "biz.orders",
        "biz.order_items",
        "biz.refunds",
        "biz.inventory",
        "biz.promotions",
    }
)
MAX_QUERY_DEPTH = 10


def query_depth(scope: Scope) -> int:
    """Count semantic queries, treating set arms and parentheses as transparent."""
    depth = 1
    while scope.parent is not None:
        if not scope.is_union and not isinstance(scope.parent.expression, exp.Subquery):
            depth += 1
        if depth > MAX_QUERY_DEPTH:
            return depth
        scope = scope.parent
    return depth


def table_reason(table: exp.Table, scope: Scope) -> PolicyReason | None:
    """Resolve a single source without exempting qualified or invisible CTE names."""
    if table.catalog:
        return PolicyReason.TABLE_NOT_ALLOWED
    if isinstance(table.this, exp.Func):
        # Function sources are checked by the validator's complete AST function pass.
        return None
    if not isinstance(table.this, exp.Identifier):
        return PolicyReason.UNSUPPORTED_SCOPE
    if table.db:
        return (
            None if f"{table.db}.{table.name}" in ALLOWED_TABLES else PolicyReason.TABLE_NOT_ALLOWED
        )
    source = scope.sources.get(table.alias_or_name)
    if isinstance(source, Scope) and table.name in scope.cte_sources:
        return None
    return None if f"biz.{table.name}" in ALLOWED_TABLES else PolicyReason.TABLE_NOT_ALLOWED


def scoped_tables_reason(query: exp.Expression, scopes: list[Scope]) -> PolicyReason | None:
    """Check every scoped table and reject tables omitted by scope traversal."""
    checked: set[int] = set()
    for scope in scopes:
        # Force SQLGlot's duplicate-source validation before trusting its resolution.
        _ = scope.selected_sources
        for table in scope.tables:
            reason = table_reason(table, scope)
            if reason is not None:
                return reason
            checked.add(id(table))
    if any(id(table) not in checked for table in query.find_all(exp.Table)):
        return PolicyReason.UNSUPPORTED_SCOPE
    return None


def validate_sources(query: exp.Expression) -> PolicyReason | None:
    """Fail closed when a physical source cannot be resolved in its own scope."""
    # PostgreSQL folds only unquoted identifiers. Preserve the caller's AST for output.
    try:
        normalized = normalize_identifiers(query.copy(), dialect="postgres")
        scopes = traverse_scope(normalized)
        if not scopes:
            return PolicyReason.UNSUPPORTED_SCOPE
        if any(query_depth(scope) > MAX_QUERY_DEPTH for scope in scopes):
            return PolicyReason.NESTING_TOO_DEEP
        return scoped_tables_reason(normalized, scopes)
    except RecursionError:
        return PolicyReason.NESTING_TOO_DEEP
    except SqlglotError:
        return PolicyReason.UNSUPPORTED_SCOPE
