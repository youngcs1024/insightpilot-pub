"""Pure, scope-aware edits of published queries; this is not the MCP SQL policy."""

from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from sqlglot.optimizer.scope import Scope, traverse_scope

from app.core.errors import InvalidMetricPatchError, MetricCatalogError
from app.schemas.metric_resolution import MetricPatch
from app.schemas.metrics import MetricDefinition
from app.schemas.schema_catalog import SchemaCatalog, SemanticType
from app.services.periods import Period

# Closed syntax, not a blacklist: no queries, comments, parameters, casts, window
# functions, qualified functions or arbitrary PostgreSQL function invocations.
_SCALAR_NODES = frozenset(
    {
        exp.Column,
        exp.Identifier,
        exp.Literal,
        exp.Null,
        exp.Boolean,
        exp.Paren,
        exp.Add,
        exp.Sub,
        exp.Mul,
        exp.Div,
        exp.Neg,
        exp.Coalesce,
        exp.Nullif,
    }
)
_BOOLEAN_NODES = _SCALAR_NODES | frozenset(
    {
        exp.EQ,
        exp.NEQ,
        exp.GT,
        exp.GTE,
        exp.LT,
        exp.LTE,
        exp.And,
        exp.Or,
        exp.Not,
        exp.Is,
        exp.In,
        exp.Between,
    }
)
_AGGREGATE_NODES = _SCALAR_NODES | frozenset(
    {exp.Sum, exp.Count, exp.Avg, exp.Min, exp.Max, exp.Distinct, exp.Star}
)
_BOOLEAN_ROOTS = (
    exp.EQ,
    exp.NEQ,
    exp.GT,
    exp.GTE,
    exp.LT,
    exp.LTE,
    exp.And,
    exp.Or,
    exp.Not,
    exp.Is,
    exp.In,
    exp.Between,
    exp.Boolean,
)
MAX_FRAGMENT_NODES = 200


def parse_fragment(value: str, *, predicate: bool = False) -> exp.Expr:
    """Require exactly one bounded, side-effect-free scalar or boolean expression."""
    try:
        parsed = sqlglot.parse(value, read="postgres")
        if len(parsed) != 1 or parsed[0] is None:
            raise InvalidMetricPatchError()
        expression = normalize_identifiers(parsed[0], dialect="postgres")
    except (SqlglotError, RecursionError) as exc:
        raise InvalidMetricPatchError() from exc
    nodes = list(expression.walk())
    allowed = _BOOLEAN_NODES if predicate else _AGGREGATE_NODES
    if len(nodes) > MAX_FRAGMENT_NODES or any(
        type(node) not in allowed or node.comments for node in nodes
    ):
        raise InvalidMetricPatchError()
    if predicate and not _boolean_shape(expression):
        raise InvalidMetricPatchError()
    for star in expression.find_all(exp.Star):
        if not isinstance(star.parent, exp.Count):
            raise InvalidMetricPatchError()
    return expression


def _boolean_shape(expression: exp.Expr) -> bool:
    value = expression.unnest()
    if isinstance(value, (exp.And, exp.Or)):
        return _boolean_shape(value.this) and _boolean_shape(value.expression)
    if isinstance(value, exp.Not):
        return _boolean_shape(value.this)
    return isinstance(value, _BOOLEAN_ROOTS)


def canonical_filter(value: str) -> str:
    """Canonical SQL identifies the same predicate across precedence layers."""
    return parse_fragment(value, predicate=True).unnest().sql(dialect="postgres")


def predicates(expression: exp.Expr | None) -> list[exp.Expr]:
    """Flatten conjunctions only; OR and NOT retain their original grouping."""
    if expression is None:
        return []
    expression = expression.unnest()
    if isinstance(expression, exp.And):
        return predicates(expression.this) + predicates(expression.expression)
    return [expression]


def canonical_filters(values: list[str]) -> list[str]:
    """Split conjunctions so additions can subsequently be removed by identity."""
    return list(
        dict.fromkeys(
            part.sql(dialect="postgres")
            for value in values
            for part in predicates(parse_fragment(value, predicate=True))
        )
    )


@dataclass(frozen=True)
class QueryScope:
    """Internal AST scope plus columns visible without adding a join."""

    select: exp.Select
    columns: dict[str, SemanticType | None]
    business: bool


def _columns(scope: Scope, schema: SchemaCatalog) -> dict[str, SemanticType | None]:
    result: dict[str, SemanticType | None] = {}
    tables = {table.table_name: table for table in schema.tables}
    for alias, (_node, source) in scope.selected_sources.items():
        if isinstance(source, exp.Table):
            table = tables.get(f"{source.db}.{source.name}")
            if table is None:
                raise MetricCatalogError("Template references an unknown table.")
            result.update({f"{alias}.{c.column_name}": c.semantic_type for c in table.columns})
        elif isinstance(source, Scope):
            if not isinstance(source.expression, exp.Select):
                raise MetricCatalogError("Template derived sources must be SELECT queries.")
            result.update({f"{alias}.{name}": None for name in source.expression.named_selects})
    return result


def query_scopes(query: exp.Select, schema: SchemaCatalog) -> list[QueryScope]:
    """Resolve CTE sources locally, never through a global alias exemption."""
    return [
        QueryScope(
            select=scope.expression,
            columns=_columns(scope, schema),
            business=any(
                isinstance(source, exp.Table) for _node, source in scope.selected_sources.values()
            ),
        )
        for scope in traverse_scope(query)
        if isinstance(scope.expression, exp.Select)
    ]


def _references(expression: exp.Expr) -> set[str]:
    columns = list(expression.find_all(exp.Column))
    if any(not c.table or c.db or c.catalog for c in columns):
        raise InvalidMetricPatchError()
    return {f"{c.table}.{c.name}" for c in columns}


def _eligible(expression: exp.Expr, scopes: list[QueryScope]) -> list[QueryScope]:
    refs = _references(expression)
    selected = [scope for scope in scopes if scope.business and refs <= scope.columns.keys()]
    if not selected:
        raise InvalidMetricPatchError()
    return selected


def _where(scope: QueryScope) -> list[exp.Expr]:
    where = scope.select.args.get("where")
    return predicates(where.this if isinstance(where, exp.Where) else None)


def _set_where(scope: QueryScope, values: list[exp.Expr]) -> None:
    if not values:
        raise InvalidMetricPatchError()  # Published time constraints must survive any patch.
    scope.select.set("where", exp.Where(this=exp.and_(*values)))


def _filter_edit(scopes: list[QueryScope], value: str, *, remove: bool) -> None:
    predicate = parse_fragment(value, predicate=True)
    target = predicate.unnest().sql(dialect="postgres")
    eligible = _eligible(predicate, scopes)
    for scope in eligible:
        current = _where(scope)
        matches = [p for p in current if p.sql(dialect="postgres") == target]
        if remove:
            current = [p for p in current if p.sql(dialect="postgres") != target]
        elif not matches:
            current.append(predicate.copy())
        _set_where(scope, current)


def _date_edit(
    scopes: list[QueryScope], definition: MetricDefinition, old: str, new: str, period: Period
) -> None:
    column = parse_fragment(new)
    if not isinstance(column, exp.Column):
        raise InvalidMetricPatchError()
    key = next(iter(_references(column)))
    targets = [s for s in scopes if s.business]
    if definition.key == "refund_rate":
        targets = [s for s in targets if s.select.parent and s.select.parent.alias == "numerator"]
    if not targets or any(s.columns.get(key) is not SemanticType.TIMESTAMP for s in targets):
        raise InvalidMetricPatchError()
    original = sqlglot.parse_one(old, read="postgres")
    bounds: dict[type[exp.Expr], exp.Expr] = {
        exp.GTE: sqlglot.parse_one(
            "CAST('" + period.start.isoformat() + "' AS TIMESTAMPTZ)", read="postgres"
        ),
        exp.LT: sqlglot.parse_one(
            "CAST('" + period.end.isoformat() + "' AS TIMESTAMPTZ)", read="postgres"
        ),
    }
    for scope in targets:
        _replace_period(scope, column, original, bounds)


def _replace_period(
    scope: QueryScope,
    column: exp.Column,
    original: exp.Expr,
    bounds: dict[type[exp.Expr], exp.Expr],
) -> None:
    # Eligibility filters such as o.paid_at IS NOT NULL stay independent.
    matched = 0
    for predicate in _where(scope):
        if (
            type(predicate) in bounds
            and predicate.this == original
            and predicate.expression == bounds[type(predicate)]
        ):
            predicate.set("this", column.copy())
            matched += 1
    if matched != len(bounds):
        raise MetricCatalogError("Published metric period slots could not be located.")
    for reference in list(scope.select.expressions[0].find_all(exp.Column)):
        if reference == original:
            reference.replace(column.copy())


def _expression_edit(scopes: list[QueryScope], query: exp.Select, value: str) -> None:
    expression = parse_fragment(value)
    scope = next(s for s in scopes if s.select is query)
    if not _references(expression) <= scope.columns.keys():
        raise InvalidMetricPatchError()
    aggregates = list(expression.find_all(exp.AggFunc))
    if scope.business:
        if not aggregates:
            raise InvalidMetricPatchError()
        for column in expression.find_all(exp.Column):
            if column.find_ancestor(exp.AggFunc) is None:
                raise InvalidMetricPatchError()
        if any(a.find_ancestor(exp.AggFunc) is not None for a in aggregates):
            raise InvalidMetricPatchError()
    elif aggregates:
        raise InvalidMetricPatchError()  # The refund ratio's final projection is already aggregated.
    projection = query.expressions[1]
    if not isinstance(projection, exp.Alias):
        raise MetricCatalogError("Metric projection must have an alias.")
    projection.set("this", expression)


def apply_sql_patch(  # noqa: PLR0913 -- explicit pure transformation inputs.
    query: exp.Select,
    patch: MetricPatch,
    *,
    definition: MetricDefinition,
    schema: SchemaCatalog,
    date_field: str,
    removable_filters: list[str],
    period: Period,
) -> exp.Select:
    """Apply atomically to a copy; callers decide fallback versus clarification."""
    result = query.copy()
    scopes = query_scopes(result, schema)
    additions = set(canonical_filters(patch.add_filters))
    removals = set(canonical_filters(patch.remove_filters))
    if additions & removals or not removals <= set(removable_filters):
        raise InvalidMetricPatchError()
    for value in sorted(removals):
        _filter_edit(scopes, value, remove=True)
    for value in sorted(additions):
        _filter_edit(scopes, value, remove=False)
    if patch.date_field is not None:
        _date_edit(scopes, definition, date_field, patch.date_field, period)
    if patch.expression is not None:
        _expression_edit(scopes, result, patch.expression)
    return result
