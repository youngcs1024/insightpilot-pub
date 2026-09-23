"""Independently validate a complete app-rendered metric query against live metadata."""

import re
from datetime import datetime
from typing import Never

import sqlglot
import structlog
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from sqlglot.optimizer.scope import Scope, traverse_scope

from app.core.errors import McpPolicyRejected
from app.core.sql_policy.sql_validator import SQLValidator
from app.schemas.mcp import PolicyReason, ValidationStatus
from app.schemas.metric_tools import MetricFragment, ResolveMetricArgs
from app.schemas.schema_catalog import SemanticType
from app.schemas.schema_tools import GetSchemaArgs, SchemaResponse
from mcp_server.config import MetricPolicySettings
from mcp_server.policy.allowlist import ALLOWED_TABLES
from mcp_server.tools.get_schema import SchemaTool

logger = structlog.get_logger(__name__)
MAX_METRIC_SQL = 32_000
METRIC_PROJECTION_INDEX = 1
MIN_USING_SOURCES = 2
_BOOLEAN = (
    exp.And,
    exp.Or,
    exp.Not,
    exp.EQ,
    exp.NEQ,
    exp.GT,
    exp.GTE,
    exp.LT,
    exp.LTE,
    exp.Is,
    exp.In,
    exp.Between,
    exp.Exists,
    exp.Boolean,
)
_SAFE_IDENTIFIER = re.compile(r"[a-z_][a-z0-9_]*\Z")


def _reject(reason: PolicyReason, *, column: str | None = None) -> Never:
    safe_column = column if column and _SAFE_IDENTIFIER.fullmatch(column) else None
    status = (
        ValidationStatus.UNSAFE
        if reason is PolicyReason.TABLE_NOT_ALLOWED
        else ValidationStatus.INVALID
    )
    raise McpPolicyRejected(status, [reason], column_name=safe_column)


def _limit(start: datetime, years: int) -> datetime:
    """Use calendar years; February 29 clamps to February 28 when necessary."""
    if start.year + years > datetime.max.year:
        return datetime.max.replace(tzinfo=start.tzinfo)
    try:
        return start.replace(year=start.year + years)
    except ValueError:
        return start.replace(year=start.year + years, day=28)


def _parse_query(sql: str) -> exp.Select:
    try:
        statements = [part for part in sqlglot.parse(sql, read="postgres") if part is not None]
    except (SqlglotError, RecursionError) as exc:
        raise McpPolicyRejected(ValidationStatus.INVALID, [PolicyReason.INVALID_SQL]) from exc
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        _reject(PolicyReason.INVALID_METRIC_BINDING)
    return normalize_identifiers(statements[0], dialect="postgres")


def _sources(scope: Scope, metadata: SchemaResponse) -> dict[str, set[str]]:
    catalog = {table.table_name: table for table in metadata.tables}
    result: dict[str, set[str]] = {}
    try:
        selected = scope.selected_sources
    except SqlglotError as exc:
        raise McpPolicyRejected(ValidationStatus.INVALID, [PolicyReason.UNSUPPORTED_SCOPE]) from exc
    for alias, (_node, source) in selected.items():
        if isinstance(source, exp.Table):
            name = f"{source.db or 'biz'}.{source.name}"
            table = catalog.get(name)
            if table is None:
                _reject(PolicyReason.TABLE_NOT_ALLOWED)
            result[alias] = {column.column_name for column in table.columns}
        elif isinstance(source, Scope) and isinstance(source.expression, exp.Select):
            result[alias] = set(source.outer_columns or source.expression.named_selects)
        else:
            _reject(PolicyReason.UNSUPPORTED_SCOPE)
    return result


def _check_column(column: exp.Column, sources: dict[str, set[str]]) -> None:
    if column.is_star:
        _reject(PolicyReason.INVALID_METRIC_BINDING)
    choices = [sources[column.table]] if column.table in sources else []
    if not column.table:
        choices = list(sources.values())
    if sum(column.name in names for names in choices) != 1:
        _reject(PolicyReason.UNKNOWN_COLUMN, column=column.name)


def _check_using(join: exp.Join, sources: dict[str, set[str]]) -> None:
    for identifier in join.args.get("using") or []:
        count = sum(identifier.name in names for names in sources.values())
        if count < MIN_USING_SOURCES:
            _reject(PolicyReason.UNKNOWN_COLUMN, column=identifier.name)


def _scope_tables(scope: Scope, metadata: SchemaResponse) -> set[str]:
    sources = _sources(scope, metadata)
    actual = {
        f"{source.db or 'biz'}.{source.name}"
        for _node, source in scope.selected_sources.values()
        if isinstance(source, exp.Table)
    }
    for column in scope.columns:
        _check_column(column, sources)
    for join in scope.find_all(exp.Join):
        _check_using(join, sources)
    return actual


def _validate_columns(query: exp.Select, metadata: SchemaResponse, declared: set[str]) -> None:
    actual: set[str] = set()
    try:
        for scope in traverse_scope(query):
            actual.update(_scope_tables(scope, metadata))
    except (SqlglotError, RecursionError) as exc:
        raise McpPolicyRejected(ValidationStatus.INVALID, [PolicyReason.UNSUPPORTED_SCOPE]) from exc
    if not actual or not actual <= declared:
        _reject(PolicyReason.TABLE_NOT_ALLOWED)


def _fragment(value: str, *, boolean: bool = False) -> exp.Expr:
    try:
        statements = [part for part in sqlglot.parse(value, read="postgres") if part is not None]
    except (SqlglotError, RecursionError) as exc:
        raise McpPolicyRejected(ValidationStatus.INVALID, [PolicyReason.INVALID_SQL]) from exc
    if len(statements) != 1 or (boolean and not isinstance(statements[0].unnest(), _BOOLEAN)):
        _reject(PolicyReason.INVALID_METRIC_BINDING)
    fragment = normalize_identifiers(statements[0], dialect="postgres")
    if any(node.comments for node in fragment.walk()):
        _reject(PolicyReason.INVALID_METRIC_BINDING)
    return fragment


def _where_parts(query: exp.Select) -> list[exp.Expr]:
    parts: list[exp.Expr] = []
    for where in query.find_all(exp.Where):
        pending = [where.this]
        while pending:
            part = pending.pop()
            if isinstance(part, exp.And):
                pending.extend((part.this, part.expression))
            else:
                parts.append(part.unnest())
    return parts


def _instant(value: exp.Expr) -> datetime | None:
    if not isinstance(value, exp.Cast) or not isinstance(value.this, exp.Literal):
        return None
    if not isinstance(value.to, exp.DataType) or value.to.this is not exp.DataType.Type.TIMESTAMPTZ:
        return None
    try:
        instant = datetime.fromisoformat(value.this.this)
    except ValueError:
        return None
    return instant if instant.utcoffset() is not None else None


def _validate_fragment_columns(
    fragment: exp.Expr, query: exp.Select, metadata: SchemaResponse
) -> None:
    sources = [_sources(scope, metadata) for scope in traverse_scope(query)]
    for column in fragment.find_all(exp.Column):
        if not any(
            column.table in available and column.name in available[column.table]
            for available in sources
        ):
            _reject(PolicyReason.UNKNOWN_COLUMN, column=column.name)


def _check_binding(query: exp.Select, args: ResolveMetricArgs, metadata: SchemaResponse) -> None:
    if len(query.expressions) <= METRIC_PROJECTION_INDEX or not isinstance(
        query.expressions[METRIC_PROJECTION_INDEX], exp.Alias
    ):
        _reject(PolicyReason.INVALID_METRIC_BINDING)
    projection = query.expressions[METRIC_PROJECTION_INDEX]
    expression = _fragment(args.expression)
    _validate_fragment_columns(expression, query, metadata)
    if projection.alias != args.metric_key or projection.this != expression:
        _reject(PolicyReason.INVALID_METRIC_BINDING)
    date_field = _fragment(args.date_field)
    if not isinstance(date_field, exp.Column) or not date_field.table or date_field.db:
        _reject(PolicyReason.INVALID_METRIC_BINDING)
    _validate_fragment_columns(date_field, query, metadata)
    catalog = {table.table_name: table for table in metadata.tables}
    typed_date = any(
        isinstance(source, exp.Table)
        and alias == date_field.table
        and any(
            column.column_name == date_field.name and column.semantic_type is SemanticType.TIMESTAMP
            for column in catalog[f"{source.db or 'biz'}.{source.name}"].columns
        )
        for scope in traverse_scope(query)
        for alias, (_node, source) in scope.selected_sources.items()
    )
    if not typed_date or not any(column == date_field for column in query.find_all(exp.Column)):
        _reject(PolicyReason.UNKNOWN_COLUMN, column=date_field.name)
    predicates = _where_parts(query)
    for value in args.filters:
        predicate = _fragment(value, boolean=True)
        outcome = SQLValidator().validate(f"SELECT 1 WHERE {value}")
        if outcome.status is not ValidationStatus.VALID:
            raise McpPolicyRejected(outcome.status, outcome.reasons)
        _validate_fragment_columns(predicate, query, metadata)
        if predicate.unnest() not in predicates:
            _reject(PolicyReason.INVALID_METRIC_BINDING)
    for operator, instant in ((exp.GTE, args.period_start), (exp.LT, args.period_end)):
        if not any(
            isinstance(part, operator)
            and part.this == date_field
            and (bound := _instant(part.expression)) is not None
            and bound == instant
            for part in predicates
        ):
            _reject(PolicyReason.INVALID_METRIC_BINDING)


def _fragments(query: exp.Select, args: ResolveMetricArgs) -> tuple[str, str, str, str, list[str]]:
    from_node = query.args.get("from_")
    joins = query.args.get("joins") or []
    group = query.args.get("group")
    where = query.args.get("where")
    complex_query = bool(query.ctes)
    if complex_query:
        field = _fragment(args.date_field).sql(dialect="postgres")
        bounds = (
            f"{field} >= TIMESTAMPTZ '{args.period_start.isoformat()}' AND "
            f"{field} < TIMESTAMPTZ '{args.period_end.isoformat()}'"
        )
        filters = [_fragment(value, boolean=True).sql(dialect="postgres") for value in args.filters]
        where_sql = "WHERE " + " AND ".join([bounds, *filters])
    else:
        where_sql = where.sql(dialect="postgres") if where is not None else ""
    source_parts = [from_node.sql(dialect="postgres") if from_node is not None else ""]
    source_parts.extend(join.sql(dialect="postgres") for join in joins)
    return (
        ", ".join(value.sql(dialect="postgres") for value in query.expressions),
        " ".join(source_parts).strip(),
        where_sql,
        group.sql(dialect="postgres") if group is not None else "",
        ["complex_query_fragments_partial"] if complex_query else [],
    )


class MetricResolver:
    """Validate policy and artifact columns without owning metric definitions."""

    def __init__(self, schema_tool: SchemaTool, settings: MetricPolicySettings) -> None:
        self._schema_tool = schema_tool
        self._settings = settings
        self._validator = SQLValidator()

    async def resolve(self, args: ResolveMetricArgs) -> MetricFragment:
        """Return display fragments and the complete normalized, checked SQL."""
        if args.period_start >= args.period_end:
            _reject(PolicyReason.INVALID_PERIOD)
        if args.period_end > _limit(args.period_start, self._settings.max_period_years):
            _reject(PolicyReason.PERIOD_TOO_LONG)
        declared = set(args.base_tables)
        if len(declared) != len(args.base_tables) or not declared <= ALLOWED_TABLES:
            _reject(PolicyReason.TABLE_NOT_ALLOWED)
        outcome = self._validator.validate(args.resolved_sql)
        if outcome.status is not ValidationStatus.VALID:
            raise McpPolicyRejected(outcome.status, outcome.reasons)
        query = _parse_query(args.resolved_sql)
        if any(node.comments for node in query.walk()):
            _reject(PolicyReason.INVALID_METRIC_BINDING)
        if any(isinstance(node, exp.Extract) for node in query.walk()):
            _reject(PolicyReason.INVALID_METRIC_BINDING)
        if any(
            isinstance(node, exp.Star) and not isinstance(node.parent, exp.Count)
            for node in query.walk()
        ):
            _reject(PolicyReason.INVALID_METRIC_BINDING)
        metadata = await self._schema_tool.read(GetSchemaArgs(tables=args.base_tables))
        _validate_columns(query, metadata, declared)
        _check_binding(query, args, metadata)
        normalized_sql = query.sql(dialect="postgres")
        if len(normalized_sql) > MAX_METRIC_SQL:
            _reject(PolicyReason.INVALID_METRIC_BINDING)
        select, source, where, group, warnings = _fragments(query, args)
        logger.info(
            "metric_binding_validated", table_count=len(declared), complex_query=bool(warnings)
        )
        return MetricFragment(
            select_fragment=select,
            from_fragment=source,
            where_fragment=where,
            group_by_fragment=group,
            normalized_sql=normalized_sql,
            normalized=normalized_sql != args.resolved_sql,
            warnings=warnings,
        )
