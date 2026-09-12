"""Deterministic metric precedence with SQL and explanations derived from one result."""

from uuid import UUID

import sqlglot
import structlog
from pydantic import Field
from sqlglot import exp

from app.agents.contracts import ResolvedMetricBinding
from app.core.errors import InvalidMetricPatchError, MetricCatalogError
from app.schemas.mcp import Contract
from app.schemas.metric_resolution import (
    BindingField,
    BindingFieldSource,
    BindingSource,
    MetricPatch,
    RegionScope,
    SelectedMetricOverride,
)
from app.schemas.metrics import Grain, MetricDefinition, MetricRenderContext
from app.schemas.schema_catalog import SchemaCatalog
from app.services.metric_patch_sql import (
    apply_sql_patch,
    canonical_filter,
    canonical_filters,
    parse_fragment,
)
from app.services.metric_templates import render_expression
from app.services.periods import Period

logger = structlog.get_logger(__name__)
MAX_PATCH_FILTERS = 16
MAX_RESOLVED_SQL = 32_000
EXPRESSION_CONFIDENCE = 0.9
_FILTER_LABELS = {
    canonical_filter("o.paid_at IS NOT NULL"): "仅统计已支付订单",
    canonical_filter("o.status <> 'cancelled'"): "排除已取消订单",
    canonical_filter("c.is_test_account = false"): "排除测试账号",
    canonical_filter("r.status <> 'rejected'"): "排除已拒绝退款申请",
    canonical_filter("r.status = 'completed'"): "仅统计已完成退款",
}
_DATE_LABELS = {
    "o.paid_at": "订单支付时间",
    "o.created_at": "订单创建时间",
    "r.requested_at": "退款申请时间",
    "r.completed_at": "退款完成时间",
}


class BindingRequest(Contract):
    """All resolution inputs are explicit; no repository or clock access is permitted."""

    definition: MetricDefinition
    period: Period
    grain: Grain
    user_id: UUID
    explicit_patch: MetricPatch = Field(default_factory=MetricPatch)
    override: SelectedMetricOverride | None = None
    region_scope: RegionScope | None = None


class BindingResult(Contract):
    """The same resolved artifact supplies downstream SQL guidance and assumptions."""

    binding: ResolvedMetricBinding
    assumptions: list[str]


class _Resolution(Contract):
    sql: str
    date_field: str
    filters: list[str]
    sources: list[BindingFieldSource]


def merge_explicit(lower: MetricPatch, higher: MetricPatch) -> MetricPatch:
    """Finalized explicit fields beat node extraction; operations merge by predicate identity."""
    operations: dict[str, bool] = {}
    for patch in (lower, higher):
        additions = set(canonical_filters(patch.add_filters))
        removals = set(canonical_filters(patch.remove_filters))
        if additions & removals:
            raise InvalidMetricPatchError()
        operations.update(dict.fromkeys(sorted(removals), False))
        operations.update(dict.fromkeys(sorted(additions), True))
    additions = {value for value, add in operations.items() if add}
    removals = operations.keys() - additions
    if max(len(additions), len(removals)) > MAX_PATCH_FILTERS:
        raise InvalidMetricPatchError()
    return MetricPatch(
        date_field=higher.date_field if higher.date_field is not None else lower.date_field,
        expression=higher.expression if higher.expression is not None else lower.expression,
        add_filters=sorted(additions),
        remove_filters=sorted(removals),
    )


def _record(
    sources: list[BindingFieldSource],
    field: BindingField,
    value: str,
    source: BindingSource,
    *,
    applied: bool = True,
) -> None:
    sources[:] = [
        item
        for item in sources
        if not (item.field is field and (field is not BindingField.FILTER or item.value == value))
    ]
    sources.append(BindingFieldSource(field=field, value=value, source=source, applied=applied))


def _validate_region_patch(patch: MetricPatch, region: RegionScope | None) -> None:
    if region is None:
        return
    for value in canonical_filters(patch.add_filters):
        refs = {
            f"{c.table}.{c.name}"
            for c in parse_fragment(value, predicate=True).find_all(exp.Column)
        }
        # ANDs have already been split. An OR combining region and another field
        # cannot be discarded without silently changing that other restriction.
        if "o.region_id" in refs and refs != {"o.region_id"}:
            raise InvalidMetricPatchError()


def _apply(
    current: _Resolution,
    patch: MetricPatch,
    source: BindingSource,
    request: BindingRequest,
    schema: SchemaCatalog,
) -> _Resolution:
    _validate_region_patch(patch, request.region_scope)
    query = sqlglot.parse_one(current.sql, read="postgres")
    if not isinstance(query, exp.Select):
        raise MetricCatalogError("Expected a complete metric SELECT.")
    removable = list(
        dict.fromkeys(
            [
                *current.filters,
                *canonical_filters(request.definition.required_filters),
                *(s.value for s in current.sources if s.field is BindingField.FILTER),
            ]
        )
    )
    query = apply_sql_patch(
        query,
        patch,
        definition=request.definition,
        schema=schema,
        date_field=current.date_field,
        removable_filters=removable,
        period=request.period,
    )
    result = current.model_copy(deep=True)
    result.sql = query.sql(dialect="postgres")
    if len(result.sql) > MAX_RESOLVED_SQL:
        raise InvalidMetricPatchError()
    _record_patch(result, patch, source, query)
    return result


def _record_patch(
    result: _Resolution, patch: MetricPatch, source: BindingSource, query: exp.Select
) -> None:
    for canonical in canonical_filters(patch.remove_filters):
        result.filters = [f for f in result.filters if f != canonical]
        _record(result.sources, BindingField.FILTER, canonical, source, applied=False)
    for canonical in canonical_filters(patch.add_filters):
        if canonical not in result.filters:
            result.filters.append(canonical)
        _record(result.sources, BindingField.FILTER, canonical, source)
    if patch.date_field is not None:
        result.date_field = parse_fragment(patch.date_field).sql(dialect="postgres")
        _record(result.sources, BindingField.DATE, result.date_field, source)
    if patch.expression is not None:
        projection = query.expressions[1]
        _record(result.sources, BindingField.EXPRESSION, projection.this.sql("postgres"), source)


def _initial(request: BindingRequest) -> _Resolution:
    definition = request.definition
    sql = render_expression(
        definition,
        MetricRenderContext(
            period_start=request.period.start,
            period_end=request.period.end,
            grain=request.grain,
        ),
    )
    filters = canonical_filters(definition.required_filters)
    query = sqlglot.parse_one(sql, read="postgres")
    sources = [
        BindingFieldSource(
            field=BindingField.DATE,
            value=definition.default_date_field.value,
            source=BindingSource.COMPANY,
        ),
        BindingFieldSource(
            field=BindingField.EXPRESSION,
            value=query.expressions[1].this.sql("postgres"),
            source=BindingSource.COMPANY,
        ),
        *(
            BindingFieldSource(field=BindingField.FILTER, value=f, source=BindingSource.COMPANY)
            for f in filters
        ),
    ]
    return _Resolution(
        sql=sql,
        date_field=definition.default_date_field.value,
        filters=filters,
        sources=sources,
    )


def _saved(
    initial: _Resolution, request: BindingRequest, schema: SchemaCatalog
) -> tuple[_Resolution, list[str]]:
    override = request.override
    if override is None:
        return initial, []
    valid_identity = (
        override.user_id == request.user_id and override.metric_key == request.definition.key
    )
    low_confidence = (
        override.patch.expression is not None and override.confidence < EXPRESSION_CONFIDENCE
    )
    if not valid_identity or low_confidence:
        return initial, ["保存的指标偏好未通过归属或置信度校验。以公司口径为基础应用本次请求。"]
    try:
        return _apply(initial, override.patch, BindingSource.SAVED, request, schema), []
    except InvalidMetricPatchError:
        logger.exception(
            "metric_saved_patch_invalid", metric_key=request.definition.key, exc_info=False
        )
        return initial, ["保存的指标偏好无法应用于当前查询。以公司口径为基础应用本次请求。"]


def _description(request: BindingRequest, resolved: _Resolution) -> str:
    projection = (
        sqlglot.parse_one(resolved.sql, read="postgres").expressions[1].this.sql("postgres")
    )
    removed = [
        s.value for s in resolved.sources if s.field is BindingField.FILTER and not s.applied
    ]
    filters = [f"{_FILTER_LABELS.get(f, '限定条件')} ({f})" for f in resolved.filters]
    date_label = _DATE_LABELS.get(resolved.date_field, "指定时间字段")
    parts = [
        f"{request.definition.display_name} v{request.definition.version}",
        f"时间字段: {date_label} ({resolved.date_field})",
        f"统计粒度: {request.grain.value}",
        "实际指标表达式: " + projection,
        "实际过滤条件: " + (" AND ".join(filters) or "无附加过滤"),
    ]
    if removed:
        parts.append("已移除过滤条件: " + ", ".join(removed))
    if request.definition.key == "refund_rate":
        parts.append(
            "分子按所选时间字段统计退款订单，分母按 o.paid_at 统计同期支付订单，两侧独立聚合"
        )
    return "；".join(parts)


def _region(resolved: _Resolution, request: BindingRequest, schema: SchemaCatalog) -> _Resolution:
    region = request.region_scope
    if region is None:
        return resolved
    removals = [
        value
        for value in resolved.filters
        if any(
            c.table == "o" and c.name == "region_id"
            for c in parse_fragment(value, predicate=True).find_all(exp.Column)
        )
    ]
    if len(removals) > MAX_PATCH_FILTERS:
        raise InvalidMetricPatchError()
    additions = []
    if region.region_ids:
        additions = [exp.column("region_id", table="o").isin(*region.region_ids).sql("postgres")]
    # Remove first so an identical finalized scope is allowed to replace its source.
    resolved = _apply(
        resolved, MetricPatch(remove_filters=removals), BindingSource.REGION, request, schema
    )
    return _apply(
        resolved, MetricPatch(add_filters=additions), BindingSource.REGION, request, schema
    )


def build_binding(request: BindingRequest, schema: SchemaCatalog) -> BindingResult:
    """Resolve defaults, saved patch, explicit patch and finalized region atomically."""
    resolved, notes = _saved(_initial(request), request, schema)
    resolved = _apply(resolved, request.explicit_patch, BindingSource.EXPLICIT, request, schema)
    resolved = _region(resolved, request, schema)
    saved_used = any(s.source is BindingSource.SAVED for s in resolved.sources)
    override = request.override if saved_used else None
    binding = ResolvedMetricBinding(
        metric_key=request.definition.key,
        definition_version=request.definition.version,
        resolved_expression=resolved.sql,
        date_field=resolved.date_field,
        period_start=request.period.start,
        period_end=request.period.end,
        filters_applied=resolved.filters,
        grain=request.grain,
        region_scope=request.region_scope,
        override_id=override.id if override else None,
        resolved_description=_description(request, resolved),
        field_sources=resolved.sources,
    )
    assumptions = [binding.resolved_description, request.period.as_assumption(), *notes]
    if override:
        assumptions.append(f"应用了你的自定义定义，设置于 {override.created_at.date().isoformat()}")
    return BindingResult(binding=binding, assumptions=assumptions)
