"""Validate finalized bindings; discard rejected saved patches once, never explicit intent."""

import sqlglot
import structlog

from app.agents.runtime import McpPort
from app.core.deadline import Deadline
from app.core.errors import McpPolicyRejected
from app.schemas.metric_tools import ResolveMetricArgs
from app.schemas.schema_catalog import SchemaCatalog
from app.services.metric_binding import BindingRequest, BindingResult, build_binding

logger = structlog.get_logger(__name__)


async def _validate(result: BindingResult, request: BindingRequest, mcp: McpPort,
                    deadline: Deadline) -> BindingResult:
    binding = result.binding
    query = sqlglot.parse_one(binding.resolved_expression, read="postgres")
    fragment = await mcp.resolve_metric(
        ResolveMetricArgs(
            metric_key=binding.metric_key,
            expression=query.expressions[1].this.sql(dialect="postgres"),
            resolved_sql=binding.resolved_expression,
            base_tables=request.definition.base_tables,
            date_field=binding.date_field,
            period_start=binding.period_start,
            period_end=binding.period_end,
            filters=binding.filters_applied,
            grain=binding.grain,
        ),
        deadline=deadline,
    )
    result.binding = type(binding).model_validate(
        {**binding.model_dump(mode="json"), "resolved_expression": fragment.normalized_sql}
    )
    return result


async def resolve_binding(
    request: BindingRequest, schema: SchemaCatalog, mcp: McpPort, *, deadline: Deadline
) -> BindingResult:
    """Only a policy refusal of an applied saved patch permits a new default candidate."""
    result = build_binding(request, schema)
    try:
        return await _validate(result, request, mcp, deadline)
    except McpPolicyRejected as exc:
        if result.binding.override_id is None:
            raise
        logger.exception(
            "metric_saved_patch_rejected",
            metric_key=request.definition.key,
            override_id=str(result.binding.override_id),
            reasons=[reason.value for reason in exc.reasons],
            exc_info=False,
        )
    deadline.check("metric_override_fallback")
    fallback_request = request.model_copy(update={"override": None})
    fallback = build_binding(fallback_request, schema)
    fallback.assumptions.append(
        f"{request.definition.display_name}保存的指标偏好未通过数据访问校验。"
        "以公司口径为基础应用本次请求。"
    )
    return await _validate(fallback, fallback_request, mcp, deadline)
