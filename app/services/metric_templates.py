"""Pure rendering of published SQL templates, never LLM SQL generation."""

from zoneinfo import ZoneInfo

from jinja2 import StrictUndefined, TemplateError, meta, nodes
from jinja2.sandbox import SandboxedEnvironment

from app.core.errors import MetricCatalogError, UnsupportedGrain
from app.schemas.metrics import (
    Grain,
    MetricDateField,
    MetricDefinition,
    MetricRenderContext,
    MetricTemplateContext,
    example_period,
)
from app.schemas.schema_catalog import BUSINESS_TABLES

_ENV = SandboxedEnvironment(undefined=StrictUndefined, autoescape=False)
_ALLOWED_NODES = (
    nodes.Template,
    nodes.Output,
    nodes.TemplateData,
    nodes.Name,
    nodes.If,
    nodes.Compare,
    nodes.Operand,
    nodes.Const,
)
INITIAL_KEYS = frozenset(
    {"gmv", "order_count", "aov", "active_customer", "refund_rate", "refund_count"}
)


def validate_grain(definition: MetricDefinition, grain: Grain) -> None:
    """Report supported values as structured data, never infer a substitute."""
    if grain not in definition.supported_grains:
        raise UnsupportedGrain([item.value for item in definition.supported_grains])


def validate_template(definition: MetricDefinition) -> None:
    """Reject executable Jinja features and unknown slots before deployment."""
    if set(definition.base_tables) - set(BUSINESS_TABLES):
        raise MetricCatalogError("Unknown business table.")
    try:
        parsed = _ENV.parse(definition.expression_template)
        if meta.find_undeclared_variables(parsed) - MetricTemplateContext.model_fields.keys():
            raise MetricCatalogError("Unknown template slot.")
        if any(not isinstance(node, _ALLOWED_NODES) for node in parsed.find_all(nodes.Node)):
            raise MetricCatalogError("Unsupported template construct.")
    except TemplateError as exc:
        raise MetricCatalogError() from exc


def _bucket(grain: Grain, date_field: MetricDateField) -> str:
    if grain is Grain.TOTAL:
        return "'total'"
    if grain is Grain.REGION:
        return "o.region_id"
    if grain is Grain.CATEGORY:
        return "p.category"
    # Both interpolated values are closed enums, never user text.
    return (
        "date_trunc('" + grain.value + "', " + date_field.value + " AT TIME ZONE 'Asia/Shanghai')"
    )


def render_expression(definition: MetricDefinition, context: MetricRenderContext) -> str:
    """Render a complete query template with fixed columns and aware timestamp literals."""
    validate_grain(definition, context.grain)
    validate_template(definition)
    timezone = ZoneInfo("Asia/Shanghai")
    slots = MetricTemplateContext(
        period_start=context.period_start.astimezone(timezone).isoformat(),
        period_end=context.period_end.astimezone(timezone).isoformat(),
        grain=context.grain,
        date_field=definition.default_date_field,
        bucket=_bucket(context.grain, definition.default_date_field),
        paid_bucket=_bucket(context.grain, MetricDateField.PAID),
    )
    try:
        return _ENV.from_string(definition.expression_template).render(**slots.model_dump()).strip()
    except TemplateError as exc:
        raise MetricCatalogError() from exc


def validate_definitions(definitions: list[MetricDefinition]) -> None:
    """Validate active catalog completeness, uniqueness and every declared grain."""
    keys = [item.key for item in definitions]
    if len(keys) != len(set(keys)) or not set(keys) >= INITIAL_KEYS:
        raise MetricCatalogError("Missing or duplicate active metric keys.")
    start, end = example_period()
    for item in definitions:
        if not item.is_active:
            raise MetricCatalogError("Inactive definition in active catalog.")
        for grain in item.supported_grains:
            render_expression(
                item, MetricRenderContext(period_start=start, period_end=end, grain=grain)
            )


def render_catalog_block(definitions: list[MetricDefinition]) -> str:
    """Render all catalog semantics deterministically, including historical versions."""
    blocks = [
        "\n".join(
            [
                f"Metric: {item.key} v{item.version} — {item.display_name}",
                "Assumptions: " + item.description,
                "Tables: " + ", ".join(item.base_tables),
                "Date field: " + item.default_date_field.value,
                "Required filters: " + " AND ".join(item.required_filters),
                "Grains: " + ", ".join(g.value for g in item.supported_grains),
                "Expression template:\n" + item.expression_template.strip(),
                *[
                    f"Question: {e.question}\nAnswer: {e.answer}\nSQL:\n{e.sql.strip()}"
                    for e in item.examples
                ],
            ]
        )
        for item in sorted(definitions, key=lambda d: (d.key, d.version))
    ]
    return "\n\n".join(blocks) + "\n"
