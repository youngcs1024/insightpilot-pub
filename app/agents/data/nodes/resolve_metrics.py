"""Resolve metric meaning through injected services before any SQL generation."""

import json

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from langgraph.types import Command

from app.agents.contracts import MetricExamplesSnapshot
from app.agents.data.state import DataAgentState
from app.agents.prompts import METRIC_INTENT
from app.agents.runtime import RuntimeContext
from app.core.errors import (
    InvalidMetricPatchError,
    MetricNotFound,
    PeriodUnresolved,
    UnsupportedGrain,
)
from app.core.llm_config import ModelRole
from app.schemas.metric_resolution import ClarificationKind, MetricClarification, MetricIntent
from app.schemas.metrics import Grain, MetricDefinition
from app.schemas.schema_catalog import SchemaCatalog
from app.services.metric_binding import BindingRequest, BindingResult, build_binding, merge_explicit
from app.services.metric_templates import render_catalog_block, validate_grain
from app.services.periods import Period, build_date_context, resolve_period


def _messages(
    state: DataAgentState, ctx: RuntimeContext, definitions: list[MetricDefinition]
) -> list[BaseMessage]:
    system = "\n\n".join(
        [METRIC_INTENT, build_date_context(now=ctx.now), render_catalog_block(definitions)]
    )
    payload = json.dumps(
        {
            "question": state.question,
            "data_intent": state.data_intent,
            "metric_hints": state.metric_hints,
            "terminology": [item.model_dump() for item in state.relevant_memories],
        },
        ensure_ascii=False,
    )
    return [SystemMessage(content=system), HumanMessage(content=payload)]


def _clarify(
    kind: ClarificationKind,
    message: str,
    *,
    available: list[str],
    key: str | None = None,
    supported: list[str] | None = None,
) -> Command[str]:
    return Command(
        update={
            "metric_bindings": [],
            "metric_examples": [],
            "assumptions": [],
            "clarification": MetricClarification(
                kind=kind,
                message=message,
                metric_key=key,
                available_metrics=available,
                supported_grains=supported or [],
            ),
        }
    )


def _grain(intent: MetricIntent, definition: MetricDefinition) -> Grain:
    supported = [grain.value for grain in definition.supported_grains]
    if intent.grain not in {item.value for item in Grain}:
        raise UnsupportedGrain(supported)
    grain = Grain(intent.grain)
    if intent.dimensions and set(intent.dimensions) != {grain.value}:
        raise UnsupportedGrain(supported)
    validate_grain(definition, grain)
    return grain


def _intent_problem(
    state: DataAgentState, intent: MetricIntent, available: list[str]
) -> Command[str] | None:
    if not intent.metric_keys:
        return _clarify(
            ClarificationKind.METRIC_NOT_IDENTIFIED,
            "请明确要统计的指标，可选: " + ", ".join(available),
            available=available,
        )
    if intent.region_mentioned and state.region_scope is None:
        return _clarify(
            ClarificationKind.REGION_UNRESOLVED,
            "请明确区域，并提供已解析的区域范围。",
            available=available,
        )
    patch_keys = {
        entry.metric_key for entry in [*intent.explicit_patch.items, *state.explicit_patch.items]
    }
    if patch_keys - set(intent.metric_keys):
        return _clarify(
            ClarificationKind.INVALID_EXPLICIT_PATCH,
            "当前口径修改必须对应本次识别的指标。",
            available=available,
        )
    return None


async def resolve_metrics(state: DataAgentState, runtime: Runtime[RuntimeContext]) -> Command[str]:
    """Return either complete bindings or clarification; operational errors propagate."""
    ctx = runtime.context
    ctx.deadline.check("resolve_metrics")
    definitions = await ctx.metrics.list_active(deadline=ctx.deadline)
    available = [item.key for item in definitions]
    intent = await ctx.llm.generate_structured(
        ModelRole.SQL, _messages(state, ctx, definitions), MetricIntent, deadline=ctx.deadline
    )
    # Current explicit names replace an upstream default; bindings retain resolved IDs.
    if intent.region.names or intent.region.all_regions:
        region = await ctx.regions.resolve(intent.region, deadline=ctx.deadline)
        state = state.model_copy(update={"region_scope": region})
        intent = intent.model_copy(update={"region_mentioned": True})
    problem = _intent_problem(state, intent, available)
    if problem is not None:
        return problem
    try:
        period = resolve_period(
            intent.period_expression, now=ctx.now, reference_period=state.reference_period
        )
    except PeriodUnresolved:
        return _clarify(
            ClarificationKind.PERIOD_UNRESOLVED,
            "请明确完整统计期间，例如 2026年8月。",
            available=available,
        )
    schema = await ctx.schema_catalog.snapshot(deadline=ctx.deadline)
    return await _resolve(state, ctx, intent, period, schema, available)


async def _resolve(  # noqa: PLR0913, PLR0917 -- typed node inputs, no hidden state.
    state: DataAgentState,
    ctx: RuntimeContext,
    intent: MetricIntent,
    period: Period,
    schema: SchemaCatalog,
    available: list[str],
) -> Command[str]:
    results: list[BindingResult] = []
    examples: list[MetricExamplesSnapshot] = []
    for key in dict.fromkeys(intent.metric_keys):
        ctx.deadline.check("resolve_metric_binding")
        try:
            definition = await ctx.metrics.get_active(key, deadline=ctx.deadline)
            result = build_binding(
                BindingRequest(
                    definition=definition,
                    period=period,
                    grain=_grain(intent, definition),
                    user_id=ctx.identity.user_id,
                    override=state.selected_overrides.for_metric(key),
                    explicit_patch=merge_explicit(
                        intent.explicit_patch.for_metric(key), state.explicit_patch.for_metric(key)
                    ),
                    region_scope=state.region_scope,
                ),
                schema,
            )
        except MetricNotFound:
            return _clarify(
                ClarificationKind.METRIC_NOT_FOUND,
                "该指标没有已发布定义，请选择目录中的指标。",
                available=available,
                key=key,
            )
        except UnsupportedGrain as exc:
            return _clarify(
                ClarificationKind.UNSUPPORTED_GRAIN,
                "请使用支持的单一统计粒度: " + ", ".join(exc.supported),
                available=available,
                key=key,
                supported=list(exc.supported),
            )
        except InvalidMetricPatchError:
            return _clarify(
                ClarificationKind.INVALID_EXPLICIT_PATCH,
                "当前口径无法安全应用，请明确日期字段、过滤条件或指标表达式。",
                available=available,
                key=key,
            )
        results.append(result)
        examples.append(
            MetricExamplesSnapshot(
                metric_key=definition.key,
                definition_version=definition.version,
                examples=[item.model_copy(deep=True) for item in definition.examples],
            )
        )
    ctx.deadline.check("resolve_metrics_complete")
    return Command(
        update={
            "metric_bindings": [result.binding for result in results],
            "metric_examples": examples,
            "assumptions": list(dict.fromkeys(a for result in results for a in result.assumptions)),
            "clarification": None,
        }
    )
