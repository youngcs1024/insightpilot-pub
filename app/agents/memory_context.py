"""Finalize typed intent and preferences through injected services only."""

import structlog
from pydantic import Field

from app.agents.contracts import Route
from app.agents.failures import NodeFailure
from app.agents.nodes.common import node_failure
from app.agents.runtime import RuntimeContext
from app.agents.state import AgentState
from app.core.errors import ConflictError, InsightPilotError
from app.schemas.intent import MetricIntentInput
from app.schemas.mcp import Contract
from app.schemas.memory import (
    FormatPreferenceContent, MetricOverrideContent, RegionFocusContent, TerminologyContent,
    TerminologyProjection,
)
from app.schemas.memory_retrieval import MemoryReadRequest, MemorySelection, MemoryStage
from app.schemas.metric_resolution import (
    ClarificationKind, MetricClarification, MetricIntent, MetricPatches, RegionScope,
    SelectedMetricOverride, SelectedOverrides,
)
from app.services.memory.retrieve import effective_patch

logger = structlog.get_logger(__name__)


class FinalMemoryContext(Contract):
    selection: MemorySelection = Field(default_factory=MemorySelection)
    intent: MetricIntent | None = None
    intent_failure: NodeFailure | None = None
    region: RegionScope | None = None
    region_mentioned: bool | None = None
    clarification: MetricClarification | None = None
    overrides: SelectedOverrides = Field(default_factory=SelectedOverrides)
    format_preference: FormatPreferenceContent | None = None


async def _intent(state: AgentState, ctx: RuntimeContext) -> FinalMemoryContext:
    result = FinalMemoryContext()
    route = state.route
    if route is None:
        raise ConflictError("missing route")
    reference = route.region
    result.region_mentioned = route.region_mentioned
    if route.route in {Route.DATA_ONLY, Route.BOTH}:
        if ctx.metric_intents is None:
            raise ConflictError("missing metric intent service")
        try:
            result.intent = await ctx.metric_intents.interpret(
                MetricIntentInput(
                    question=state.question, data_intent=route.data_intent,
                    metric_hints=route.metric_hints,
                    terminology=[TerminologyProjection(term=row.content.term, means=row.content.means)
                        for row in state.memory_preselection.selected
                        if isinstance(row.content, TerminologyContent)],
                ), deadline=ctx.deadline, now=ctx.now,
            )
            reference = result.intent.region
            result.region_mentioned = result.intent.region_mentioned
        except Exception as exc:
            ctx.deadline.check("metric_intent_failed")
            error = exc if isinstance(exc, InsightPilotError) else InsightPilotError()
            logger.exception("metric_intent_failed", code=error.code, exc_info=False)
            result.intent_failure = node_failure("resolve_metrics", error)
            result.region_mentioned = None
            return result
    if route.route is Route.CLARIFY:
        return result
    if reference.names or reference.all_regions:
        result.region_mentioned = True
        try:
            result.region = await ctx.regions.resolve(reference, deadline=ctx.deadline)
        except InsightPilotError as exc:
            if route.route not in {Route.DATA_ONLY, Route.BOTH}:
                raise
            ctx.deadline.check("metric_region_failed")
            logger.exception("metric_region_failed", code=exc.code, exc_info=False)
            result.intent_failure = node_failure("resolve_metrics", exc)
            return result
    if result.region_mentioned and result.region is None:
        result.clarification = MetricClarification(
            kind=ClarificationKind.REGION_UNRESOLVED,
            message="请明确区域，也可明确选择全部区域。",
        )
    return result


async def finalize_memories(state: AgentState, ctx: RuntimeContext) -> FinalMemoryContext:
    """Only this stage reads metric/region memories, after interpreting current intent."""
    result = await _intent(state, ctx)
    route = state.route
    if route is None:
        raise ConflictError("missing route")
    if state.memory_disabled or state.memory_preselection.failed or ctx.memories is None:
        return result
    result.selection = await ctx.memories.retrieve(
        MemoryReadRequest(
            user_id=ctx.identity.user_id, question=state.question, stage=MemoryStage.FINALIZE,
            data_route=route.route in {Route.DATA_ONLY, Route.BOTH},
            clarify=route.route is Route.CLARIFY or result.clarification is not None,
            metric_keys=result.intent.metric_keys if result.intent else [],
            explicit_patch=result.intent.explicit_patch if result.intent else MetricPatches(),
            region_mentioned=result.region_mentioned,
        ), deadline=ctx.deadline, counter=ctx.schema_token_counter,
    )
    for row in result.selection.selected:
        if isinstance(row.content, MetricOverrideContent):
            explicit = result.intent.explicit_patch.for_metric(row.content.metric_key) if result.intent else None
            result.overrides.items.append(SelectedMetricOverride(
                id=row.id, user_id=row.user_id, created_at=row.created_at, confidence=row.confidence,
                metric_key=row.content.metric_key,
                patch=effective_patch(row.content.patch, explicit) if explicit else row.content.patch.model_copy(deep=True),
            ))
        elif isinstance(row.content, RegionFocusContent) and result.region_mentioned is False:
            result.region = RegionScope(region_ids=sorted(set(row.content.region_ids)))
        elif isinstance(row.content, FormatPreferenceContent):
            result.format_preference = row.content.model_copy(deep=True)
    return result
