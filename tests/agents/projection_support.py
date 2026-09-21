"""Typed routed inputs shared by isolated projection and wrapper acceptance."""

from datetime import UTC, datetime
from uuid import uuid4

from app.agents.contracts import PreparedContext, Route, RouteDecision
from app.agents.runtime import RuntimeContext
from app.agents.state import AgentState, TurnContext
from app.schemas.knowledge_query import KnowledgeHistoryTurn
from app.schemas.memory import (
    FormatPreferenceContent,
    Memory,
    MemoryType,
    MetricOverrideContent,
    RegionFocusContent,
    TerminologyContent,
)
from app.schemas.metric_resolution import (
    MetricPatch,
    MetricPatchEntry,
    MetricPatches,
    RegionScope,
    SelectedMetricOverride,
    SelectedOverrides,
)
from app.schemas.retrieval import PointTimeScope
from app.services.periods import resolve_period


def selected_context(ctx: RuntimeContext) -> TurnContext:
    """Finalized fixtures represent supplied context, never a memory lookup."""
    payloads = [
        (MemoryType.TERMINOLOGY, TerminologyContent(term="营收", means="GMV")),
        (
            MemoryType.METRIC_OVERRIDE,
            MetricOverrideContent(metric_key="gmv", patch=MetricPatch(date_field="o.paid_at")),
        ),
        (MemoryType.REGION_FOCUS, RegionFocusContent(region_ids=[2])),
        (MemoryType.FORMAT_PREFERENCE, FormatPreferenceContent(prefer="table", decimals=2)),
    ]
    memories = [
        Memory(
            id=uuid4(),
            user_id=ctx.identity.user_id,
            source_turn_id=uuid4(),
            memory_type=kind,
            content=content,
            summary="reference",
            confidence=1,
        )
        for kind, content in payloads
    ]
    scope = PointTimeScope(as_of="2026-08-01")
    return TurnContext(
        time_scope=scope,
        memories=memories,
        summary="private knowledge summary",
        region_scope=RegionScope(region_ids=[1]),
        selected_overrides=SelectedOverrides(
            items=[
                SelectedMetricOverride(
                    id=memories[1].id,
                    user_id=ctx.identity.user_id,
                    created_at=datetime(2026, 8, 1, tzinfo=UTC),
                    confidence=1,
                    metric_key="gmv",
                    patch=MetricPatch(date_field="o.paid_at"),
                )
            ]
        ),
        explicit_patch=MetricPatches(
            items=[MetricPatchEntry(metric_key="gmv", patch=MetricPatch(date_field="o.created_at"))]
        ),
        reference_period=resolve_period("2026年7月", now=ctx.now),
        prior_sql=["SELECT 41", "SELECT 42"],
        knowledge_history=[
            KnowledgeHistoryTurn(
                turn_id=uuid4(), question="退款政策", answer_summary="政策摘要", time_scope=scope
            )
        ],
        format_preference=FormatPreferenceContent(prefer="table", decimals=2),
    )


def routed_state(ctx: RuntimeContext, route: Route = Route.BOTH) -> AgentState:
    """Use a fully specified question so a first-turn knowledge call never rewrites."""
    question = "2026年8月1日的GMV和退款政策"
    return AgentState(
        **ctx.identity.model_dump(),
        question=question,
        prepared=PreparedContext(
            question=question, summary="legacy private", messages=[], prior_sql=[]
        ),
        route=RouteDecision(
            route=route,
            confidence=1,
            data_intent="2026年8月1日的GMV",
            knowledge_intent="2026年8月1日的退款政策",
            metric_hints=["gmv"],
        ),
        context=selected_context(ctx),
    )
