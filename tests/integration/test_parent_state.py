"""Parent v2 round-trips and recovery using the real protected PostgreSQL service."""

from dataclasses import replace

import pytest

from app.agents.knowledge.nodes.package_evidence import package_evidence
from app.agents.state import AgentState
from app.core.config_models import DatabaseSettings
from app.core.errors import LlmStructuredOutputError
from app.db.session import Database
from app.schemas.knowledge_query import KnowledgeClarification, KnowledgeClarificationKind
from app.schemas.metric_resolution import ClarificationKind, MetricClarification
from app.services.graph import GraphService
from tests.agents.knowledge_support import ranked
from tests.agents.projection_support import selected_context
from tests.agents.support import metric_intent, sql_candidate
from tests.answer_support import data_draft
from tests.fakes.chat_model import FakeChatModel
from tests.integration.checkpoint_support import (
    admitted,
    checkpoint_setup,
    connected_context,
    graph_database,
)
from tests.router_support import decision

pytestmark = pytest.mark.integration
__all__ = ["checkpoint_setup", "graph_database"]


async def test_parent_v2_checkpoint_roundtrip_after_restart(
    graph_database: tuple[Database, DatabaseSettings],
) -> None:
    database, settings = graph_database
    identity = await admitted(database)
    ctx = connected_context(database, identity)
    config = {"configurable": {"thread_id": str(identity.turn_id)}}
    first = GraphService(settings)
    await first.start()
    try:
        original = await first.invoke(ctx)
        await first.graph.aupdate_state(
            config,
            {
                "context": selected_context(ctx),
                "data_clarification": MetricClarification(
                    kind=ClarificationKind.PERIOD_UNRESOLVED, message="data period?"
                ),
                "knowledge_clarification": KnowledgeClarification(
                    kind=KnowledgeClarificationKind.REFERENCE_UNRESOLVED,
                    message="which policy?",
                ),
                "knowledge_abstention_reason": "no matching policy",
                "route": decision(),
                "knowledge_evidence": package_evidence(ranked(), ctx),
                "assumptions": ["old turn"],
                "degraded_components": ["old component"],
                "abstained": True,
            },
            as_node="format_answer",
        )
        saved = AgentState.model_validate((await first.graph.aget_state(config)).values)
    finally:
        await first.aclose()
    restarted = GraphService(settings)
    await restarted.start()
    try:
        restored = AgentState.model_validate((await restarted.graph.aget_state(config)).values)
        assert restored == saved
        assert restored.context.memories[0].content.term == "营收"
        assert restored.context.reference_period == saved.context.reference_period
        assert restored.context.explicit_patch == saved.context.explicit_patch
        assert restored.context.knowledge_history == saved.context.knowledge_history
        assert restored.knowledge_clarification == saved.knowledge_clarification
        assert restored.knowledge_abstention_reason == "no matching policy"
        assert restored.data_clarification == saved.data_clarification
        assert restored.knowledge_evidence.chunks[0].original_text
        replay = await restarted.invoke(replace(ctx, llm=FakeChatModel([])), resume=True)
        assert replay == original.model_copy(update={"route": saved.route})
        assert len(ctx.mcp.calls) == 1
        next_identity = await admitted(database, identity=identity)
        next_ctx = connected_context(database, next_identity, followup=True)
        await restarted.invoke(next_ctx)
        fresh = AgentState.model_validate(
            (
                await restarted.graph.aget_state(
                    {"configurable": {"thread_id": str(next_identity.turn_id)}}
                )
            ).values
        )
        assert fresh.context is not None
        assert fresh.route is not None
        assert fresh.knowledge_evidence is None
        assert fresh.assumptions == fresh.data_evidence.assumptions
        assert fresh.failures == fresh.degraded_components == []
        assert fresh.knowledge_clarification is fresh.data_clarification is None
        assert fresh.knowledge_abstention_reason is None
        assert not fresh.abstained
        assert fresh.answer != restored.answer
        assert fresh.evidence_refs != restored.evidence_refs
        assert fresh.prepared.messages
        assert fresh.question == fresh.prepared.question
    finally:
        await restarted.aclose()


async def test_checkpoint_key_is_assistant_turn(
    graph_database: tuple[Database, DatabaseSettings],
) -> None:
    database, settings = graph_database
    identity = await admitted(database)
    service = GraphService(settings)
    await service.start()
    try:
        ctx = connected_context(database, identity)
        await service.invoke(ctx)
        actual = await service.graph.aget_state(
            {"configurable": {"thread_id": str(identity.turn_id)}}
        )
        wrong = await service.graph.aget_state(
            {"configurable": {"thread_id": str(identity.conversation_id)}}
        )
        assert actual.values["turn_id"] == identity.turn_id
        assert actual.values["graph_version"] == "phase4-v5"
        assert not wrong.values
        assert actual.values["messages"][-1].content == actual.values["prepared"].question
    finally:
        await service.aclose()


async def test_same_turn_recovery_failure_then_success_resets_only_current_failures(
    graph_database: tuple[Database, DatabaseSettings],
) -> None:
    database, settings = graph_database
    identity = await admitted(database)
    ctx = replace(
        connected_context(database, identity),
        llm=FakeChatModel(
            [
                metric_intent(),
                sql_candidate(),
                LlmStructuredOutputError(),
            ]
        ),
    )
    service = GraphService(settings)
    await service.start()
    try:
        first = await service.invoke(ctx)
        assert first.status == "failed"
        assert len(first.failures) == 1
        config = {"configurable": {"thread_id": str(identity.turn_id)}}
        before = await service.graph.checkpointer.aget_tuple(config)
        assert before is not None
        assert before.checkpoint["channel_values"]["failures"] == first.failures
        second = await service.invoke(
            replace(ctx, llm=FakeChatModel([LlmStructuredOutputError()])), resume=True
        )
        assert second.status == "failed"
        assert len(second.failures) == 1
        third = await service.invoke(
            replace(ctx, llm=FakeChatModel([data_draft(markdown="Recovered", confidence=1)])),
            resume=True,
        )
        assert third.status == "succeeded"
        assert third.failures == []
        assert third.evidence_refs == second.evidence_refs == first.evidence_refs
        assert len(ctx.mcp.calls) == 1
        historical = await service.graph.checkpointer.aget_tuple(before.config)
        assert historical is not None
        assert historical.checkpoint["channel_values"] == before.checkpoint["channel_values"]
        assert historical.checkpoint["channel_values"]["failures"] == first.failures
    finally:
        await service.aclose()
