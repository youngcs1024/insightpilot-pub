"""Parent v2 round-trips and recovery using the real protected PostgreSQL service."""

from dataclasses import replace

import pytest

from app.agents.contracts import AnswerDraft
from app.agents.knowledge.nodes.package_evidence import package_evidence
from app.agents.state import AgentState
from app.core.config_models import DatabaseSettings
from app.core.errors import LlmStructuredOutputError
from app.db.session import Database
from app.services.graph import GraphService
from tests.agents.knowledge_support import ranked
from tests.agents.state_support import finalized, memory
from tests.agents.support import metric_intent, sql_candidate
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
        await first.graph.aupdate_state(config, {
            "context": finalized().model_copy(update={"memories": [memory()]}),
            "route": decision(),
            "knowledge_evidence": package_evidence(ranked(), ctx),
            "assumptions": ["old turn"], "degraded_components": ["old component"],
            "abstained": True,
        }, as_node="format_answer")
        saved = AgentState.model_validate((await first.graph.aget_state(config)).values)
    finally:
        await first.aclose()
    restarted = GraphService(settings)
    await restarted.start()
    try:
        restored = AgentState.model_validate((await restarted.graph.aget_state(config)).values)
        assert restored == saved
        assert restored.context.memories[0].content.term == "大促"
        assert restored.knowledge_evidence.chunks[0].original_text
        replay = await restarted.invoke(replace(ctx, llm=FakeChatModel([])), resume=True)
        assert replay == original
        assert len(ctx.mcp.calls) == 1
        next_identity = await admitted(database, identity=identity)
        next_ctx = connected_context(database, next_identity, followup=True)
        await restarted.invoke(next_ctx)
        fresh = AgentState.model_validate((await restarted.graph.aget_state(
            {"configurable": {"thread_id": str(next_identity.turn_id)}}
        )).values)
        assert fresh.context is fresh.route is fresh.knowledge_evidence is None
        assert fresh.assumptions == fresh.failures == fresh.degraded_components == []
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
        actual = await service.graph.aget_state({"configurable": {"thread_id": str(identity.turn_id)}})
        wrong = await service.graph.aget_state({"configurable": {"thread_id": str(identity.conversation_id)}})
        assert actual.values["turn_id"] == identity.turn_id
        assert actual.values["graph_version"] == "phase4-v1"
        assert not wrong.values
        assert actual.values["messages"][-1].content == actual.values["prepared"].question
    finally:
        await service.aclose()


async def test_same_turn_recovery_failure_then_success_resets_only_current_failures(
    graph_database: tuple[Database, DatabaseSettings],
) -> None:
    database, settings = graph_database
    identity = await admitted(database)
    ctx = replace(connected_context(database, identity), llm=FakeChatModel([
        metric_intent(), sql_candidate(), LlmStructuredOutputError(),
    ]))
    service = GraphService(settings)
    await service.start()
    try:
        first = await service.invoke(ctx)
        assert first.status == "failed"
        assert len(first.failures) == 1
        config = {"configurable": {"thread_id": str(identity.turn_id)}}
        before = await service.graph.aget_state(config)
        second = await service.invoke(
            replace(ctx, llm=FakeChatModel([LlmStructuredOutputError()])), resume=True
        )
        assert second.status == "failed"
        assert len(second.failures) == 1
        third = await service.invoke(replace(
            ctx, llm=FakeChatModel([AnswerDraft(markdown="Recovered", confidence=1)])
        ), resume=True)
        assert third.status == "succeeded"
        assert third.failures == []
        assert third.evidence_refs == second.evidence_refs == first.evidence_refs
        assert len(ctx.mcp.calls) == 1
        historical = await service.graph.aget_state(before.config)
        assert len(historical.values["failures"]) == 1
    finally:
        await service.aclose()
