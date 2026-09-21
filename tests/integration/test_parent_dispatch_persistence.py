"""Atomic knowledge/data snapshots and real PostgreSQL parent restart acceptance."""

from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import update

from app.agents.contracts import Route
from app.agents.knowledge.nodes.package_evidence import package_evidence
from app.agents.state import AgentState
from app.agents.summarize import package_result
from app.core.config_models import DatabaseSettings
from app.core.errors import ConflictError, LlmStructuredOutputError, NotFoundError
from app.db.models import Turn
from app.db.session import Database
from app.services.conversations import ConversationService
from app.services.evidence import EvidenceService
from app.services.graph import GraphService
from app.services.knowledge_generation import KnowledgeGenerationService
from tests.agents.knowledge_support import ranked
from tests.agents.parent_support import parent_context
from tests.agents.support import context, result
from tests.agents.synthesis_support import synthesis_draft
from tests.fakes.chat_model import FakeChatModel
from tests.integration.checkpoint_support import admitted, checkpoint_setup, graph_database

pytestmark = pytest.mark.integration
__all__ = ["checkpoint_setup", "graph_database"]


async def test_bundle_atomic_idempotent_owned_and_frozen(
    graph_database: tuple[Database, DatabaseSettings],
) -> None:
    database, _ = graph_database
    identity = await admitted(database)
    service = EvidenceService(database)
    data = package_result(result(), [])
    knowledge = package_evidence(ranked(), context())
    first = await service.commit_bundle(identity, data, knowledge)
    assert await service.commit_bundle(identity, data, knowledge) == first
    saved = first.model_dump_json()
    data.rows[0][0] = 999
    assert (await EvidenceService(database).read_bundle(identity)).model_dump_json() == saved
    with pytest.raises(ConflictError):
        await service.commit_bundle(identity, data, knowledge)
    other = await admitted(database)
    with pytest.raises(NotFoundError):
        await service.read_bundle(identity.model_copy(update={"user_id": other.user_id}))
    assert (await service.read_bundle(other)).refs.data_snapshot_id is None


async def test_conflicting_knowledge_rolls_back_new_data(
    graph_database: tuple[Database, DatabaseSettings],
) -> None:
    database, _ = graph_database
    identity = await admitted(database)
    service = EvidenceService(database)
    knowledge = package_evidence(ranked(), context())
    first = await service.commit_bundle(identity, None, knowledge)
    changed = knowledge.model_copy(update={"query_used": "different question"})
    with pytest.raises(ConflictError):
        await service.commit_bundle(identity, package_result(result(), []), changed)
    after = await service.read_bundle(identity)
    assert after.data is None
    assert after == first


async def test_empty_bundle_does_not_create_snapshots(
    graph_database: tuple[Database, DatabaseSettings],
) -> None:
    database, _ = graph_database
    identity = await admitted(database)
    service = EvidenceService(database)
    empty = await service.commit_bundle(identity, None, None)
    assert empty.data is empty.knowledge is None
    assert await service.read_bundle(identity) == empty


async def test_both_parent_restart_reuses_committed_sources_without_external_calls(
    graph_database: tuple[Database, DatabaseSettings],
) -> None:
    database, settings = graph_database
    identity = await admitted(database)
    async with database.session() as session, session.begin():
        assistant = await session.get(Turn, identity.turn_id)
        await session.execute(
            update(Turn)
            .where(Turn.id == assistant.reply_to_turn_id)
            .values(content="请分析2026年8月的经营情况")
        )
    base = parent_context(Route.BOTH)
    # Fail synthesis generation after both snapshots have committed.
    scripted = list(base.llm._responses)
    scripted[-1] = LlmStructuredOutputError()
    llm = FakeChatModel(scripted)
    ctx = replace(
        base,
        identity=identity,
        conversations=ConversationService(database),
        evidence=EvidenceService(database),
        llm=llm,
        knowledge_generation=KnowledgeGenerationService(llm),
    )
    first = GraphService(settings)
    await first.start()
    try:
        failed = await first.invoke(ctx)
        assert failed.status == "failed"
        bundle = await ctx.evidence.read_bundle(identity)
        assert bundle.data
        assert bundle.knowledge
        checkpoint = await first.graph.aget_state(
            {"configurable": {"thread_id": str(identity.turn_id)}}
        )
        assert AgentState.model_validate(checkpoint.values).evidence_refs == bundle.refs
    finally:
        await first.aclose()
    llm = FakeChatModel(
        [
            synthesis_draft(knowledge_id=bundle.knowledge.knowledge.chunks[0].chunk_id),
        ]
    )
    resumed = replace(
        ctx,
        llm=llm,
        knowledge_generation=KnowledgeGenerationService(llm),
        mcp=AsyncMock(),
        retrieval=AsyncMock(),
    )
    restarted = GraphService(settings)
    await restarted.start()
    try:
        output = await restarted.invoke(resumed, resume=True)
        assert output.status == "succeeded"
        assert output.evidence_refs == bundle.refs
        resumed.mcp.call_tool.assert_not_called()
        resumed.retrieval.retrieve.assert_not_called()
        assert await resumed.evidence.read_bundle(identity) == bundle
        # Replay and a fresh next turn cannot inherit failures or source snapshots.
        assert await restarted.invoke(resumed, resume=True) == output
        next_identity = await admitted(database, identity=identity)
        prepared = await ConversationService(database).prepare(next_identity)
        assert len(prepared.knowledge_history) == 1
        assert prepared.knowledge_history[0].turn_id == identity.turn_id
        assert prepared.knowledge_history[0].time_scope.kind == "ranges"
        assert "SELECT" not in prepared.knowledge_history[0].question
    finally:
        await restarted.aclose()
