"""Owned bounded application history and durable rewrite checkpoints."""

from dataclasses import replace

import pytest
from sqlalchemy import update

from app.agents.contracts import Route, RouteDecision
from app.agents.state import AgentState
from app.agents.summarize import package_result
from app.core.config_models import DatabaseSettings
from app.core.errors import NotFoundError
from app.db.models import Turn
from app.db.session import Database
from app.services.conversations import ConversationService
from app.services.evidence import EvidenceService
from app.services.graph import GraphService
from tests.agents.support import result
from tests.fakes.chat_model import FakeChatModel
from tests.integration.checkpoint_support import (
    admitted,
    checkpoint_setup,
    connected_context,
    graph_database,
)

pytestmark = pytest.mark.integration
__all__ = ["checkpoint_setup", "graph_database"]


async def test_owned_prior_sql_is_latest_three_before_current_turn(
    graph_database: tuple[Database, DatabaseSettings],
) -> None:
    database, _ = graph_database
    evidence = EvidenceService(database)
    identity = await admitted(database)
    for index in range(4):
        await evidence.commit(
            identity,
            package_result(result().model_copy(update={"executed_sql": f"SELECT {index}"}), []),
        )
        identity = await admitted(database, identity=identity)
    prepared = await ConversationService(database).prepare(identity)
    assert prepared.has_prior_turns
    assert prepared.prior_sql == ["SELECT 3", "SELECT 2", "SELECT 1"]
    other = await admitted(database)
    isolated = await ConversationService(database).prepare(other)
    assert not isolated.has_prior_turns
    assert not isolated.prior_sql
    with pytest.raises(NotFoundError):
        await ConversationService(database).prepare(
            identity.model_copy(update={"user_id": other.user_id})
        )


async def test_history_presence_survives_oversized_message_trimming(
    graph_database: tuple[Database, DatabaseSettings],
) -> None:
    database, _ = graph_database
    prior = await admitted(database)
    current = await admitted(database, identity=prior)
    async with database.session() as session, session.begin():
        await session.execute(
            update(Turn).where(Turn.id == prior.turn_id).values(content="中" * 2000)
        )
    prepared = await ConversationService(database).prepare(current)
    assert prepared.has_prior_turns
    assert prepared.messages == []


async def test_clarification_rewrite_survives_checkpoint_restart(
    graph_database: tuple[Database, DatabaseSettings],
) -> None:
    database, settings = graph_database
    prior = await admitted(database)
    identity = await admitted(database, identity=prior)
    rewritten = RouteDecision(
        route=Route.CLARIFY, confidence=1, clarification_question="请说明昨天所指的问题。"
    )
    async with database.session() as session, session.begin():
        row = await session.get(Turn, identity.turn_id)
        await session.execute(
            update(Turn).where(Turn.id == row.reply_to_turn_id).values(content="昨天那个")
        )
    ctx = replace(connected_context(database, identity), llm=FakeChatModel([rewritten]))
    graph = GraphService(settings)
    await graph.start()
    try:
        output = await graph.invoke(ctx)
        assert output.clarification.kind.value == "reference_unresolved"
    finally:
        await graph.aclose()
    restarted = GraphService(settings)
    await restarted.start()
    try:
        ctx = replace(ctx, llm=FakeChatModel([]))
        replay = await restarted.invoke(ctx, resume=True)
        assert replay.clarification == output.clarification
        checkpoint = await restarted.graph.aget_state(
            {"configurable": {"thread_id": str(identity.turn_id)}}
        )
        saved = AgentState.model_validate(checkpoint.values)
        assert saved.route == rewritten
        assert saved.prepared.question == "昨天那个"
        assert not ctx.llm.calls
    finally:
        await restarted.aclose()
