"""Real PostgreSQL proves history, immutable snapshots and checkpoint recovery."""

import asyncio
import json
from dataclasses import replace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from sqlalchemy import text

from app.agents.data import summarize
from app.agents.data.graph import topology as data_topology
from app.agents.data.state import DataAgentInput, DataAgentOutput, DataAgentState
from app.agents.state import AgentState
from app.agents.summarize import package_result
from app.core.config_models import DatabaseSettings
from app.core.errors import (
    ConflictError,
    LlmStructuredOutputError,
    NotFoundError,
    SqlExecutionError,
)
from app.db.session import Database
from app.schemas.mcp import ColumnSpec, SqlErrorKind
from app.schemas.sanity import SanityFlag
from app.schemas.sql_correction import CorrectionStatus
from app.schemas.synthesis import RowCountReference
from app.services.conversations import ConversationService
from app.services.evidence import EvidenceService
from app.services.graph import GraphService
from tests.agents.correction_support import correction_context
from tests.agents.support import metric_intent, result, sql_candidate
from tests.answer_support import data_draft
from tests.factories import business_schema
from tests.fakes.chat_model import FakeChatModel
from tests.fakes.mcp_client import FakeMcpClient
from tests.integration.checkpoint_support import (
    admitted,
    checkpoint_setup,
    connected_context,
    graph_database,
)

pytestmark = pytest.mark.integration
__all__ = ["checkpoint_setup", "graph_database"]


async def test_conversation_resumes_after_restart(
    graph_database: tuple[Database, DatabaseSettings],
) -> None:
    database, settings = graph_database
    identity = await admitted(database)
    first = GraphService(settings)
    await first.start()
    ctx = connected_context(database, identity)
    try:
        output = await first.invoke(ctx)
        assert output.status == "succeeded"
    finally:
        await first.aclose()
    new_identity = await admitted(database, identity=identity)
    second = GraphService(settings)
    await second.start()
    new_ctx = connected_context(database, new_identity, followup=True)
    try:
        output2 = await second.invoke(new_ctx)
        assert output2.status == "succeeded"
        prepared = await new_ctx.conversations.prepare(new_identity)
        assert any("42 orders" in message.content for message in prepared.messages)
        assert prepared.prior_sql == [output.answer.sql]
        checkpoint = await second.graph.aget_state(
            {"configurable": {"thread_id": str(new_identity.turn_id)}}
        )
        assert checkpoint.values["turn_id"] == new_identity.turn_id
        assert AgentState.model_validate(checkpoint.values).failures == []
        saved = AgentState.model_validate(checkpoint.values)
        assert saved.rewritten is None
        assert saved.context is not None
        assert saved.route is not None
        assert saved.route.data_intent == "2026年8月GMV"
        assert output2.evidence_refs != output.evidence_refs
    finally:
        await second.aclose()


async def test_resume_same_turn_reuses_evidence(
    graph_database: tuple[Database, DatabaseSettings],
) -> None:
    database, settings = graph_database
    identity = await admitted(database)
    ctx = connected_context(database, identity)
    ctx = replace(
        ctx,
        llm=FakeChatModel(
            [
                metric_intent(),
                sql_candidate(),
                LlmStructuredOutputError(),
            ]
        ),
    )
    graph = GraphService(settings)
    await graph.start()
    try:
        failed = await graph.invoke(ctx)
        assert failed.status == "failed"
    finally:
        await graph.aclose()
    snapshot = await ctx.evidence.find(identity)
    restarted = GraphService(settings)
    await restarted.start()
    ctx = replace(ctx, llm=FakeChatModel([data_draft(markdown="Recovered 42", confidence=1)]))
    try:
        output = await restarted.invoke(ctx, resume=True)
        assert output.status == "succeeded"
        assert output.failures == []
        assert output.evidence_refs.data_snapshot_id == snapshot.id
        assert len(ctx.mcp.calls) == 1
    finally:
        await restarted.aclose()


async def test_immutable_idempotent_and_owned_snapshot(
    graph_database: tuple[Database, DatabaseSettings],
) -> None:
    database, settings = graph_database
    identity = await admitted(database)
    ctx = connected_context(database, identity)
    graph = GraphService(settings)
    await graph.start()
    try:
        await graph.invoke(ctx)
        snapshot = await ctx.evidence.find(identity)
        assert (await ctx.evidence.commit(identity, snapshot.data)).id == snapshot.id
        changed = snapshot.data.model_copy(update={"sql": "SELECT 900"})
        with pytest.raises(ConflictError):
            await ctx.evidence.commit(identity, changed)
        foreign = identity.model_copy(update={"user_id": uuid4()})
        with pytest.raises(NotFoundError):
            await ctx.evidence.find(foreign)
        with pytest.raises(NotFoundError):
            await graph.invoke(replace(ctx, identity=foreign), resume=True)
        # Changing a mutable source object cannot change the stored snapshot.
        snapshot.data.rows[0][0] = 999
        reloaded = await ctx.evidence.find(identity)
        assert reloaded.data.rows == [[42]]
        assert len(ctx.mcp.calls) == 1
        with pytest.raises(ConflictError):
            await graph.invoke(ctx)
    finally:
        await graph.aclose()


async def test_checkpoint_runtime_not_serialized(
    graph_database: tuple[Database, DatabaseSettings],
) -> None:
    database, settings = graph_database
    identity = await admitted(database)
    graph = GraphService(settings)
    await graph.start()
    try:
        ctx = connected_context(database, identity)
        await graph.invoke(ctx)
        async with database.session() as session:
            rows = (
                await session.execute(
                    text(
                        "SELECT metadata::text, checkpoint::text FROM checkpoints WHERE thread_id=:tid"
                    ),
                    {"tid": str(identity.turn_id)},
                )
            ).all()
            blobs = (
                (
                    await session.execute(
                        text("SELECT blob FROM checkpoint_blobs WHERE thread_id=:tid"),
                        {"tid": str(identity.turn_id)},
                    )
                )
                .scalars()
                .all()
            )
            serialized = json.dumps([list(row) for row in rows]).encode() + b"".join(
                bytes(blob) for blob in blobs if blob
            )
            assert settings.app_password.get_secret_value().encode() not in serialized
            assert b"RuntimeContext" not in serialized
            assert b"test-key" not in serialized
            assert b"FakeChatModel" not in serialized
            role = (
                await session.execute(
                    text("SELECT has_schema_privilege(current_user,'public','CREATE')")
                )
            ).scalar()
            assert role is False
            assert (
                await session.execute(
                    text("SELECT has_table_privilege(current_user,'data_evidence','UPDATE')")
                )
            ).scalar() is False
    finally:
        await graph.aclose()


async def test_previous_failure_not_in_new_turn(
    graph_database: tuple[Database, DatabaseSettings],
) -> None:
    database, settings = graph_database
    identity = await admitted(database)
    graph = GraphService(settings)
    await graph.start()
    try:
        ctx = connected_context(database, identity)
        bad = replace(ctx, llm=FakeChatModel([LlmStructuredOutputError()]))
        assert (await graph.invoke(bad)).status == "failed"
        identity2 = await admitted(database, identity=identity)
        output = await graph.invoke(connected_context(database, identity2, followup=True))
        assert output.status == "succeeded"
        assert output.failures == []
    finally:
        await graph.aclose()


async def test_concurrent_snapshot_writes_reuse_one_id(
    graph_database: tuple[Database, DatabaseSettings],
) -> None:
    database, _ = graph_database
    identity = await admitted(database)
    service = EvidenceService(database)
    data = package_result(result(), [])
    snapshots = await asyncio.gather(service.commit(identity, data), service.commit(identity, data))
    assert snapshots[0].id == snapshots[1].id


async def test_sanity_flags_survive_committed_snapshot_reload(
    graph_database: tuple[Database, DatabaseSettings],
) -> None:
    database, _ = graph_database
    identity = await admitted(database)
    data = package_result(result(), [])
    # Exercise the complete enum's JSONB compatibility, separately from detector tests.
    data.sanity_flags = list(SanityFlag)
    snapshot = await EvidenceService(database).commit(identity, data)
    data.sanity_flags.clear()
    restored = await EvidenceService(database).find(identity, snapshot.id)
    assert restored is not None
    assert restored.data.sanity_flags == list(SanityFlag)


@pytest.mark.parametrize(
    "version", ["phase2-v1", "phase2-v2", "phase4-v2", "phase4-v3", "phase4-v4", "phase4-v5"]
)
async def test_unknown_checkpoint_version_rejected(
    graph_database: tuple[Database, DatabaseSettings],
    version: str,
) -> None:
    database, settings = graph_database
    identity = await admitted(database)
    ctx = connected_context(database, identity)
    graph = GraphService(settings)
    await graph.start()
    try:
        await graph.invoke(ctx)
        await graph.graph.aupdate_state(
            {"configurable": {"thread_id": str(identity.turn_id)}},
            {"graph_version": version},
            as_node="format_answer",
        )
        with pytest.raises(ConflictError):
            await graph.invoke(ctx, resume=True)
        assert len(ctx.mcp.calls) == 1
    finally:
        await graph.aclose()


async def test_concurrent_graph_invocation_rejected(
    graph_database: tuple[Database, DatabaseSettings],
) -> None:
    database, settings = graph_database
    identity = await admitted(database)
    ctx = connected_context(database, identity)
    graph = GraphService(settings)
    await graph.start()
    try:
        async with graph._guard(ctx):
            with pytest.raises(ConflictError):
                await graph.invoke(ctx)
    finally:
        await graph.aclose()


async def test_new_turn_has_new_checkpoint(
    graph_database: tuple[Database, DatabaseSettings],
) -> None:
    database, settings = graph_database
    first_identity = await admitted(database)
    graph = GraphService(settings)
    await graph.start()
    try:
        await graph.invoke(connected_context(database, first_identity))
        second_identity = await admitted(database, identity=first_identity)
        config = {"configurable": {"thread_id": str(second_identity.turn_id)}}
        assert not (await graph.graph.aget_state(config)).values
        await graph.invoke(connected_context(database, second_identity, followup=True))
        assert (await graph.graph.aget_state(config)).values
        conversation_config = {"configurable": {"thread_id": str(first_identity.conversation_id)}}
        assert not (await graph.graph.aget_state(conversation_config)).values
    finally:
        await graph.aclose()


async def test_restart_loads_application_history(
    graph_database: tuple[Database, DatabaseSettings],
) -> None:
    database, _ = graph_database
    first_identity = await admitted(database)
    second_identity = await admitted(database, identity=first_identity)
    # No graph ever ran: the recreated service must still find application history.
    prepared = await ConversationService(database).prepare(second_identity)
    assert [message.content for message in prepared.messages] == [
        "2026年8月GMV",
        "There were 42 orders.",
    ]
    assert prepared.question == "2026年8月GMV"


async def test_small_pool_reserves_checkpoint_capacity(
    graph_database: tuple[Database, DatabaseSettings],
) -> None:
    database, settings = graph_database
    identities = [await admitted(database), await admitted(database)]
    graph = GraphService(settings.model_copy(update={"pool_size": 2, "pool_timeout_s": 0.5}))
    await graph.start()
    try:
        outputs = await asyncio.gather(
            *(graph.invoke(connected_context(database, identity)) for identity in identities)
        )
        assert all(output.status == "succeeded" for output in outputs)
    finally:
        await graph.aclose()


async def test_parent_child_checkpoints_use_same_turn_and_isolated_namespace(
    graph_database: tuple[Database, DatabaseSettings],
) -> None:
    database, settings = graph_database
    identity = await admitted(database)
    graph = GraphService(settings)
    await graph.start()
    try:
        output = await graph.invoke(connected_context(database, identity))
        assert output.status == "succeeded"
        async with database.session() as session:
            namespaces = (
                (
                    await session.execute(
                        text("SELECT DISTINCT checkpoint_ns FROM checkpoints WHERE thread_id=:tid"),
                        {"tid": str(identity.turn_id)},
                    )
                )
                .scalars()
                .all()
            )
        assert "" in namespaces
        assert any(namespace.startswith("data_agent:") for namespace in namespaces)
    finally:
        await graph.aclose()


async def test_child_resume_retains_correction_budget_and_bindings(
    graph_database: tuple[Database, DatabaseSettings],
) -> None:
    database, settings = graph_database
    identity = await admitted(database)
    ctx = replace(
        correction_context([SqlExecutionError(SqlErrorKind.UNDEFINED_TABLE), result()]),
        identity=identity,
    )
    config = {"configurable": {"thread_id": str(identity.turn_id)}, "recursion_limit": 32}
    first = GraphService(settings)
    await first.start()
    try:
        child = data_topology().compile(
            checkpointer=first.graph.checkpointer, interrupt_after=["correct_sql"]
        )
        await child.ainvoke(DataAgentInput(question="2026年8月的GMV"), config, context=ctx)
        checkpoint = await child.aget_state(config)
        saved = DataAgentState.model_validate(checkpoint.values)
        assert saved.correction_count == 1
        assert saved.correction_status is CorrectionStatus.PENDING_VALIDATION
        assert saved.metric_bindings
    finally:
        await first.aclose()
    second = GraphService(settings)
    await second.start()
    try:
        child = data_topology().compile(checkpointer=second.graph.checkpointer)
        resumed_ctx = replace(ctx, llm=FakeChatModel([]))
        output = DataAgentOutput.model_validate(
            await child.ainvoke(None, config, context=resumed_ctx)
        )
        assert output.failure is None
        assert output.evidence.metric_bindings == saved.metric_bindings
        restored = DataAgentState.model_validate((await child.aget_state(config)).values)
        assert restored.correction_count == 1
        assert len(ctx.mcp.calls) == 2  # noqa: PLR2004 -- original and corrected execution.
        assert not resumed_ctx.llm.calls
    finally:
        await second.aclose()


async def test_completed_clarification_resumes_without_models(
    graph_database: tuple[Database, DatabaseSettings],
) -> None:
    database, settings = graph_database
    identity = await admitted(database)
    ctx = replace(
        connected_context(database, identity),
        llm=FakeChatModel([metric_intent().model_copy(update={"metric_keys": []})]),
    )
    first = GraphService(settings)
    await first.start()
    try:
        original = await first.invoke(ctx)
        assert original.clarification is not None
        assert original.failures == []
    finally:
        await first.aclose()
    second = GraphService(settings)
    await second.start()
    try:
        resumed = await second.invoke(replace(ctx, llm=FakeChatModel([])), resume=True)
        assert resumed.clarification == original.clarification
        assert resumed.answer == original.answer
        assert resumed.answer.abstained
        assert not ctx.mcp.calls
    finally:
        await second.aclose()


async def test_exact_budgeted_generation_survives_database_and_restart(
    graph_database: tuple[Database, DatabaseSettings], monkeypatch: pytest.MonkeyPatch
) -> None:
    database, settings = graph_database
    identity = await admitted(database)
    payload = result([["private-long-description" * 200]])
    payload.columns = [ColumnSpec(name="description", type="text")]
    model = FakeChatModel([metric_intent(), sql_candidate(), LlmStructuredOutputError()])
    ctx = replace(
        connected_context(database, identity),
        mcp=FakeMcpClient([payload], schema_responses=[business_schema()]),
        llm=model,
    )
    graph = GraphService(settings)
    await graph.start()
    try:
        assert (await graph.invoke(ctx)).status == "failed"
    finally:
        await graph.aclose()
    snapshot = await ctx.evidence.find(identity)
    saved = snapshot.model_dump_json()
    stored_block = snapshot.data.generation_block
    assert json.loads(model.calls[-1].messages[-1].content)["generation_block"] == stored_block
    assert json.loads(stored_block)["sample_rows"] == []
    assert snapshot.data.rows == payload.rows
    assert not snapshot.data.result_summary.sample_truncated
    assert json.loads(stored_block)["sample_truncated"]

    payload.rows[0][0] = "source changed"
    forbidden = Mock(side_effect=AssertionError("Historical reads must not rebuild evidence"))
    monkeypatch.setattr(summarize, "summarize_result", forbidden)
    monkeypatch.setattr(summarize, "render_block", forbidden)
    recovered_model = FakeChatModel(
        [
            data_draft(
                markdown="Recovered answer", confidence=1, reference=RowCountReference(value=1)
            )
        ]
    )
    restarted = GraphService(settings)
    await restarted.start()
    try:
        output = await restarted.invoke(replace(ctx, llm=recovered_model), resume=True)
        assert output.status == "succeeded"
        assert output.evidence_refs.data_snapshot_id == snapshot.id
    finally:
        await restarted.aclose()
    assert (
        json.loads(recovered_model.calls[-1].messages[-1].content)["generation_block"]
        == stored_block
    )
    assert (await ctx.evidence.find(identity)).model_dump_json() == saved
    assert len(ctx.mcp.calls) == 1
    forbidden.assert_not_called()
