"""Step 4.9: exact durable evidence, source independence and publication barriers."""

import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock
from uuid import uuid4

import psycopg
import pytest
from sqlalchemy import select

from app.agents.contracts import EvidenceBundle, Route
from app.agents.summarize import package_result
from app.clients.mcp_client import McpClient
from app.core.config_models import DatabaseSettings, Settings
from app.core.errors import ConflictError, DatabaseError
from app.db.models import Turn
from app.db.models.evidence import DataEvidenceRecord, KnowledgeEvidenceRecord
from app.db.session import Database
from app.repositories.evidence import EvidenceRepository, payload_fingerprint
from app.retrieval.config import EvidenceConfig, RetrievalConfig
from app.retrieval.evidence import package_evidence
from app.services.conversations import ConversationService
from app.services.evidence import EvidenceService
from app.services.graph import GraphService
from app.services.schema_tokens import SchemaTokenCounter
from tests.agents.knowledge_support import ranked
from tests.agents.parent_support import parent_context
from tests.agents.support import result
from tests.api.chat_support import Harness, chat, events
from tests.corpus_support import entry, write_inventory
from tests.database_support import DatabaseStack
from tests.evidence_support import committed_knowledge
from tests.integration.checkpoint_support import (
    admitted,
    checkpoint_setup,
    connected_context,
    graph_database,
)
from tests.integration.mcp_support import business_tables, mcp_endpoint, query
from tests.integration.mcp_support import client as mcp_client
from tests.retrieval_support import RetrievalHarness, deadline, harness
from tests.retrieval_support import query as retrieval_query

pytestmark = pytest.mark.integration
__all__ = [
    "business_tables",
    "chat",
    "checkpoint_setup",
    "graph_database",
    "harness",
    "mcp_client",
    "mcp_endpoint",
]


async def test_data_evidence_persisted_with_sql_and_bindings(
    graph_database: tuple[Database, DatabaseSettings], mcp_client: McpClient, business_tables: None
) -> None:
    database, settings = graph_database
    identity = await admitted(database)
    ctx = replace(connected_context(database, identity), mcp=mcp_client)
    graph = GraphService(settings)
    await graph.start()
    try:
        output = await graph.invoke(ctx)
    finally:
        await graph.aclose()
    assert output.status == "succeeded"
    bundle = await EvidenceService(database).read_bundle(identity)
    data = bundle.data.data
    assert data.metric_bindings
    assert data.sql == output.answer.sql
    assert data.rows == [[42]]
    assert data.assumptions == output.answer.assumptions
    assert data.result_summary.returned_row_count == data.row_count
    assert data.result_summary.statistics_scope == "returned_rows"
    assert data.mcp_call_id
    async with database.session() as session:
        row = await session.scalar(
            select(DataEvidenceRecord).where(DataEvidenceRecord.id == bundle.data.id)
        )
        assert row.payload == data.model_dump(mode="json")
        assert row.schema_version == data.schema_version
        assert row.created_at is not None
        assert row.content_sha256 == payload_fingerprint(row.payload)


async def test_knowledge_evidence_persisted_with_chunk_ids_and_scores(
    graph_database: tuple[Database, DatabaseSettings],
) -> None:
    database, _ = graph_database
    identity = await admitted(database)
    value = package_evidence(ranked(), EvidenceConfig(), SchemaTokenCounter())
    bundle = await EvidenceService(database).commit_bundle(identity, None, value)
    async with database.session() as session:
        row = await session.scalar(
            select(KnowledgeEvidenceRecord).where(KnowledgeEvidenceRecord.id == bundle.knowledge.id)
        )
        assert row.payload == value.model_dump(mode="json")
        assert row.schema_version == value.schema_version
        assert row.created_at is not None
        assert row.content_sha256 == payload_fingerprint(row.payload)
    restored = (await EvidenceService(database).read_bundle(identity)).knowledge.knowledge
    assert restored == value
    assert restored.chunks[0].scores.rerank is not None
    assert restored.chunks[0].original_text == value.chunks[0].original_text
    assert restored.generation_block == value.generation_block


async def test_answer_references_persisted_ids(chat: Harness) -> None:
    ctx = parent_context(Route.BOTH, settings=chat.app.state.settings)
    chat.app.state.llm = ctx.llm
    chat.app.state.retrieval = ctx.retrieval
    chat.app.state.knowledge_generation = ctx.knowledge_generation
    response = await chat.client.post(chat.url, json={"content": "请分析2026年8月的经营情况"})
    assert response.status_code == 200, response.text  # noqa: PLR2004
    body = response.json()
    evidence = (
        await chat.client.get(f"/api/v1/conversations/{chat.cid}/turns/{body['id']}/evidence")
    ).json()
    assert body["answer"]["evidence_refs"]["data_snapshot_id"] == evidence["data"]["id"]
    assert body["answer"]["evidence_refs"]["knowledge_snapshot_id"] == evidence["knowledge"]["id"]
    assert (await chat.stored())[-1]["answer"] == body["answer"]


async def test_evidence_readable_after_restart(
    graph_database: tuple[Database, DatabaseSettings],
) -> None:
    database, settings = graph_database
    identity = await admitted(database)
    before = await EvidenceService(database).commit_bundle(
        identity,
        package_result(result(), []),
        package_evidence(ranked(), EvidenceConfig(), SchemaTokenCounter()),
    )
    # Dispose the writer's entire connection pool before constructing another service.
    await database.aclose()
    restarted = Database(settings)
    restarted.start()
    try:
        assert await EvidenceService(restarted).read_bundle(identity) == before
    finally:
        await restarted.aclose()


async def test_evidence_endpoint_requires_ownership(chat: Harness) -> None:
    body = (await chat.client.post(chat.url, json={"content": "count"})).json()
    base = f"/api/v1/conversations/{chat.cid}/turns/{body['id']}/evidence"
    assert (await chat.client.get(base)).status_code == 200  # noqa: PLR2004
    assert (await chat.client.get(base.replace(str(chat.cid), str(uuid4())))).status_code == 404  # noqa: PLR2004
    original = chat.user.id
    chat.user.id = uuid4()
    assert (await chat.client.get(base)).status_code == 404  # noqa: PLR2004
    chat.user.id = original
    await chat.client.delete(f"/api/v1/conversations/{chat.cid}")
    assert (await chat.client.get(base)).status_code == 404  # noqa: PLR2004


async def test_retrieval_config_snapshot_present(
    graph_database: tuple[Database, DatabaseSettings],
) -> None:
    database, _ = graph_database
    identity = await admitted(database)
    retrieved = ranked()
    value = package_evidence(retrieved, EvidenceConfig(), SchemaTokenCounter())
    await EvidenceService(database).commit_bundle(identity, None, value)
    retrieved.retrieval_config.use_rerank = False
    restored = (await EvidenceService(database).read_bundle(identity)).knowledge.knowledge
    assert restored.retrieval_config == value.retrieval_config
    assert restored.retrieval_config.use_rerank
    assert restored.packaging_config == value.packaging_config


async def test_business_update_preserves_snapshot(
    graph_database: tuple[Database, DatabaseSettings],
    mcp_client: McpClient,
    business_tables: None,
    database_stack: DatabaseStack,
) -> None:
    database, _ = graph_database
    identity = await admitted(database)
    sql = "SELECT count(*) AS n FROM biz.regions"
    before = await query(mcp_client, sql)
    frozen = await EvidenceService(database).commit(identity, package_result(before, []))
    admin = await psycopg.AsyncConnection.connect(
        host="127.0.0.1",
        port=database_stack.settings.db_host_port,
        dbname="insightpilot_business",
        user="postgres",
        password=database_stack.settings.postgres_superuser_password.get_secret_value(),
        connect_timeout=5,
    )
    async with admin, admin.transaction():
        await admin.execute("SET LOCAL ROLE biz_owner")
        await admin.execute("SET LOCAL statement_timeout = '5s'")
        await admin.execute(
            "INSERT INTO biz.regions (region_id, name, name_en, effective_from) "
            "VALUES (49001, 'Audit region', 'Audit region', '2026-09-21T00:00:00Z')"
        )
    assert (await query(mcp_client, sql)).rows != before.rows
    assert await EvidenceService(database).find(identity) == frozen


@pytest.mark.storage
async def test_reingest_preserves_original_citation_text(
    harness: RetrievalHarness,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieved = await harness.pipeline(RetrievalConfig(record_arm_scores=True)).retrieve(
        retrieval_query(), deadline=deadline()
    )
    value = package_evidence(retrieved, EvidenceConfig(), SchemaTokenCounter())
    history = await committed_knowledge(harness.database, settings, value)
    for name in ("sku.md", "august.md"):
        path = harness.root / name
        path.write_text(path.read_text().replace("退款规则", "替换后的政策"))
    await harness.ingest()
    changed = await harness.pipeline().retrieve(retrieval_query(), deadline=deadline())
    assert changed.corpus_version != value.corpus_version
    await history.assert_unchanged(monkeypatch)


@pytest.mark.storage
async def test_source_removal_preserves_history(
    harness: RetrievalHarness,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieved = await harness.pipeline().retrieve(retrieval_query(), deadline=deadline())
    value = package_evidence(retrieved, EvidenceConfig(), SchemaTokenCounter())
    history = await committed_knowledge(harness.database, settings, value)
    write_inventory(harness.root, [entry("july.md")])
    await harness.ingest()
    for chunk in value.chunks:
        assert await harness.ingestion.identities(chunk.document_id) == []
    await history.assert_unchanged(monkeypatch)


@pytest.mark.storage
async def test_milvus_rebuild_does_not_affect_history(
    harness: RetrievalHarness,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    retrieved = await harness.pipeline().retrieve(retrieval_query(), deadline=deadline())
    value = package_evidence(retrieved, EvidenceConfig(), SchemaTokenCounter())
    history = await committed_knowledge(harness.database, settings, value)
    store = harness.ingestion
    async with asyncio.timeout(store.settings.startup_timeout_s):
        await store._client.drop_collection(store.settings.collection, timeout=None, retry_times=0)
    await store.ensure_collection()
    for chunk in value.chunks:
        assert await store.identities(chunk.document_id) == []
    await harness.ingest()
    for chunk in value.chunks:
        assert await store.identities(chunk.document_id)
    await history.assert_unchanged(monkeypatch)


async def test_snapshot_endpoint_makes_no_external_calls(
    chat: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = parent_context(Route.BOTH, settings=chat.app.state.settings)
    chat.app.state.llm = ctx.llm
    chat.app.state.retrieval = ctx.retrieval
    chat.app.state.knowledge_generation = ctx.knowledge_generation
    body = (await chat.client.post(chat.url, json={"content": "请分析2026年8月的经营情况"})).json()
    for service, method in (
        (chat.app.state.mcp, "call_tool"),
        (ctx.llm, "generate_structured"),
        (ctx.retrieval, "retrieve"),
        (ctx.knowledge_generation, "generate"),
    ):
        monkeypatch.setattr(service, method, AsyncMock(side_effect=AssertionError("external read")))
    response = await chat.client.get(
        f"/api/v1/conversations/{chat.cid}/turns/{body['id']}/evidence"
    )
    assert response.status_code == 200, response.text  # noqa: PLR2004
    assert response.json()["knowledge"]["knowledge"]["chunks"]


async def test_snapshot_retry_idempotent(graph_database: tuple[Database, DatabaseSettings]) -> None:
    database, _ = graph_database
    identity = await admitted(database)
    service = EvidenceService(database)
    data = package_result(result(), [])
    knowledge = package_evidence(ranked(), EvidenceConfig(), SchemaTokenCounter())
    bundles = await asyncio.gather(
        *(service.commit_bundle(identity, data, knowledge) for _ in range(3))
    )
    assert bundles[0] == bundles[1] == bundles[2]


async def test_conflicting_retry_payload_rejected(
    graph_database: tuple[Database, DatabaseSettings],
) -> None:
    database, _ = graph_database
    identity = await admitted(database)
    service = EvidenceService(database)
    knowledge = package_evidence(ranked(), EvidenceConfig(), SchemaTokenCounter())
    first = await service.commit_bundle(identity, None, knowledge)
    changed = knowledge.model_copy(update={"query_used": "changed"})
    with pytest.raises(ConflictError):
        await service.commit_bundle(identity, package_result(result(), []), changed)
    assert await service.read_bundle(identity) == first


@pytest.mark.parametrize("streamed", [False, True])
async def test_failed_evidence_write_never_publishes_answer(
    chat: Harness,
    monkeypatch: pytest.MonkeyPatch,
    streamed: bool,
) -> None:
    ctx = parent_context(Route.BOTH, settings=chat.app.state.settings)
    chat.app.state.llm = ctx.llm
    chat.app.state.retrieval = ctx.retrieval
    chat.app.state.knowledge_generation = ctx.knowledge_generation
    monkeypatch.setattr(
        EvidenceRepository, "insert_knowledge", AsyncMock(side_effect=DatabaseError())
    )
    response = await chat.client.post(
        chat.url + ("/stream" if streamed else ""), json={"content": "请分析2026年8月的经营情况"}
    )
    if streamed:
        assert all(name != "token" for name, _ in events(response))
        assert events(response)[-1][0] == "error"
    else:
        assert response.status_code == 500  # noqa: PLR2004
    stored = (await chat.stored())[-1]
    assert stored["status"] == "failed"
    assert stored["answer"] is None
    response = await chat.client.get(
        f"/api/v1/conversations/{chat.cid}/turns/{stored['id']}/evidence"
    )
    assert response.json()["data"] is response.json()["knowledge"] is None


@pytest.mark.parametrize("truncated", [False, True])
async def test_bounded_samples_preserve_all_returned_statistics(
    graph_database: tuple[Database, DatabaseSettings],
    truncated: bool,
) -> None:
    database, _ = graph_database
    identity = await admitted(database)
    values = [[value] for value in range(500)]
    returned = result(values).model_copy(update={"result_truncated": truncated})
    data = package_result(returned, ["Statistics describe returned rows only."])
    before = data.model_dump_json()
    await EvidenceService(database).commit(identity, data)
    stored = (await EvidenceService(database).find(identity)).data
    assert stored.model_dump_json() == before
    assert stored.result_summary.sample_truncated
    assert stored.result_summary.result_truncated is truncated
    assert stored.result_summary.columns[0].total == str(sum(range(500)))
    assert len(stored.rows) <= 200  # noqa: PLR2004 -- immutable audit sample contract.
    assert stored.row_count == len(values)


@pytest.mark.parametrize("route", [Route.DATA_ONLY, Route.KNOWLEDGE_ONLY, Route.BOTH])
async def test_recovery_after_commit_before_checkpoint_reference(
    graph_database: tuple[Database, DatabaseSettings],
    monkeypatch: pytest.MonkeyPatch,
    route: Route,
) -> None:
    database, settings = graph_database
    identity = await admitted(database)
    async with database.session() as session, session.begin():
        assistant = await session.get(Turn, identity.turn_id)
        user = await session.get(Turn, assistant.reply_to_turn_id)
        user.content = "请分析2026年8月的经营情况"
    base = parent_context(route)
    service = EvidenceService(database)
    ctx = replace(
        base, identity=identity, conversations=ConversationService(database), evidence=service
    )
    original = service.commit_bundle

    async def interrupted(*args: object, **kwargs: object) -> EvidenceBundle:
        await original(*args, **kwargs)
        raise DatabaseError()

    first = GraphService(settings)
    await first.start()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(service, "commit_bundle", interrupted)
            failed = await first.invoke(ctx)
        assert failed.status == "failed"
        assert failed.evidence_refs is None
        bundle = await service.read_bundle(identity)
        assert bundle.data or bundle.knowledge
    finally:
        await first.aclose()
    restarted = GraphService(settings)
    await restarted.start()
    resumed = replace(
        ctx,
        deadline=deadline(60),
        mcp=AsyncMock(call_tool=AsyncMock(side_effect=AssertionError("SQL repeated"))),
        retrieval=AsyncMock(retrieve=AsyncMock(side_effect=AssertionError("retrieval repeated"))),
    )
    try:
        output = await restarted.invoke(resumed, resume=True)
        assert output.status == "succeeded"
        assert output.evidence_refs == bundle.refs
        assert await service.read_bundle(identity) == bundle
        resumed.mcp.call_tool.assert_not_awaited()
        resumed.retrieval.retrieve.assert_not_awaited()
    finally:
        await restarted.aclose()


@pytest.mark.parametrize("kind", ["data", "knowledge"])
@pytest.mark.parametrize("damage", ["digest", "version"])
async def test_corrupt_snapshot_endpoint_fails_safely(
    chat: Harness,
    migration_stack: DatabaseStack,
    kind: str,
    damage: str,
) -> None:
    ctx = parent_context(Route.BOTH, settings=chat.app.state.settings)
    chat.app.state.llm = ctx.llm
    chat.app.state.retrieval = ctx.retrieval
    chat.app.state.knowledge_generation = ctx.knowledge_generation
    response = await chat.client.post(chat.url, json={"content": "请分析2026年8月的经营情况"})
    assert response.status_code == 200, response.text  # noqa: PLR2004
    body = response.json()
    table = "data_evidence" if kind == "data" else "knowledge_evidence"
    field = "content_sha256" if damage == "digest" else "schema_version"
    replacement = "0" * 64 if damage == "digest" else 99
    admin = await psycopg.AsyncConnection.connect(
        host="127.0.0.1",
        port=migration_stack.settings.db_host_port,
        dbname="insightpilot_app",
        user="postgres",
        password=migration_stack.settings.postgres_superuser_password.get_secret_value(),
        connect_timeout=5,
    )
    async with admin, admin.transaction():
        await admin.execute("SET LOCAL ROLE app_owner")
        await admin.execute("SET LOCAL statement_timeout = '5s'")
        await admin.execute(
            psycopg.sql.SQL("UPDATE {} SET {} = %s WHERE id = %s").format(
                psycopg.sql.Identifier(table), psycopg.sql.Identifier(field)
            ),
            (replacement, body["evidence_refs"][kind + "_snapshot_id"]),
        )
    response = await chat.client.get(
        f"/api/v1/conversations/{chat.cid}/turns/{body['id']}/evidence"
    )
    assert response.status_code == 500  # noqa: PLR2004
    assert response.json()["code"] == "EVIDENCE_INTEGRITY_ERROR"
    assert "SELECT" not in response.text
    assert "payload" not in response.text
