"""Historical evidence survives actual business-table changes without another MCP call."""

from dataclasses import replace
from unittest.mock import AsyncMock

import psycopg
import pytest

from app.agents.contracts import AnswerDraft
from app.clients.mcp_client import McpClient
from app.core.config_models import DatabaseSettings
from app.db.session import Database
from app.services.graph import GraphService
from mcp_server.db import BusinessDatabase
from tests.agents.support import metric_intent, sql_candidate
from tests.database_support import DatabaseStack
from tests.fakes.chat_model import FakeChatModel
from tests.integration.checkpoint_support import (
    admitted,
    checkpoint_setup,
    connected_context,
    graph_database,
)
from tests.integration.mcp_support import business, business_tables, client, mcp_endpoint, query

pytestmark = pytest.mark.integration
# Register the existing isolated fixtures rather than inventing another deployment path.
__all__ = [
    "business",
    "business_tables",
    "checkpoint_setup",
    "client",
    "graph_database",
    "mcp_endpoint",
]


async def test_business_mutation_does_not_change_historical_evidence(
    graph_database: tuple[Database, DatabaseSettings],
    client: McpClient,
    business: BusinessDatabase,
    database_stack: DatabaseStack,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database, settings = graph_database
    identity = await admitted(database)
    ctx = replace(
        connected_context(database, identity),
        mcp=client,
        llm=FakeChatModel(
            [
                metric_intent(),
                sql_candidate("SELECT count(*) AS n FROM biz.regions"),
                AnswerDraft(markdown="Query completed", confidence=1),
            ]
        ),
    )
    graph = GraphService(settings)
    await graph.start()
    try:
        output = await graph.invoke(ctx)
        before = await ctx.evidence.find(identity)
        original_rows = (await query(client, "SELECT count(*) AS n FROM biz.regions")).rows
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
                "VALUES (27001, 'Evidence region', 'Evidence region', '2026-09-08T00:00:00Z')"
            )
        changed_rows = (await query(client, "SELECT count(*) AS n FROM biz.regions")).rows
        assert changed_rows != original_rows
        monkeypatch.setattr(
            client, "call_tool", AsyncMock(side_effect=AssertionError("history must not call MCP"))
        )
        after = await ctx.evidence.find(identity, output.evidence_refs.data_snapshot_id)
        assert after.model_dump() == before.model_dump()
        await graph.invoke(ctx, resume=True)
        client.call_tool.assert_not_awaited()
    finally:
        await graph.aclose()
