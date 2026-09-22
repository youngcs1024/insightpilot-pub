"""Two committed turns exercise rewriting, region binding and persisted responses."""

import json
from uuid import UUID

import pytest
import sqlglot
from sqlglot import exp

from app.agents.contracts import Route, RouteDecision, SqlGeneratorOutput
from app.schemas.metric_resolution import MetricIntent, RegionReference
from app.services.schema_catalog import SchemaCatalogService
from tests.agents.support import metric_intent, sql_candidate
from tests.answer_support import data_draft
from tests.api.chat_support import Harness, chat, events
from tests.factories import business_schema, query_result
from tests.fakes.chat_model import FakeChatModel
from tests.fakes.mcp_client import FakeMcpClient
from tests.region_support import FOLLOWUP_QUESTION, region_result

pytestmark = pytest.mark.integration
__all__ = ["chat"]
OK = 200
AUGUST, SEPTEMBER = 8, 9
EXPECTED_MCP_CALLS = 4


def regional_sql(region_id: int) -> str:
    query = (
        "SELECT SUM(o.gross_amount - COALESCE(o.discount_amount, 0)) AS gmv "
        "FROM biz.orders o JOIN biz.customers c USING (customer_id) "
        "WHERE o.paid_at >= TIMESTAMPTZ '2026-08-01T00:00:00+08:00' "
        "AND o.paid_at < TIMESTAMPTZ '2026-09-01T00:00:00+08:00' "
        "AND o.paid_at IS NOT NULL AND o.status <> 'cancelled' "
        "AND c.is_test_account = FALSE"
    )

    return (
        sqlglot.parse_one(query, read="postgres")
        .where(exp.column("region_id", table="o").isin(region_id))
        .sql("postgres")
    )


async def test_two_turn_region_followup_generates_and_executes_correct_sql(chat: Harness) -> None:
    east, south = regional_sql(3), regional_sql(2)
    mcp = FakeMcpClient(
        [
            region_result(),
            query_result().model_copy(update={"executed_sql": east}),
            region_result(),
            query_result().model_copy(update={"executed_sql": south}),
        ],
        schema_responses=[
            business_schema(),
            *[business_schema() for _ in range(3)],
        ],
    )
    chat.app.state.mcp = mcp
    chat.app.state.schema_catalog = SchemaCatalogService(
        chat.database, mcp, chat.app.state.settings.schema_catalog
    )
    llm = FakeChatModel(
        [
            MetricIntent(
                metric_keys=["gmv"],
                period_expression="2026年8月",
                region_mentioned=True,
                region=RegionReference(names=["华东"]),
            ),
            SqlGeneratorOutput(
                thinking="GMV in East China", sql=east, tables_used=["biz.orders", "biz.customers"]
            ),
            data_draft(markdown="华东GMV为42。", confidence=1),
            RouteDecision(route=Route.DATA_ONLY, confidence=1, data_intent="2026年8月华南的GMV"),
            MetricIntent(
                metric_keys=["gmv"],
                period_expression="2026年8月",
                region_mentioned=True,
                region=RegionReference(names=["华南"]),
            ),
            SqlGeneratorOutput(
                thinking="GMV in South China",
                sql=south,
                tables_used=["biz.orders", "biz.customers"],
            ),
            data_draft(markdown="华南GMV为42。", confidence=1),
        ]
    )
    chat.app.state.llm = llm
    first = await chat.client.post(chat.url, json={"content": "2026年8月华东的GMV"})
    assert first.status_code == OK, first.text
    assert first.json()["answer"]["sql"] == east
    second = await chat.client.post(chat.url, json={"content": FOLLOWUP_QUESTION})
    assert second.status_code == OK, second.text
    assert second.json()["answer"]["sql"] == south
    assert len(mcp.calls) == EXPECTED_MCP_CALLS
    assert mcp.calls[-1].arguments.sql == south
    rewrite_input = json.loads(llm.calls[3].messages[1].content)
    assert rewrite_input["routing_context"]["recent_messages"][0]["content"] == "2026年8月华东的GMV"
    generated_input = json.loads(llm.calls[5].messages[1].content)
    assert generated_input == {
        "question": "2026年8月华南的GMV",
        "prior_queries_for_reference": [east],
    }
    evidence = await chat.app.state.evidence.find(chat.graph.last.identity)
    binding = evidence.data.metric_bindings[0]
    assert binding.region_scope.region_ids == [2]
    assert binding.period_start.month == AUGUST
    assert binding.period_end.month == SEPTEMBER
    assert binding.metric_key == "gmv"
    assert "o.region_id IN (2)" in binding.filters_applied
    assert first.json()["evidence_refs"] != second.json()["evidence_refs"]
    assert (await chat.stored())[-2]["content"] == FOLLOWUP_QUESTION


@pytest.mark.parametrize("streamed", [False, True])
async def test_reference_clarification_persists_and_replays(chat: Harness, streamed: bool) -> None:
    # Keep the antecedent inside the history budget, independently of the long SSE fixture.
    chat.app.state.llm = FakeChatModel(
        [
            RouteDecision(route=Route.DATA_ONLY, confidence=1, data_intent="2026年8月GMV"),
            metric_intent(),
            sql_candidate(),
            data_draft(markdown="订单总数为 42。"),
        ]
    )
    first = await chat.client.post(chat.url, json={"content": "count"})
    assert first.status_code == OK
    llm = FakeChatModel(
        [
            RouteDecision(
                route=Route.CLARIFY,
                confidence=1,
                clarification_question="请说明昨天所指的指标或政策。",
            )
        ]
    )
    chat.app.state.llm = llm
    calls = len(chat.app.state.mcp.calls)
    suffix = "/stream" if streamed else ""
    headers = {"Idempotency-Key": "unresolved-followup"}
    response = await chat.client.post(
        chat.url + suffix, json={"content": "帮我看看昨天那个"}, headers=headers
    )
    assert response.status_code == OK
    body = events(response)[-1][1] if streamed else response.json()
    assert body["clarification"]["kind"] == "reference_unresolved"
    assert body["status"] == "abstained"
    assert body["answer"]["abstained"]
    assert body["answer"]["claims"] == []
    assert body["evidence_refs"]["data_snapshot_id"] is None
    assert body["content"] == (await chat.stored())[-1]["content"]
    replay = await chat.client.post(
        chat.url + "/stream", json={"content": "帮我看看昨天那个"}, headers=headers
    )
    replayed = events(replay)[-1][1]
    assert replayed["clarification"] == body["clarification"]
    assert UUID(replayed["id"]) == UUID(body["id"])
    assert len(llm.calls) == 1
    routing_input = json.loads(llm.calls[0].messages[1].content)
    assert routing_input["routing_context"]["recent_messages"][0]["content"] == "count"
    assert len(chat.app.state.mcp.calls) == calls
