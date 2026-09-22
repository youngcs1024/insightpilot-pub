"""Strict deterministic HTTP scripts, not live provider or retrieval quality tests."""

import json

import httpx

from data.seed.validation import validate
from tests.e2e.dataset import business

from app.agents.contracts import RouteDecision
from app.schemas.model_runtime import EmbedResult, RerankResult
from tests.e2e.contracts import DATA_QUESTION
from tests.e2e.inference import application


def router_request(question: str) -> dict[str, object]:
    return {"model": "synthetic", "temperature": 0, "max_tokens": 100,
            "messages": [{"role": "user", "content":
                          '{"question":'+json.dumps(question)+',"routing_context":{}}'}],
            "response_format": {"type": "json_schema", "json_schema": {
                "schema": RouteDecision.model_json_schema()}}}


async def test_script_requires_expected_context_and_rejects_exhaustion() -> None:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application()), base_url="http://test") as client:
        await client.post("/_e2e/script", json={"scenario": "data"})
        wrong = await client.post("/v1/chat/completions", json=router_request("wrong"))
        assert wrong.status_code == 400
        await client.post("/_e2e/script", json={"scenario": "data"})
        response = await client.post("/v1/chat/completions", json=router_request(DATA_QUESTION))
        assert response.status_code == 200, response.text
        decision = RouteDecision.model_validate_json(response.json()["choices"][0]["message"]["content"])
        assert decision.route == "data_only"
        duplicate = await client.post("/v1/chat/completions", json=router_request(DATA_QUESTION))
        assert duplicate.status_code == 400
        status = (await client.get("/_e2e/status")).json()
        assert status["errors"] and status["remaining"]["MetricIntent"] == 1


async def test_substitute_vectors_scores_and_document_query_counters() -> None:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application()), base_url="http://test") as client:
        for mode in ("document", "query"):
            result = await client.post("/v1/embed", json={"texts": ["退款政策"], "mode": mode})
            response = EmbedResult.model_validate_json(result.content)
            assert len(response.dense[0]) == 1024
            assert response.sparse == [{42: 1.0}]
        result = await client.post("/v1/rerank", json={"query": "退款", "passages": ["规则", "例外"]})
        assert RerankResult.model_validate_json(result.content).scores == [0.9, 0.9]
        counters = (await client.get("/_e2e/status")).json()
        assert counters["embed_document"] == counters["embed_query"] == counters["rerank"] == 1
        invalid = await client.post("/v1/embed", json={"texts": [], "mode": "query"})
        assert invalid.status_code == 422


def test_fixture_totals_are_independent_of_generated_sql() -> None:
    dataset = business()
    validate(dataset)
    customers = {row.customer_id: row for row in dataset.tables[3]}
    eligible = [row for row in dataset.tables[4] if row.paid_at.month == 8
                and row.status != "cancelled" and not customers[row.customer_id].is_test_account]
    assert sum(row.gross_amount-row.discount_amount for row in eligible) == 600
    assert sum(row.gross_amount-row.discount_amount for row in eligible if row.region_id == 2) == 300
