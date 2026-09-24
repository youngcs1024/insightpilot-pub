"""Strict deterministic HTTP scripts, not live provider or retrieval quality tests."""

import json
from http import HTTPStatus

# ruff: noqa: PLR2004 -- exact vector dimensions and independent small-fixture oracle.
import httpx
import pytest

from app.agents.contracts import RouteDecision
from app.schemas.memory_extraction import MemoryExtraction
from app.schemas.model_runtime import EmbedResult, RerankResult
from data.seed.validation import validate
from tests.e2e.contracts import DATA_QUESTION
from tests.e2e.dataset import business
from tests.e2e.inference import application
from tests.memory_extraction_support import extraction_input


def router_request(question: str) -> dict[str, object]:
    return {
        "model": "synthetic",
        "temperature": 0,
        "max_tokens": 100,
        "messages": [
            {
                "role": "user",
                "content": '{"question":' + json.dumps(question) + ',"routing_context":{}}',
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"schema": RouteDecision.model_json_schema()},
        },
    }


async def test_script_requires_expected_context_and_rejects_exhaustion() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application()), base_url="http://test"
    ) as client:
        await client.post("/_e2e/script", json={"scenario": "data"})
        wrong = await client.post("/v1/chat/completions", json=router_request("wrong"))
        assert wrong.status_code == HTTPStatus.BAD_REQUEST
        await client.post("/_e2e/script", json={"scenario": "data"})
        response = await client.post("/v1/chat/completions", json=router_request(DATA_QUESTION))
        assert response.status_code == HTTPStatus.OK, response.text
        decision = RouteDecision.model_validate_json(
            response.json()["choices"][0]["message"]["content"]
        )
        assert decision.route == "data_only"
        duplicate = await client.post("/v1/chat/completions", json=router_request(DATA_QUESTION))
        assert duplicate.status_code == HTTPStatus.BAD_REQUEST
        status = (await client.get("/_e2e/status")).json()
        assert status["errors"]
        assert status["remaining"]["MetricIntent"] == 1


async def test_substitute_vectors_scores_and_document_query_counters() -> None:
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application()), base_url="http://test"
    ) as client:
        for mode in ("document", "query"):
            result = await client.post("/v1/embed", json={"texts": ["退款政策"], "mode": mode})
            response = EmbedResult.model_validate_json(result.content)
            assert len(response.dense[0]) == 1024
            assert response.sparse == [{42: 1.0}]
        result = await client.post(
            "/v1/rerank", json={"query": "退款", "passages": ["规则", "例外"]}
        )
        assert RerankResult.model_validate_json(result.content).scores == [0.9, 0.9]
        counters = (await client.get("/_e2e/status")).json()
        assert counters["embed_document"] == counters["embed_query"] == counters["rerank"] == 1
        invalid = await client.post("/v1/embed", json={"texts": [], "mode": "query"})
        assert invalid.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


def test_fixture_totals_are_independent_of_generated_sql() -> None:
    dataset = business()
    validate(dataset)
    customers = {row.customer_id: row for row in dataset.tables[3]}
    eligible = [
        row
        for row in dataset.tables[4]
        if row.paid_at.month == 8
        and row.status != "cancelled"
        and not customers[row.customer_id].is_test_account
    ]
    assert sum(row.gross_amount - row.discount_amount for row in eligible) == 600
    assert (
        sum(row.gross_amount - row.discount_amount for row in eligible if row.region_id == 2) == 300
    )


@pytest.mark.parametrize(
    ("scenario", "allowed"),
    [
        ("data", 1),
        ("knowledge", 1),
        ("both", 1),
        ("followup", 1),
        ("clarify", 0),
        ("chaos", 0),
        ("red_sql", 0),
        ("red_credential", 0),
        ("red_cross_user", 0),
        ("red_document", 1),
        ("red_citation", 1),
        ("red_false_policy", 1),
        ("red_causality", 1),
        ("red_widen", 2),
    ],
)
async def test_memory_calls_are_explicitly_scripted_and_bounded(scenario: str, allowed: int) -> None:
    request = router_request(DATA_QUESTION)
    request["messages"] = [{"role": "user", "content": extraction_input().model_dump_json()}]
    request["response_format"] = {
        "type": "json_schema",
        "json_schema": {"schema": MemoryExtraction.model_json_schema()},
    }
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application()), base_url="http://test"
    ) as client:
        await client.post("/_e2e/script", json={"scenario": scenario})
        for _ in range(allowed):
            response = await client.post("/v1/chat/completions", json=request)
            assert response.status_code == HTTPStatus.OK, response.text
            parsed = MemoryExtraction.model_validate_json(
                response.json()["choices"][0]["message"]["content"]
            )
            assert parsed.candidates == []
        excess = await client.post("/v1/chat/completions", json=request)
        assert excess.status_code == HTTPStatus.BAD_REQUEST
