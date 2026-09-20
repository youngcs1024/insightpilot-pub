"""Real provider ladder and SDK export acceptance with deterministic transports."""

import asyncio
import json
from dataclasses import replace
from uuid import uuid4

import httpx
import pytest
import respx
from tenacity import wait_none

from app.agents.contracts import Route, RouterInput
from app.agents.nodes.router import route_question
from app.core.config_models import Settings
from app.core.errors import LlmRequestError, LlmUnavailableError
from app.core.llm_config import ModelRole
from app.core.masking import REDACTED, mask, safe_attributes
from app.core.observability import TraceMetadata
from app.services.llm.registry import StructuredTier
from app.services.llm.usage import collect_usage
from tests.llm_support import URL, response, service
from tests.observability_support import tracing
from tests.router_support import BOTH_QUESTION, decision, runtime

FULL_LADDER = 4
TOKENS_PER_RESPONSE = 20


async def test_original_route_preserved_in_trace_on_gate(settings: Settings) -> None:
    telemetry, exporter = tracing(settings)
    try:
        with telemetry.turn(uuid4().hex, TraceMetadata()):
            result = await route_question(
                RouterInput(question=BOTH_QUESTION),
                runtime([decision(confidence=0.4)]),
            )
        telemetry.client.flush()
        spans = exporter.get_finished_spans()
        attrs = next(span.attributes for span in spans if span.name == "router")
        assert result.route is Route.CLARIFY
        assert attrs["langfuse.observation.metadata.route"] == "clarify"
        assert attrs["langfuse.observation.metadata.original_route"] == "both"
        assert attrs["langfuse.observation.metadata.confidence"] == "0.4"
        assert attrs["langfuse.observation.metadata.decided_by"] == "llm"
        assert attrs["langfuse.observation.metadata.prefilter_hit"] == "false"
        assert BOTH_QUESTION not in json.dumps(
            [dict(span.attributes) for span in spans], ensure_ascii=False
        )
        assert decision().data_intent not in str(attrs)
    finally:
        await telemetry.aclose()


async def test_prefilter_trace_has_zero_tokens(settings: Settings) -> None:
    telemetry, exporter = tracing(settings)
    try:
        with telemetry.turn(uuid4().hex, TraceMetadata()):
            await route_question(RouterInput(question="8月GMV"), runtime())
        telemetry.client.flush()
        spans = exporter.get_finished_spans()
        attrs = next(span.attributes for span in spans if span.name == "router")
        assert attrs["langfuse.observation.metadata.router_tokens"] == "0"
        assert attrs["langfuse.observation.metadata.prefilter_hit"] == "true"
        assert not any(span.name == "llm_completion" for span in spans)
    finally:
        await telemetry.aclose()


async def test_unknown_route_exhausts_real_ladder_and_clarifies(
    respx_mock: respx.MockRouter,
) -> None:
    route = respx_mock.post(URL).mock(return_value=response('{"route":"invented","confidence":1}'))
    async with service() as llm:
        result = await route_question(
            RouterInput(question=BOTH_QUESTION), replace(runtime(), llm=llm)
        )
    assert result.route is Route.CLARIFY
    assert result.clarification_question
    assert route.call_count == FULL_LADDER


async def test_missing_intent_is_repaired_by_real_ladder(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(URL).mock(
        side_effect=[
            response('{"route":"both","confidence":0.9,"data_intent":"calculate"}'),
            response(decision().model_dump_json(), tool_name="structured_response"),
        ]
    )
    async with service() as llm:
        result = await route_question(
            RouterInput(question=BOTH_QUESTION), replace(runtime(), llm=llm)
        )
    assert result.route is Route.BOTH
    assert route.call_count == 2  # noqa: PLR2004 -- native failure then tool success.


async def test_router_tokens_sum_all_tiers_and_repair(
    respx_mock: respx.MockRouter,
    settings: Settings,
) -> None:
    respx_mock.post(URL).mock(
        side_effect=[
            response("invalid"),
            response("invalid", tool_name="structured_response"),
            response("invalid"),
            response(decision().model_dump_json()),
        ]
    )
    telemetry, exporter = tracing(settings)
    try:
        async with service() as llm:
            with telemetry.turn(uuid4().hex, TraceMetadata()):
                result = await route_question(
                    RouterInput(question=BOTH_QUESTION), replace(runtime(), llm=llm)
                )
        telemetry.client.flush()
        spans = exporter.get_finished_spans()
        router_span = next(span for span in spans if span.name == "router")
        assert result.route is Route.BOTH
        assert router_span.attributes["langfuse.observation.metadata.router_tokens"] == str(
            FULL_LADDER * TOKENS_PER_RESPONSE
        )
        generations = [span for span in spans if span.name == "llm_completion"]
        assert len(generations) == FULL_LADDER
        assert all(span.parent.span_id == router_span.context.span_id for span in generations)
        assert BOTH_QUESTION not in str([span.attributes for span in spans])
    finally:
        await telemetry.aclose()


@pytest.mark.parametrize("missing", ["usage", "retry"])
async def test_unreported_usage_is_unknown(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    monkeypatch.setattr("app.core.retry.wait_exponential", lambda **kwargs: wait_none())
    payload = json.loads(response(decision().model_dump_json()).content)
    payload.pop("usage")
    replies = (
        [httpx.Response(503), response(decision().model_dump_json())]
        if missing == "retry"
        else [httpx.Response(200, json=payload)]
    )
    respx_mock.post(URL).mock(side_effect=replies)
    async with service() as llm:
        with collect_usage() as usage:
            await llm.generate_structured(
                role=ModelRole.ROUTER,
                messages=[],
                schema=type(decision()),
                deadline=runtime().deadline,
            )
        assert usage.total is None


async def test_concurrent_router_token_totals_are_isolated(
    respx_mock: respx.MockRouter,
    settings: Settings,
) -> None:
    barrier = asyncio.Barrier(2)

    async def provider(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        question = json.loads(body["messages"][-1]["content"])["question"]
        await barrier.wait()
        payload = json.loads(response(decision().model_dump_json()).content)
        payload["usage"] = {"prompt_tokens": 10 if "甲" in question else 30, "completion_tokens": 1}
        return httpx.Response(200, json=payload)

    respx_mock.post(URL).mock(side_effect=provider)
    telemetry, exporter = tracing(settings)
    try:
        async with service() as llm:

            async def run(question: str) -> Route:
                with telemetry.turn(uuid4().hex, TraceMetadata()):
                    result = await route_question(
                        RouterInput(question=question), replace(runtime(), llm=llm)
                    )
                    return result.route

            assert await asyncio.gather(run("为什么甲上涨"), run("为什么乙上涨")) == [
                Route.BOTH,
                Route.BOTH,
            ]
        telemetry.client.flush()
        totals = sorted(
            int(span.attributes["langfuse.observation.metadata.router_tokens"])
            for span in exporter.get_finished_spans()
            if span.name == "router"
        )
        assert totals == [11, 31]
        with collect_usage() as later:
            assert later.attempts == 0
            assert later.total is None
    finally:
        await telemetry.aclose()


@pytest.mark.parametrize(("status", "error"), [(401, LlmRequestError), (503, LlmUnavailableError)])
async def test_real_service_failure_remains_typed(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    error: type[Exception],
) -> None:
    monkeypatch.setattr("app.core.retry.wait_exponential", lambda **kwargs: wait_none())
    route = respx_mock.post(URL).mock(return_value=httpx.Response(status))
    async with service(StructuredTier.PROMPTED) as llm:
        with pytest.raises(error):
            await route_question(RouterInput(question=BOTH_QUESTION), replace(runtime(), llm=llm))
    assert route.call_count == (1 if status == 401 else 3)  # noqa: PLR2004 -- typed retry policy.


def test_routing_metadata_masks_prose_and_invalid_diagnostic_types() -> None:
    safe = mask(
        {
            "route": "both",
            "original_route": "data_only",
            "confidence": 0.5,
            "router_tokens": 20,
            "prefilter_hit": False,
            "decided_by": "llm",
            "question": "private-question",
            "data_intent": "private-intent",
            "reasoning": "private-reason",
        }
    )
    assert safe["router_tokens"] == TOKENS_PER_RESPONSE
    assert "private-" not in str(safe)
    for name in (
        "route",
        "original_route",
        "confidence",
        "decided_by",
        "prefilter_hit",
        "router_tokens",
    ):
        assert mask({name: "private-value"})[name] == REDACTED
        key = "langfuse.observation.metadata." + name
        assert "private-" not in safe_attributes({key: "private-value"})[key]
    assert mask({"router_tokens": True, "confidence": float("nan")}) == {
        "router_tokens": REDACTED,
        "confidence": REDACTED,
    }
    assert mask({"route": "DATA_ONLY"}) == {"route": "DATA_ONLY"}
