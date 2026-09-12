"""Real async HTTP adapter with scripted provider responses; no paid API calls."""

import asyncio
import json
from time import monotonic
from unittest.mock import MagicMock

import httpx
import pytest
import respx
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, ConfigDict
from structlog.testing import capture_logs
from tenacity import wait_none

from app.core.config_models import LLMSettings
from app.core.deadline import Deadline
from app.core.errors import (
    DeadlineExceededError,
    LlmRequestError,
    LlmStructuredOutputError,
    LlmUnavailableError,
)
from app.services.llm.registry import ModelRegistry, ModelRole, StructuredTier
from app.services.llm.service import LlmService
from tests.llm_support import URL, report, response, service

ATTEMPTS = 3
EXPECTED_COUNT = 7
TWO_CALLS = 2
FULL_LADDER_CALLS = 4


class Region(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    count: int
    region: Region


async def generate(llm: LlmService, question: str = "count") -> Answer:
    return await llm.generate_structured(
        ModelRole.SQL,
        [HumanMessage(content=question)],
        Answer,
        deadline=Deadline(monotonic() + 10),
    )


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.core.retry.wait_exponential", lambda **kwargs: wait_none())


async def test_tier1_used_when_supported(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(URL).mock(return_value=response())
    async with service() as llm:
        assert (await generate(llm)).count == EXPECTED_COUNT
    sent = json.loads(route.calls[0].request.content)
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["schema"]["$defs"]["Region"]
    assert sent["enable_thinking"] is False
    assert "tools" not in sent
    assert route.call_count == 1


async def test_falls_back_to_tier2_when_tier1_rejected(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(URL).mock(
        side_effect=[
            httpx.Response(
                400, json={"error": {"code": "unsupported_parameter", "param": "response_format"}}
            ),
            response(tool_name="structured_response"),
        ]
    )
    async with service() as llm:
        assert (await generate(llm)).region.name == "华东"
    sent = json.loads(route.calls[1].request.content)
    assert sent["tool_choice"]["function"]["name"] == "structured_response"
    assert sent["parallel_tool_calls"] is False
    assert "response_format" not in sent


async def test_schema_invalid_http_200_drops_tier(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(URL).mock(
        side_effect=[response('{"count":"bad"}'), response(tool_name="structured_response")]
    )
    async with service() as llm:
        assert (await generate(llm)).count == EXPECTED_COUNT
    assert route.call_count == TWO_CALLS


async def test_tier3_repair_on_parse_failure(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(URL).mock(side_effect=[response("not JSON"), response()])
    async with service(StructuredTier.PROMPTED) as llm:
        assert (await generate(llm)).count == EXPECTED_COUNT
    messages = json.loads(route.calls[1].request.content)["messages"]
    repair = json.loads(messages[-1]["content"])
    assert repair["previous_output"] == "not JSON"
    assert "json_invalid" in repair["validation_errors"]
    assert "not instructions" in " ".join(messages[-2]["content"].split())
    assert route.call_count == TWO_CALLS


async def test_raises_typed_error_after_all_tiers(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(URL).mock(return_value=response("invalid"))
    async with service() as llm:
        with pytest.raises(LlmStructuredOutputError) as error:
            await generate(llm)
    assert error.value.code == "LLM_STRUCTURED_OUTPUT_FAILED"
    assert route.call_count == FULL_LADDER_CALLS  # native, tool, prompted, one repair


async def test_fallback_state_not_shared_between_requests(respx_mock: respx.MockRouter) -> None:
    failing_started, other_completed = asyncio.Event(), asyncio.Event()
    models: dict[str, list[str]] = {"limited": [], "healthy": [], "later": []}

    async def provider(request: httpx.Request) -> httpx.Response:
        sent = json.loads(request.content)
        question = next(
            message["content"] for message in sent["messages"] if message["role"] == "user"
        )
        models[question].append(sent["model"])
        if question == "limited" and sent["model"] == "primary":
            failing_started.set()
            await other_completed.wait()
            return httpx.Response(429)
        if question == "healthy":
            other_completed.set()
        return response()

    respx_mock.post(URL).mock(side_effect=provider)
    async with service(fallback=True) as llm:
        async with asyncio.TaskGroup() as group:
            one = group.create_task(generate(llm, "limited"))
            await failing_started.wait()
            two = group.create_task(generate(llm, "healthy"))
        assert one.result() == two.result()
        await generate(llm, "later")
    assert models == {
        "limited": ["primary", "primary", "primary", "backup"],
        "healthy": ["primary"],
        "later": ["primary"],
    }


async def test_deadline_stops_tier_escalation(respx_mock: respx.MockRouter) -> None:
    cancelled = asyncio.Event()
    attempts = []

    async def slow(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return response()

    respx_mock.post(URL).mock(side_effect=slow)
    async with service(fallback=True) as llm:
        with pytest.raises(DeadlineExceededError):
            await llm.generate_structured(
                ModelRole.SQL,
                [HumanMessage(content="count")],
                Answer,
                deadline=Deadline(monotonic() + 0.02),
            )
    assert cancelled.is_set()
    assert len(attempts) == 1


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_other_4xx_never_retry_or_fallback(respx_mock: respx.MockRouter, status: int) -> None:
    route = respx_mock.post(URL).mock(
        return_value=httpx.Response(
            status,
            json={
                "error": {
                    "code": "invalid_request",
                    "message": "unsupported response_format secret-token",
                },
            },
        )
    )
    async with service(fallback=True) as llm:
        with pytest.raises(LlmRequestError, match="could not accept"):
            await generate(llm)
    assert route.call_count == 1


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
async def test_temporary_failure_retries_then_stops(
    respx_mock: respx.MockRouter, status: int
) -> None:
    route = respx_mock.post(URL).mock(return_value=httpx.Response(status))
    async with service() as llm:
        with pytest.raises(LlmUnavailableError):
            await generate(llm)
    assert route.call_count == ATTEMPTS


@pytest.mark.parametrize("failure", [httpx.ConnectError, httpx.ReadTimeout])
async def test_transport_failure_is_typed_and_retried(
    respx_mock: respx.MockRouter, failure: type[httpx.TransportError]
) -> None:
    route = respx_mock.post(URL).mock(side_effect=failure("private provider diagnostic"))
    async with service() as llm:
        with pytest.raises(LlmUnavailableError):
            await generate(llm)
    assert route.call_count == ATTEMPTS


async def test_cancellation_propagates_without_repair(respx_mock: respx.MockRouter) -> None:
    entered = asyncio.Event()
    attempts = []

    async def slow(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        entered.set()
        await asyncio.Event().wait()
        return response()

    respx_mock.post(URL).mock(side_effect=slow)
    async with service() as llm, asyncio.TaskGroup() as group:
        task = group.create_task(generate(llm))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(attempts) == 1


@pytest.mark.parametrize("kind", ["name", "multiple", "truncated"])
async def test_invalid_tool_response_drops_to_prompted(
    respx_mock: respx.MockRouter, kind: str
) -> None:
    invalid = json.loads(
        response(tool_name="wrong" if kind == "name" else "structured_response").content
    )
    if kind == "multiple":
        invalid["choices"][0]["message"]["tool_calls"] *= 2
    if kind == "truncated":
        invalid["choices"][0]["finish_reason"] = "length"
    route = respx_mock.post(URL).mock(side_effect=[httpx.Response(200, json=invalid), response()])
    async with service(StructuredTier.TOOL) as llm:
        await generate(llm)
    assert "tools" not in json.loads(route.calls[1].request.content)
    assert route.call_count == TWO_CALLS


async def test_plain_generation_and_input_not_mutated(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(URL).mock(return_value=response("hello"))
    messages: list[BaseMessage] = [HumanMessage(content="question")]
    async with service() as llm:
        result = await llm.call(ModelRole.SYNTHESIS, messages, deadline=Deadline(monotonic() + 1))
    assert isinstance(result, AIMessage)
    assert result.content == "hello"
    assert len(messages) == 1
    assert messages[0].content == "question"


async def test_safe_metadata_and_trace_failure_is_nonfatal(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    span = MagicMock()
    monkeypatch.setattr("app.services.llm.transport.trace.get_current_span", lambda: span)
    respx_mock.post(URL).mock(return_value=response())
    async with service() as llm:
        with capture_logs() as logs:
            await generate(llm, "private-question")
        assert "private-question" not in json.dumps(logs)
        assert "华东" not in json.dumps(logs, ensure_ascii=False)
        assert "test-key-only" not in json.dumps(logs)
        span.set_attribute.assert_any_call("llm.structured_tier", 1)
        span.set_attribute.assert_any_call("llm.role", "sql")
        span.set_attribute.side_effect = RuntimeError("private-trace-diagnostic")
        with capture_logs() as failed_logs:
            assert (await generate(llm)).count == EXPECTED_COUNT
        assert "private-trace-diagnostic" not in json.dumps(failed_logs)


async def test_repair_errors_do_not_echo_invalid_values(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(URL).mock(
        side_effect=[response('{"count":"sensitive-value","region":{"name":"x"}}'), response()]
    )
    async with service(StructuredTier.PROMPTED) as llm:
        await generate(llm)
    repair = json.loads(json.loads(route.calls[1].request.content)["messages"][-1]["content"])
    assert "sensitive-value" not in repair["validation_errors"]


async def test_malformed_envelope_exhausts_with_typed_error(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(URL).mock(return_value=httpx.Response(200, text="not-json"))
    async with service() as llm:
        with pytest.raises(LlmStructuredOutputError):
            await generate(llm)
    assert route.call_count == FULL_LADDER_CALLS


async def test_expired_deadline_makes_no_http_call(respx_mock: respx.MockRouter) -> None:
    async with service() as llm:
        with pytest.raises(DeadlineExceededError):
            await llm.generate_structured(
                ModelRole.SQL,
                [HumanMessage(content="question")],
                Answer,
                deadline=Deadline(monotonic() - 1),
            )
    assert not respx_mock.calls


async def test_deadline_between_tiers_prevents_next_http(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = monotonic()
    monkeypatch.setattr("app.core.deadline.time.monotonic", lambda: clock)

    def provider(request: httpx.Request) -> httpx.Response:
        nonlocal clock
        clock += 20
        return response("invalid")

    route = respx_mock.post(URL).mock(side_effect=provider)
    async with service() as llm:
        with pytest.raises(DeadlineExceededError):
            await llm.generate_structured(
                ModelRole.SQL,
                [HumanMessage(content="question")],
                Answer,
                deadline=Deadline(clock + 10),
            )
    assert route.call_count == 1


async def test_successful_rate_limit_retry_does_not_change_model(
    respx_mock: respx.MockRouter,
) -> None:
    route = respx_mock.post(URL).mock(side_effect=[httpx.Response(429), response()])
    async with service(fallback=True) as llm:
        await generate(llm)
    assert [json.loads(call.request.content)["model"] for call in route.calls] == [
        "primary",
        "primary",
    ]


async def test_role_configuration_reaches_http(respx_mock: respx.MockRouter) -> None:
    config = LLMSettings(
        base_url="https://provider.invalid/v1",
        model="primary",
        api_key="synthetic-key",
        roles={"sql": {"temperature": 0.25, "max_tokens": 128, "timeout_s": 2}},
    )
    route = respx_mock.post(URL).mock(return_value=response())
    llm = LlmService(
        config, registry=ModelRegistry(config, [report("primary", StructuredTier.NATIVE)])
    )
    await llm.start()
    try:
        await generate(llm)
    finally:
        await llm.aclose()
    sent = json.loads(route.calls[0].request.content)
    assert (sent["temperature"], sent["max_tokens"]) == (0.25, 128)
    assert route.calls[0].request.extensions["timeout"] == {
        "connect": 2,
        "read": 2,
        "write": 2,
        "pool": 2,
    }


async def test_fallback_uses_its_own_capability_tier(respx_mock: respx.MockRouter) -> None:
    config = LLMSettings(
        base_url="https://provider.invalid/v1",
        model="primary",
        api_key="synthetic-key",
        roles={"sql": {"fallback_models": ["backup"]}},
    )
    registry = ModelRegistry(
        config, [report("primary", StructuredTier.NATIVE), report("backup", StructuredTier.TOOL)]
    )
    llm = LlmService(config, registry=registry)
    route = respx_mock.post(URL).mock(
        side_effect=[httpx.Response(503)] * ATTEMPTS + [response(tool_name="structured_response")]
    )
    await llm.start()
    try:
        await generate(llm)
    finally:
        await llm.aclose()
    sent = json.loads(route.calls[-1].request.content)
    assert sent["model"] == "backup"
    assert "response_format" not in sent
    assert sent["tool_choice"]["function"]["name"] == "structured_response"


async def test_native_schema_includes_provider_required_json_instruction(
    respx_mock: respx.MockRouter,
) -> None:
    """The live provider rejects native-schema calls whose messages omit JSON."""

    def provider(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if not any(
            "json" in (message.get("content") or "").lower() for message in body["messages"]
        ):
            return httpx.Response(
                400, json={"error": {"code": "invalid_parameter_error", "param": None}}
            )
        assert body["response_format"]["type"] == "json_schema"
        return response()

    route = respx_mock.post(URL).mock(side_effect=provider)
    async with service() as llm:
        assert (await generate(llm, "统计订单数量")).count == EXPECTED_COUNT
    assert route.call_count == 1


async def test_repair_identifies_nested_field_without_echoing_input(
    respx_mock: respx.MockRouter,
) -> None:
    """A generic type code cannot identify which nested value needs repair."""
    invalid = '{"count":7,"region":{"name":{"private":"do not log"}}}'
    route = respx_mock.post(URL).mock(side_effect=[response(invalid), response()])
    async with service(StructuredTier.PROMPTED) as llm:
        assert (await generate(llm)).region.name == "华东"
    messages = json.loads(route.calls[1].request.content)["messages"]
    repair = json.loads(messages[-1]["content"])
    assert json.loads(repair["validation_errors"]) == [
        {"type": "string_type", "loc": ["region", "name"]}
    ]
    assert "private" not in repair["validation_errors"]
    assert "do not log" not in repair["validation_errors"]
    assert route.call_count == TWO_CALLS
