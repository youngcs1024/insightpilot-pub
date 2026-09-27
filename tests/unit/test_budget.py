"""Source caps and final-wire request guards, without provider or database access."""

import json
from time import monotonic

import httpx
import pytest
import respx
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from tenacity import wait_none

from app.core.budget import ContextBudget, ContextSlot, ModelContextLimits
from app.core.config_models import LLMSettings
from app.core.deadline import Deadline
from app.core.errors import ContextBudgetExceeded, LlmConfigurationError
from app.core.llm_config import ModelRole
from app.core.masking import REDACTED, mask, safe_attributes
from app.services.llm.budget import check_request, collect_budget_reports, prompt_budget
from app.services.llm.contracts import CompletionRequest, Message
from app.services.llm.registry import ModelRegistry, StructuredTier
from app.services.llm.service import LlmService
from app.services.schema_tokens import SchemaTokenCounter
from tests.factories import business_schema
from tests.llm_support import URL, report, response, service


class SmallOutput(BaseModel):
    count: int


def test_charge_within_limit() -> None:
    counter = SchemaTokenCounter()
    budget = ContextBudget(counter)
    budget.charge(ContextSlot.SUMMARY, "华东")
    budget.charge(ContextSlot.SUMMARY, "退款率")
    assert budget.used[ContextSlot.SUMMARY] == counter.count("华东") + counter.count("退款率")


def test_exceeding_limit_raises() -> None:
    counter = SchemaTokenCounter()
    budget = ContextBudget(counter, {ContextSlot.SUMMARY: counter.count("word")})
    budget.charge(ContextSlot.SUMMARY, "word")
    with pytest.raises(ContextBudgetExceeded) as failure:
        budget.charge(ContextSlot.SUMMARY, "word")
    assert failure.value.slot == "summary"
    assert failure.value.used > failure.value.limit
    assert not failure.value.retryable


def test_report_returns_all_slots() -> None:
    report_values = ContextBudget(SchemaTokenCounter()).report()
    assert {item.slot for item in report_values} == set(ContextSlot)
    assert all(item.used == 0 for item in report_values)


def test_complete_schema_fits_default_budget() -> None:
    schema = business_schema()
    budget = ContextBudget(SchemaTokenCounter())
    budget.charge(ContextSlot.SCHEMA, schema.rendered)
    assert len(schema.tables) == 8
    assert sum(len(table.columns) for table in schema.tables) == 52
    assert budget.used[ContextSlot.SCHEMA] <= 3072


def test_oversize_schema_raises_without_truncation() -> None:
    original = business_schema().rendered * 3
    budget = ContextBudget(SchemaTokenCounter())
    with pytest.raises(ContextBudgetExceeded, match="context"):
        budget.charge(ContextSlot.SCHEMA, original)
    assert original == business_schema().rendered * 3
    assert budget.used[ContextSlot.SCHEMA] == 0


def test_metrics_budget_is_separate() -> None:
    budget = ContextBudget(SchemaTokenCounter())
    budget.charge(ContextSlot.SCHEMA, business_schema().rendered)
    with pytest.raises(ContextBudgetExceeded) as failure:
        budget.charge(ContextSlot.METRICS, " metric" * 401)
    assert failure.value.slot == "metrics"


def request() -> CompletionRequest:
    return CompletionRequest(
        model="test", temperature=0, max_tokens=50,
        messages=[Message(role="user", content="private 中文")],
        tools=[{"type": "function", "function": {"name": "foo", "parameters": {"type": "object"}}}],
        model_limits=ModelContextLimits(context_window=10000, max_input_tokens=9000, max_output_tokens=1000),
    )


def test_total_under_model_context_window() -> None:
    value = request()
    report_value = check_request(value, SchemaTokenCounter())
    assert report_value.total_upper_bound == report_value.input_upper_bound + value.max_tokens
    assert report_value.wrapper_reserve == 1024 + 32 + 128
    assert report_value.total_upper_bound < report_value.context_window
    without_tools = check_request(value.model_copy(update={"tools": None}), SchemaTokenCounter())
    assert without_tools.input_upper_bound < report_value.input_upper_bound


@pytest.mark.parametrize("field", ["request_input", "request_output", "request_total"])
def test_total_includes_wrapper_and_output_reserve(field: str) -> None:
    value = request()
    assert value.model_limits is not None
    if field == "request_input":
        value.model_limits.max_input_tokens = 1
    elif field == "request_output":
        value.model_limits.max_output_tokens = 1
    else:
        value.model_limits.context_window = 1
    with pytest.raises(ContextBudgetExceeded) as failure:
        check_request(value, SchemaTokenCounter())
    assert failure.value.slot == field


def test_unknown_model_requires_explicit_window() -> None:
    config = LLMSettings(base_url="https://provider.invalid", api_key="test", model="unknown")
    with pytest.raises(LlmConfigurationError):
        ModelRegistry(config, [report("unknown", StructuredTier.NATIVE)])
    with pytest.raises(LlmConfigurationError):
        check_request(request().model_copy(update={"model_limits": None}), SchemaTokenCounter())


def test_role_exceptions_are_explicit_and_bounded() -> None:
    config = LLMSettings(base_url="https://provider.invalid", api_key="test", model="test")
    assert config.for_role(ModelRole.ROUTER).context_limits[ContextSlot.SYSTEM_PROMPT] == 2048
    assert config.for_role(ModelRole.SQL).context_limits[ContextSlot.METRICS] == 16384
    assert not config.for_role(ModelRole.SYNTHESIS).context_limits


def test_budget_report_mask_preserves_only_typed_diagnostics() -> None:
    report_value = check_request(request(), SchemaTokenCounter())
    data = report_value.model_dump(mode="json") | {"secret_content": "private"}
    masked = mask({"context_budget": data})
    assert "private" not in json.dumps(masked)
    assert masked == {"context_budget": report_value.model_dump(mode="json")}
    assert mask({"context_budget": {"input_upper_bound": "secret"}}) == {"context_budget": REDACTED}
    attributes = safe_attributes({"langfuse.observation.metadata.context_budget": json.dumps(data)})
    assert "private" not in json.dumps(attributes)
    assert "utf8_upper_bound" in json.dumps(attributes)


async def test_slot_overflow_prevents_http(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(URL).mock(return_value=response())
    async with service() as llm:
        with pytest.raises(ContextBudgetExceeded):
            await llm.generate_structured(
                ModelRole.SQL, [HumanMessage(content="question")], SmallOutput,
                deadline=Deadline(monotonic() + 10),
                budget=prompt_budget(schema=business_schema().rendered * 3),
            )
    assert not route.called


async def test_real_request_emits_report_without_sending_internal_budget(respx_mock: respx.MockRouter) -> None:
    route = respx_mock.post(URL).mock(return_value=response('{"count":7}'))
    async with service() as llm:
        with collect_budget_reports() as reports:
            await llm.generate_structured(
                ModelRole.SQL, [HumanMessage(content="question")], SmallOutput,
                deadline=Deadline(monotonic() + 10), budget=prompt_budget(schema=business_schema().rendered),
            )
    assert len(reports) == 1
    assert next(s.used for s in reports[0].slots if s.slot is ContextSlot.SCHEMA) > 0
    sent = json.loads(route.calls[0].request.content)
    assert not {"context", "slot_limits", "model_limits"} & sent.keys()
    print(reports[0].model_dump_json())


async def test_smaller_fallback_window_checked_before_dispatch(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.core.retry.wait_exponential", lambda **kwargs: wait_none())
    config = LLMSettings(
        base_url="https://provider.invalid/v1", api_key="test", model="primary",
        roles={"sql": {"fallback_models": ["backup"]}},
        model_contexts={
            "primary": {"context_window":1000000,"max_input_tokens":991808},
            "backup": {"context_window":1000,"max_input_tokens":1000,"max_output_tokens":1000},
        },
    )
    llm = LlmService(config, registry=ModelRegistry(config, [report(m, StructuredTier.NATIVE) for m in ("primary", "backup")]))
    route = respx_mock.post(URL).mock(return_value=httpx.Response(503))
    await llm.start()
    try:
        with pytest.raises(ContextBudgetExceeded):
            await llm.generate_structured(ModelRole.SQL, [HumanMessage(content="q")], SmallOutput, deadline=Deadline(monotonic()+10))
    finally:
        await llm.aclose()
    assert route.call_count == 3
    assert all(json.loads(call.request.content)["model"] == "primary" for call in route.calls)


async def test_repair_rechecks_expanded_request(respx_mock: respx.MockRouter) -> None:
    config = LLMSettings(
        base_url="https://provider.invalid/v1", api_key="test", model="primary",
        model_contexts={"primary": {"context_window":10000,"max_input_tokens":5000,"max_output_tokens":3000}},
    )
    llm = LlmService(config, registry=ModelRegistry(config, [report("primary", StructuredTier.PROMPTED)]))
    route = respx_mock.post(URL).mock(return_value=response("x" * 6000))
    await llm.start()
    try:
        with pytest.raises(ContextBudgetExceeded):
            await llm.generate_structured(ModelRole.SQL, [HumanMessage(content="q")], SmallOutput, deadline=Deadline(monotonic()+10))
    finally:
        await llm.aclose()
    assert route.call_count == 1
