"""Resolved SQL input and runtime builders for generation and correction tests."""

from dataclasses import replace

from app.agents.contracts import MetricExamplesSnapshot
from app.agents.data.state import DataAgentState
from app.agents.runtime import RuntimeContext
from app.schemas.metric_resolution import MetricIntent
from app.schemas.schema_catalog import BUSINESS_TABLES
from app.services.metric_binding import build_binding
from tests.fakes.chat_model import FakeChatModel
from tests.metric_resolution_support import definition, request, runtime, schema


def state() -> DataAgentState:
    item = definition()
    binding = build_binding(request(), schema())
    return DataAgentState(
        question="2026年8月的GMV是多少?",
        schema_block="biz.orders; biz.customers",
        schema_tables=list(BUSINESS_TABLES),
        metric_bindings=[binding.binding],
        assumptions=binding.assumptions,
        metric_examples=[
            MetricExamplesSnapshot(
                metric_key=item.key, definition_version=item.version, examples=item.examples
            )
        ],
    )


def context(fake_llm: FakeChatModel) -> RuntimeContext:
    return replace(
        runtime(MetricIntent(metric_keys=["gmv"], period_expression="2026年8月")), llm=fake_llm
    )
