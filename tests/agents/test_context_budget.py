"""Production SQL/intent prompts with real request accounting and scripted HTTP."""

# ruff: noqa: PLR2004 -- measured contract budgets and fixed catalog size.

import json
from dataclasses import replace
from time import monotonic
from uuid import uuid4

import httpx
import pytest
from langgraph.runtime import Runtime

from app.agents.contracts import MetricExamplesSnapshot
from app.agents.data.nodes.generate_sql import generate_sql
from app.core.budget import ContextSlot
from app.core.deadline import Deadline
from app.retrieval.config import EvidenceConfig
from app.retrieval.evidence import package_evidence
from app.schemas.intent import MetricIntentInput
from app.services.knowledge_generation import KnowledgeGenerationService
from app.services.llm.budget import collect_budget_reports
from app.services.metric_binding import build_binding
from app.services.metric_intent import MetricIntentService
from app.services.schema_tokens import SchemaTokenCounter
from tests.agents.sql_support import context, state
from tests.factories import business_schema
from tests.fakes.chat_model import FakeChatModel
from tests.knowledge_support import draft, retrieval
from tests.llm_support import response, service
from tests.metric_resolution_support import Catalog, request, schema


async def test_complete_production_sql_request_budget(capsys: pytest.CaptureFixture[str]) -> None:
    inputs = state()
    inputs.schema_block = business_schema().rendered
    definitions = await Catalog().list_active()
    inputs.metric_bindings = [build_binding(request(d.key), schema()).binding for d in definitions]
    inputs.metric_examples = [
        MetricExamplesSnapshot(metric_key=d.key, definition_version=d.version, examples=d.examples)
        for d in definitions
    ]
    sent: list[httpx.Request] = []

    def provider(request_value: httpx.Request) -> httpx.Response:
        sent.append(request_value)
        return response('{"thinking":"constant","sql":"SELECT 42","tables_used":[],"notes":""}')

    async with service() as llm:
        await llm._client.aclose()
        llm._client = httpx.AsyncClient(
            base_url="https://provider.invalid/v1/", transport=httpx.MockTransport(provider)
        )
        runtime = replace(context(FakeChatModel([])), llm=llm, deadline=Deadline(monotonic() + 30))
        with collect_budget_reports() as reports:
            await generate_sql(inputs, Runtime(context=runtime))
    assert len(sent) == len(reports) == 1
    payload = json.loads(sent[0].content)
    assert inputs.schema_block in payload["messages"][-2]["content"]
    slots = {slot.slot: slot for slot in reports[0].slots}
    assert slots[ContextSlot.SCHEMA].used <= 3072
    assert 400 < slots[ContextSlot.METRICS].used <= 16384
    assert reports[0].total_upper_bound < reports[0].context_window
    with capsys.disabled():
        print("production_sql_budget=" + reports[0].model_dump_json())


async def test_full_metric_catalog_fits_explicit_sql_exception() -> None:
    async with service() as llm:
        await llm._client.aclose()
        llm._client = httpx.AsyncClient(
            base_url="https://provider.invalid/v1/",
            transport=httpx.MockTransport(
                lambda request: response('{"metric_keys":["gmv"],"period_expression":"2026年8月"}')
            ),
        )
        ctx = context(FakeChatModel([]))
        with collect_budget_reports() as reports:
            await MetricIntentService(llm, Catalog()).interpret(
                MetricIntentInput(question="2026年8月GMV"),
                deadline=Deadline(monotonic() + 30),
                now=ctx.now,
            )
    slots = {slot.slot: slot for slot in reports[0].slots}
    assert 400 < slots[ContextSlot.METRICS].used <= 16384
    assert 800 < slots[ContextSlot.SYSTEM_PROMPT].used <= 1536


async def test_knowledge_citation_repair_stays_within_default_system_budget() -> None:
    evidence = package_evidence(retrieval(), EvidenceConfig(), SchemaTokenCounter())
    outputs = iter(
        [
            draft(uuid4()).model_dump_json(),
            draft(evidence.chunks[0].chunk_id).model_dump_json(),
        ]
    )
    async with service() as llm:
        await llm._client.aclose()
        llm._client = httpx.AsyncClient(
            base_url="https://provider.invalid/v1/",
            transport=httpx.MockTransport(lambda request: response(next(outputs))),
        )
        with collect_budget_reports() as reports:
            generated = await KnowledgeGenerationService(llm).generate(
                evidence, deadline=Deadline(monotonic() + 30)
            )
    assert generated.attempts == 2
    assert len(reports) == 2
    for report_value in reports:
        system = next(s for s in report_value.slots if s.slot is ContextSlot.SYSTEM_PROMPT)
        assert system.used <= system.limit == 800
    assert generated.citations[0].chunk_id == evidence.chunks[0].chunk_id
