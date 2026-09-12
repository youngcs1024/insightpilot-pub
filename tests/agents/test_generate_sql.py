"""Step 2.8 contracts use scripted models, with no live-model quality claim."""

import json
from dataclasses import replace
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime
from structlog.testing import capture_logs

from app.agents.contracts import SqlGeneratorOutput
from app.agents.data.nodes.generate_sql import (
    build_messages,
    clean_sql,
    generate_sql,
    physical_tables,
)
from app.agents.data.nodes.resolve_metrics import resolve_metrics
from app.agents.data.state import DataAgentInput, DataAgentState
from app.core.deadline import Deadline
from app.core.errors import DeadlineExceededError, LlmStructuredOutputError, SqlGenerationError
from app.core.llm_config import ModelRole
from app.schemas.metric_resolution import MetricIntent, MetricPatch, RegionScope
from app.services.metric_binding import build_binding
from scripts.dev_sql import demonstrate
from tests.agents.sql_support import context, state
from tests.fakes.chat_model import FakeChatModel
from tests.metric_resolution_support import definition, request, runtime, schema
from tests.schema_support import full_block


async def test_prompt_includes_metric_bindings(fake_llm: FakeChatModel) -> None:
    inputs = state()
    fake_llm.enqueue(SqlGeneratorOutput(thinking="count", sql="SELECT 42", tables_used=[]))
    await generate_sql(inputs, Runtime(context=context(fake_llm)))
    prompt = str(fake_llm.calls[0].messages[0].content)
    assert (
        inputs.metric_bindings[0].resolved_expression
        in json.loads(
            prompt.split("## Resolved metric bindings\n", maxsplit=1)[1].split(
                "\n\n## Metric examples", maxsplit=1
            )[0]
        )[0]["resolved_expression"]
    )
    assert "2026-08-01T00:00:00+08:00" in prompt
    assert "c.is_test_account = FALSE" in prompt
    assert "o.status <> 'cancelled'" in prompt
    assert fake_llm.calls[0].role is ModelRole.SQL
    assert fake_llm.calls[0].schema_name == "SqlGeneratorOutput"


def test_prompt_includes_date_context(fake_llm: FakeChatModel) -> None:
    prompt = str(build_messages(state(), context(fake_llm))[0].content)
    assert "2026-09-08 (Tuesday)" in prompt
    assert "Q3" in prompt
    assert "2026-W37" in prompt
    assert "Asia/Shanghai" in prompt


def test_prompt_includes_dialect(fake_llm: FakeChatModel) -> None:
    messages = build_messages(state(), context(fake_llm))
    prompt = str(messages[0].content)
    assert "PostgreSQL 16" in prompt
    blocks = [
        "PostgreSQL 16",
        "## Business date context",
        "## Selected schema",
        "## Resolved metric bindings",
        "## Metric examples",
        "## Rules",
    ]
    assert [prompt.index(block) for block in blocks] == sorted(
        prompt.index(block) for block in blocks
    )
    assert json.loads(str(messages[-1].content)) == {
        "question": state().question,
        "prior_queries_for_reference": [],
    }
    assert list(SqlGeneratorOutput.model_fields) == ["thinking", "sql", "tables_used", "notes"]


@pytest.mark.parametrize("fence", ["sql", "SQL", ""])
def test_markdown_fences_stripped(fence: str) -> None:
    assert clean_sql(f"  ```{fence}\nSELECT 42;\n```  ") == "SELECT 42;"


def test_whitespace_not_collapsed() -> None:
    query = "SELECT 'two  spaces',\n  'literal ``` text' -- keep  comment\nFROM biz.orders"
    assert clean_sql(f"```sql\n{query}\n```") == query


async def test_named_ai_message_added(fake_llm: FakeChatModel) -> None:
    inputs = state()
    inputs.messages = [AIMessage(content="previous", name="earlier")]
    before = inputs.model_dump()
    fake_llm.enqueue(
        SqlGeneratorOutput(thinking="brief", sql="```sql\nSELECT 42\n```", tables_used=[])
    )
    result = await generate_sql(inputs, Runtime(context=context(fake_llm)))
    assert result.update["messages"][-1].name == "sql_generator"
    assert result.update["messages"][-1].content == "SELECT 42"
    assert result.update["messages"][0].content == "previous"
    assert "assumptions" not in result.update
    assert inputs.model_dump() == before
    restored = DataAgentState.model_validate_json(
        DataAgentState.model_validate({**before, **result.update}).model_dump_json()
    )
    assert isinstance(restored.messages[-1], AIMessage)


@pytest.mark.parametrize("sql", ["", "  ", "```sql\n\n```", "SELECT 1; SELECT 2", "SELECT FROM"])
async def test_empty_sql_produces_typed_failure(fake_llm: FakeChatModel, sql: str) -> None:
    fake_llm.enqueue(SqlGeneratorOutput(thinking="brief", sql=sql, tables_used=[]))
    with pytest.raises(SqlGenerationError) as error:
        await generate_sql(state(), Runtime(context=context(fake_llm)))
    assert error.value.code == "SQL_GENERATION_FAILED"
    assert not error.value.retryable


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("WITH x AS (SELECT * FROM biz.orders) SELECT * FROM x", ["biz.orders"]),
        ("WITH orders AS (SELECT 1) SELECT * FROM biz.orders", ["biz.orders"]),
        (
            "SELECT * FROM BIZ.ORDERS o UNION ALL SELECT * FROM biz.customers",
            ["biz.customers", "biz.orders"],
        ),
        ('SELECT * FROM biz."Orders"', ['biz."Orders"']),
        ("SELECT 42", []),
    ],
)
def test_physical_tables(sql: str, expected: list[str]) -> None:
    assert physical_tables(sql) == expected


async def test_tables_mismatch_warns_and_trusts_parse(
    fake_llm: FakeChatModel,
) -> None:
    fake_llm.enqueue(
        SqlGeneratorOutput(
            thinking="brief", sql="SELECT * FROM biz.orders", tables_used=["invented"]
        )
    )
    with capture_logs() as logs:
        result = await generate_sql(state(), Runtime(context=context(fake_llm)))
    assert result.update["tables_used"] == ["biz.orders"]
    assert any(
        log["event"] == "sql_tables_mismatch" and log["log_level"] == "warning" for log in logs
    )


async def test_structured_failure_and_deadline_propagate(fake_llm: FakeChatModel) -> None:
    failure = LlmStructuredOutputError()
    fake_llm.enqueue(failure)
    with pytest.raises(LlmStructuredOutputError):
        await generate_sql(state(), Runtime(context=context(fake_llm)))
    with pytest.raises(DeadlineExceededError):
        await generate_sql(
            state(), Runtime(context=replace(context(fake_llm), deadline=Deadline(0)))
        )
    assert len(fake_llm.calls) == 1


@pytest.mark.parametrize(
    "field", ["schema_block", "schema_tables", "metric_bindings", "metric_examples"]
)
async def test_missing_context_does_not_call_model(fake_llm: FakeChatModel, field: str) -> None:
    inputs = state().model_copy(update={field: "" if field == "schema_block" else []})
    with pytest.raises(SqlGenerationError):
        await generate_sql(inputs, Runtime(context=context(fake_llm)))
    assert not fake_llm.calls


async def test_examples_pin_resolved_version(fake_llm: FakeChatModel) -> None:
    ctx = runtime(MetricIntent(metric_keys=["gmv"], period_expression="2026年8月"))
    command = await resolve_metrics(DataAgentState(question="GMV"), Runtime(context=ctx))
    inputs = DataAgentState.model_validate({**state().model_dump(), **command.update})
    assert (
        inputs.metric_examples[0].definition_version == inputs.metric_bindings[0].definition_version
    )
    inputs.metric_examples[0].definition_version += 1
    with pytest.raises(SqlGenerationError):
        build_messages(inputs, context(fake_llm))


def test_override_binding_is_authoritative(fake_llm: FakeChatModel) -> None:
    inputs = state()
    changed = build_binding(request(patch=MetricPatch(date_field="o.created_at")), schema())
    inputs.metric_bindings = [changed.binding]
    prompt = str(build_messages(inputs, context(fake_llm))[0].content)
    assert "o.created_at" in prompt
    assert "must never override the resolved bindings" in prompt
    assert DataAgentState(question="old").metric_examples == []


async def test_demo_explicit_period_and_region(fake_llm: FakeChatModel) -> None:
    intent = MetricIntent(metric_keys=["gmv"], period_expression="2026年8月", region_mentioned=True)
    ctx = replace(
        runtime(intent),
        llm=fake_llm,
        schema_catalog=Mock(
            render=AsyncMock(return_value=full_block()), snapshot=AsyncMock(return_value=schema())
        ),
    )
    region = RegionScope(region_ids=[1])
    binding = build_binding(request().model_copy(update={"region_scope": region}), schema()).binding
    fake_llm.enqueue(intent)
    fake_llm.enqueue(
        SqlGeneratorOutput(
            thinking="resolved",
            sql=binding.resolved_expression,
            tables_used=["biz.orders", "biz.customers"],
        )
    )
    result = await demonstrate(
        DataAgentInput(question="2026年8月华东地区的GMV是多少?", region_scope=region), ctx
    )
    assert "2026-08-01" in result.sql
    assert "+08:00" in result.sql
    assert "2026-09-01" in result.sql
    assert "o.status <> 'cancelled'" in result.sql
    assert "c.is_test_account = FALSE" in result.sql
    assert "o.region_id IN (1)" in result.sql
    assert result.prompt[-1] == json.dumps(
        {"question": "2026年8月华东地区的GMV是多少?", "prior_queries_for_reference": []},
        ensure_ascii=False,
    )
    assert full_block() in result.prompt[0]
    assert not ctx.mcp.calls


async def test_clarification_clears_example_snapshots(fake_llm: FakeChatModel) -> None:
    ctx = runtime(MetricIntent(metric_keys=[], period_expression="2026年8月"))
    command = await resolve_metrics(state(), Runtime(context=ctx))
    updated = DataAgentState.model_validate({**state().model_dump(), **command.update})
    assert updated.metric_examples == []
    assert updated.metric_bindings == []
    with pytest.raises(SqlGenerationError):
        build_messages(updated, context(fake_llm))


async def test_examples_detached_and_no_generation_catalog_reread(fake_llm: FakeChatModel) -> None:
    item = definition()
    catalog = Mock(
        list_active=AsyncMock(return_value=[item]), get_active=AsyncMock(return_value=item)
    )
    ctx = replace(
        runtime(MetricIntent(metric_keys=["gmv"], period_expression="2026年8月")), metrics=catalog
    )
    command = await resolve_metrics(state(), Runtime(context=ctx))
    inputs = DataAgentState.model_validate({**state().model_dump(), **command.update})
    snapshot = inputs.metric_examples[0].model_dump()
    item.examples[0].sql = "SELECT 123"
    assert inputs.metric_examples[0].model_dump() == snapshot
    fake_llm.enqueue(SqlGeneratorOutput(thinking="brief", sql="SELECT 42", tables_used=[]))
    await generate_sql(inputs, Runtime(context=replace(ctx, llm=fake_llm)))
    catalog.get_active.assert_awaited_once()
    catalog.list_active.assert_awaited_once()


@pytest.mark.parametrize(
    "sql", ["SELECT * FROM biz.orders x JOIN biz.customers x ON TRUE", "DELETE FROM biz.orders"]
)
def test_invalid_generation_candidate(sql: str) -> None:
    with pytest.raises(SqlGenerationError):
        physical_tables(sql)
