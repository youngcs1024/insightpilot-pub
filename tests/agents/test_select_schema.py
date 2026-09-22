"""Full schema selection and safe measurement without network dependencies."""

# ruff: noqa: PLR2004 -- fixed schema and tokenizer acceptance values.

import json
from dataclasses import replace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import ValidationError

from app.agents.data.nodes.select_schema import select_schema
from app.agents.data.state import DataAgentState
from app.agents.runtime import RuntimeContext
from app.core.config_models import DataAgentSettings, SchemaStrategy
from app.core.errors import DeadlineExceededError, SchemaDriftError
from app.core.masking import mask, safe_attributes
from app.core.observability import GraphTraceCallback, TraceMetadata
from app.schemas.schema_catalog import BUSINESS_TABLES
from app.schemas.schema_tools import GetSchemaArgs
from app.services.schema_tokens import SchemaTokenCounter
from data.seed.schema_metadata_loader import load_catalog
from scripts.render_schema_catalog import measure
from tests.agents.support import context
from tests.factories import business_schema
from tests.observability_support import tracing
from tests.schema_support import AUTHORING, full_block


async def test_full_strategy_includes_all_tables() -> None:
    block = full_block()
    catalog = Mock(get_schema=AsyncMock(return_value=business_schema()))
    ctx = replace(context(), mcp=catalog)
    state = DataAgentState(question="八月 GMV")
    before = state.model_dump()
    result = await select_schema(state, Runtime(context=ctx))
    assert result.update == {"schema_block": block, "schema_tables": list(BUSINESS_TABLES)}
    assert state.model_dump() == before
    catalog.get_schema.assert_awaited_once_with(
        GetSchemaArgs(include_samples=True), deadline=ctx.deadline
    )
    assert len(load_catalog(AUTHORING).tables) == 8
    assert sum(len(table.columns) for table in load_catalog(AUTHORING).tables) == 52


@pytest.mark.parametrize("failure", [SchemaDriftError(), DeadlineExceededError()])
async def test_catalog_failure_propagates(failure: Exception) -> None:
    catalog = Mock(get_schema=AsyncMock(side_effect=failure))
    ctx = replace(context(), mcp=catalog)
    with pytest.raises(type(failure)) as caught:
        await select_schema(DataAgentState(question="GMV"), Runtime(context=ctx))
    assert caught.value is failure


def test_strategy_configuration() -> None:
    assert DataAgentSettings().schema_strategy is SchemaStrategy.FULL
    assert {item.value for item in SchemaStrategy} == {"full", "heuristic", "retrieved"}
    for value in ("heuristic", "retrieved", "unknown"):
        with pytest.raises(ValidationError):
            DataAgentSettings.model_validate({"schema_strategy": value})


def test_named_token_count_matches_cli() -> None:
    counter = SchemaTokenCounter()
    assert counter.count("hello world") == 2
    block = full_block() + "<|endoftext|>"
    tokens, upper = measure(block)
    assert counter.name == "cl100k_base"
    assert counter.count(block) == tokens
    assert tokens > 1500
    assert upper == len(block.encode("utf-8"))


async def test_schema_tokens_recorded_in_trace() -> None:
    block = full_block()
    ctx = replace(context(), mcp=Mock(get_schema=AsyncMock(return_value=business_schema())))
    graph = StateGraph(DataAgentState, context_schema=RuntimeContext)
    graph.add_node("select_schema", select_schema)
    graph.add_edge(START, "select_schema")
    graph.add_edge("select_schema", END)
    service, exporter = tracing(ctx.settings)
    try:
        with service.turn(uuid4().hex, TraceMetadata()):
            result = await graph.compile().ainvoke(
                DataAgentState(question="GMV"),
                context=ctx,
                config={"callbacks": [GraphTraceCallback()], "recursion_limit": 5},
            )
        service.client.flush()
        assert result["schema_block"] == block
        span = next(s for s in exporter.get_finished_spans() if s.name == "select_schema")
        exported = safe_attributes(span.attributes)
        metadata = {
            key.removeprefix("langfuse.observation.metadata."): value
            for key, value in exported.items()
            if key.startswith("langfuse.observation.metadata.")
        }
        assert metadata["schema_tokens"] == str(measure(block)[0])
        assert metadata["schema_strategy"] == "full"
        assert metadata["schema_tables"] == "8"
        assert metadata["schema_tokenizer"] == "cl100k_base"
        assert metadata["schema_utf8_bytes"] == str(len(block.encode("utf-8")))
        assert block not in json.dumps(exported)
    finally:
        await service.aclose()


def test_schema_metadata_masking() -> None:
    assert mask({"schema_tokenizer": "private"}) == {"schema_tokenizer": "<redacted>"}
    assert mask({"schema_tokenizer": "cl100k_base"}) == {"schema_tokenizer": "cl100k_base"}
    payload = {"schema_tokens": 2345, "schema_block": "private schema"}
    assert mask(payload) == {"schema_tokens": 2345, "schema_block": "<redacted>"}
    attributes = safe_attributes({"langfuse.observation.metadata": json.dumps(payload)})
    assert json.loads(attributes["langfuse.observation.metadata"]) == mask(payload)


async def test_telemetry_failure_does_not_change_result() -> None:
    ctx = context()
    service, _exporter = tracing(ctx.settings)
    try:
        with service.turn(uuid4().hex, TraceMetadata()) as root:
            root.span = Mock(update=Mock(side_effect=RuntimeError("export failed")))
            result = await select_schema(DataAgentState(question="GMV"), Runtime(context=ctx))
        assert result.update["schema_block"] == full_block()
    finally:
        await service.aclose()


async def test_node_records_returned_subset_without_reading_local_catalog() -> None:
    response = business_schema()
    response.tables = response.tables[:1]
    response.rendered = "server-rendered subset"
    ctx = replace(
        context(),
        mcp=Mock(get_schema=AsyncMock(return_value=response)),
        schema_catalog=Mock(render=AsyncMock(side_effect=AssertionError("local renderer called"))),
    )
    result = await select_schema(DataAgentState(question="GMV"), Runtime(context=ctx))
    assert result.update == {
        "schema_block": response.rendered,
        "schema_tables": [response.tables[0].table_name],
    }
    ctx.schema_catalog.render.assert_not_called()
