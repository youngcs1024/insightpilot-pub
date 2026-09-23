"""Current business graph tracing contracts, including nested nodes and service calls."""

import asyncio
import json
from dataclasses import replace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
import respx
from mcp.types import CallToolResult
from tenacity import wait_none

from app.agents.contracts import PreparedContext, Route, RouteDecision
from app.clients.mcp_client import McpClient
from app.core.errors import McpUnavailableError
from app.core.observability import GraphTraceCallback, TraceMetadata
from app.schemas.metric_tools import MetricFragment
from tests.agents.support import context, invoke, metric_intent, sql_candidate
from tests.answer_support import data_draft
from tests.factories import business_schema
from tests.factories import mcp_success as success
from tests.llm_support import URL, response
from tests.llm_support import service as llm_service
from tests.metric_tool_support import metric_args
from tests.observability_support import tracing


async def test_business_graph_span_topology_and_parentage() -> None:
    ctx = context()
    service, exporter = tracing(ctx.settings)

    async def run() -> str:
        trace_id = uuid4().hex
        with service.turn(trace_id, TraceMetadata(status="running")) as root:
            result = await invoke(context(), [GraphTraceCallback()])
            root.update(TraceMetadata(status=result.status))
        return trace_id

    try:
        ids = await asyncio.gather(run(), run())
        service.client.flush()
        spans = exporter.get_finished_spans()
        assert len(spans) == 32  # noqa: PLR2004 -- sixteen spans for each isolated turn.
        for trace_id in ids:
            group = [span for span in spans if format(span.context.trace_id, "032x") == trace_id]
            root = next(span for span in group if span.name == "turn")
            children = [span for span in group if span.name != "turn"]
            assert {span.name for span in children} == {
                "prepare_context",
                "route",
                "router",
                "finalize_context",
                "data_agent",
                "specialist_projection",
                "persist_evidence",
                "format_answer",
                "select_schema",
                "resolve_metrics",
                "generate_sql",
                "validate_sql",
                "execute_sql",
                "sanity_check",
                "package_evidence",
            }
            data = next(span for span in children if span.name == "data_agent")
            parent_nodes = {
                "prepare_context",
                "route",
                "router",
                "finalize_context",
                "data_agent",
                "persist_evidence",
                "format_answer",
            }
            for span in children:
                parent = (
                    next(item for item in children if item.name == "route")
                    if span.name == "router"
                    else root
                    if span.name in parent_nodes
                    else data
                )
                assert span.parent is not None
                assert span.parent.span_id == parent.context.span_id
                assert span.parent.trace_id == root.context.trace_id
        assert not any(span.events for span in spans)
    finally:
        await service.aclose()


async def test_typed_node_failure_has_failed_status() -> None:
    ctx = context(mcp_results=[McpUnavailableError()])
    service, exporter = tracing(ctx.settings)
    try:
        with service.turn(uuid4().hex, TraceMetadata()):
            await invoke(ctx, [GraphTraceCallback()])
        service.client.flush()
        node = next(span for span in exporter.get_finished_spans() if span.name == "data_agent")
        assert node.attributes["langfuse.observation.metadata.status"] == "failed"
    finally:
        await service.aclose()


async def test_real_service_spans_under_nodes_include_retries_and_fallback(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = context()
    service, exporter = tracing(ctx.settings)
    mcp = McpClient(ctx.settings.mcp)
    session = AsyncMock()
    normalized = metric_args().resolved_sql
    session.call_tool.side_effect = [
        CallToolResult(content=[], structured_content=business_schema().model_dump(mode="json")),
        CallToolResult(
            content=[],
            structured_content=MetricFragment(
                select_fragment="",
                from_fragment="",
                where_fragment="",
                group_by_fragment="",
                normalized_sql=normalized,
                normalized=False,
            ).model_dump(mode="json"),
        ),
        success(),
    ]
    monkeypatch.setattr(mcp, "_get_session", AsyncMock(return_value=session))
    monkeypatch.setattr("app.core.retry.wait_exponential", lambda **kwargs: wait_none())
    respx_mock.post(URL).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(503),
            httpx.Response(503),
            response(metric_intent().model_dump_json()),
            response(sql_candidate("SELECT 1").model_dump_json()),
            response(data_draft("private-answer", value=1).model_dump_json()),
        ]
    )
    try:
        async with llm_service(fallback=True) as llm:
            with service.turn(uuid4().hex, TraceMetadata()) as root:
                output = await invoke(replace(ctx, llm=llm, mcp=mcp), [GraphTraceCallback()])
                assert output.status == "succeeded"
                assert output.answer.claims[0].data_refs[0].value == 1
                root.update(TraceMetadata(status=output.status))
        service.client.flush()
        spans = exporter.get_finished_spans()
        metrics = next(span for span in spans if span.name == "resolve_metrics")
        generation = next(span for span in spans if span.name == "generate_sql")
        execution = next(span for span in spans if span.name == "execute_sql")
        synthesis = next(span for span in spans if span.name == "format_answer")
        generations = [span for span in spans if span.name == "llm_completion"]
        assert len(generations) == 6  # noqa: PLR2004 -- three failures, metric, SQL and synthesis.
        for span, parent in zip(generations, [metrics] * 4 + [generation, synthesis], strict=True):
            assert span.parent.span_id == parent.context.span_id
        assert all(
            "langfuse.observation.usage_details" not in s.attributes for s in generations[:3]
        )
        assert generations[3].attributes["langfuse.observation.model.name"] == "backup"
        tool = next(span for span in spans if span.name == "mcp_execute")
        metric_tool = next(span for span in spans if span.name == "mcp_metric")
        assert metric_tool.parent.span_id == metrics.context.span_id
        assert tool.parent.span_id == execution.context.span_id
        assert tool.attributes["langfuse.observation.metadata.mcp_call_id"] == "test-call"
        root_span = next(span for span in spans if span.name == "turn")
        assert root_span.attributes["langfuse.internal.is_app_root"] is True
        assert (
            root_span.attributes["langfuse.observation.metadata.degraded_components"] == '["llm"]'
        )
        serialized = json.dumps([dict(span.attributes) for span in spans])
        assert "private-" not in serialized
        assert "SELECT 1" not in serialized
    finally:
        await mcp.aclose()
        await service.aclose()


async def test_rewrite_trace_has_diagnostics_without_question_prose() -> None:
    ctx = context(
        responses=[
            RouteDecision(
                route=Route.CLARIFY,
                confidence=0.9,
                clarification_question="private-reference-marker",
            )
        ]
    )
    ctx = replace(
        ctx,
        conversations=AsyncMock(
            prepare=AsyncMock(
                return_value=PreparedContext(
                    question="private-original-marker",
                    summary="",
                    messages=[],
                    prior_sql=[],
                    has_prior_turns=True,
                )
            )
        ),
    )
    service, exporter = tracing(ctx.settings)
    try:
        with service.turn(uuid4().hex, TraceMetadata()):
            await invoke(ctx, [GraphTraceCallback()])
        service.client.flush()
        spans = exporter.get_finished_spans()
        rewrite = next(span for span in spans if span.name == "router")
        assert rewrite.attributes["langfuse.observation.metadata.route"] == "clarify"
        assert rewrite.attributes["langfuse.observation.metadata.decided_by"] == "llm"
        assert "private-" not in str([span.attributes for span in spans])
    finally:
        await service.aclose()
