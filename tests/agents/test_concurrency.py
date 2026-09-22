"""Step 4.5 concurrency acceptance against the production parent topology."""

# ruff: noqa: PLR2004 -- specified timing, repetition and failure-count contracts.

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.types import Command
from structlog.testing import capture_logs

from app.agents.contracts import Route
from app.agents.failures import FailureKind
from app.agents.graph import topology
from app.agents.nodes.answer_data import answer_data_routed
from app.agents.nodes.answer_knowledge import answer_knowledge
from app.agents.state import AgentState, GraphInput
from app.clients.model_runtime import ModelRuntimeClient
from app.core.deadline import Deadline
from app.core.errors import McpUnavailableError, RetrievalUnavailableError
from app.core.observability import GraphTraceCallback, TraceMetadata
from app.schemas.model_runtime import EmbedMode
from app.services.graph import serializer
from app.services.knowledge_generation import KnowledgeGenerationService
from tests.agents.concurrency_support import ParallelProbe, sleeping_children
from tests.agents.knowledge_support import FakeRetrieval
from tests.agents.parent_support import parent_context
from tests.agents.projection_support import routed_state
from tests.agents.support import FakeEvidence, invoke
from tests.fakes.chat_model import FakeChatModel
from tests.fakes.mcp_client import FakeMcpClient
from tests.model_client_support import embedding_response, settings
from tests.observability_support import tracing


async def test_both_specialists_overlap_in_time(
    monkeypatch: pytest.MonkeyPatch,
    record_testsuite_property: Callable[..., None],
) -> None:
    ctx = parent_context(Route.BOTH)
    probe = ParallelProbe(ctx, delay=1)
    sleeping_children(ctx, probe, monkeypatch)
    started = time.monotonic()
    async with asyncio.timeout(5):
        output = await invoke(ctx)
    elapsed = time.monotonic() - started
    durations = {name: probe.ends[name] - probe.starts[name] for name in probe.starts}
    record_testsuite_property("both_parent_wall_seconds", elapsed)
    record_testsuite_property("both_specialist_service_seconds_sum", sum(durations.values()))
    assert output.status == "succeeded"
    assert elapsed < 1.6
    assert min(durations.values()) >= 1
    assert max(probe.starts.values()) < min(probe.ends.values())


@pytest.mark.parametrize("unexpected", [False, True])
async def test_data_failure_does_not_cancel_knowledge(unexpected: bool) -> None:
    error = RuntimeError("private-data-error") if unexpected else McpUnavailableError()
    ctx = parent_context(Route.BOTH, data_error=error)
    probe = ParallelProbe(ctx, first="data")
    with capture_logs() as logs:
        async with asyncio.timeout(5):
            output = await invoke(ctx)
    assert output.status == "degraded"
    assert output.answer.degraded_components == ["data"]
    assert ctx.evidence.knowledge is not None
    assert ctx.evidence.snapshot is None
    assert probe.finished["knowledge"].is_set()
    assert output.failures[0].kind is (
        FailureKind.NODE_OPERATION_FAILED if unexpected else FailureKind.MCP_UNAVAILABLE
    )
    assert "private-data-error" not in output.model_dump_json() + str(logs)


@pytest.mark.parametrize("unexpected", [False, True])
async def test_knowledge_failure_does_not_cancel_data(unexpected: bool) -> None:
    error = RuntimeError("private-knowledge-error") if unexpected else RetrievalUnavailableError()
    ctx = parent_context(Route.BOTH, knowledge_error=error)
    probe = ParallelProbe(ctx, first="knowledge")
    async with asyncio.timeout(5):
        output = await invoke(ctx)
    assert output.status == "degraded"
    assert output.answer.degraded_components == ["knowledge"]
    assert ctx.evidence.snapshot is not None
    assert ctx.evidence.knowledge is None
    assert probe.finished["data"].is_set()
    assert "private-knowledge-error" not in output.model_dump_json()


async def test_both_unexpected_failures_are_merged_without_answer() -> None:
    ctx = parent_context(
        Route.BOTH, data_error=RuntimeError("data secret"), knowledge_error=ValueError("kb secret")
    )
    ParallelProbe(ctx)
    async with asyncio.timeout(5):
        output = await invoke(ctx)
    assert output.status == "failed"
    assert output.answer is None
    assert len(output.failures) == 2
    assert all(f.kind is FailureKind.NODE_OPERATION_FAILED for f in output.failures)
    assert not ctx.evidence.committed
    assert "secret" not in output.model_dump_json()


async def test_unrelated_timeout_does_not_claim_request_deadline_expired() -> None:
    ctx = parent_context(Route.BOTH, data_error=TimeoutError("private-service-timeout"))
    ParallelProbe(ctx)
    async with asyncio.timeout(5):
        output = await invoke(ctx)
    assert output.status == "degraded"
    assert ctx.deadline.remaining() > 0
    assert output.failures[0].kind is FailureKind.NODE_OPERATION_FAILED
    assert "private-service-timeout" not in output.model_dump_json()


async def test_shared_deadline_respected_by_both() -> None:
    ctx = parent_context(Route.BOTH)
    probe = ParallelProbe(ctx, delay=0.02)
    async with asyncio.timeout(5):
        output = await invoke(ctx)
    assert output.status == "succeeded"
    assert probe.deadlines["data"] is probe.deadlines["knowledge"] is ctx.deadline
    assert ctx.deadline.remaining() < ctx.deadline.at - min(probe.starts.values())


async def test_shared_client_saturation_serializes_without_leaking_admission(
    record_testsuite_property: Callable[..., None],
) -> None:
    ctx = parent_context(Route.BOTH)
    probe = ParallelProbe(ctx)
    wait = probe.wait
    active = peak = 0
    intervals = []

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        started = time.monotonic()
        try:
            await asyncio.sleep(0.04)
            return embedding_response(request)
        finally:
            intervals.append((started, time.monotonic()))
            active -= 1

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = ModelRuntimeClient(settings(), http)

        async def limited(name: str, deadline: Deadline) -> None:
            await wait(name, deadline)
            await client.embed([name], EmbedMode.QUERY, deadline=deadline)

        probe.wait = limited
        started = time.monotonic()
        async with asyncio.timeout(5):
            output = await invoke(ctx)
            elapsed = time.monotonic() - started
            assert len(intervals) == 2
            assert intervals[0][1] <= intervals[1][0]
            assert elapsed >= sum(end - start for start, end in intervals)
            await client.embed(["after"], EmbedMode.QUERY, deadline=ctx.deadline)
        await client.aclose()
    record_testsuite_property("saturated_parent_wall_seconds", elapsed)
    assert output.status == "succeeded"
    assert peak == 1
    assert active == 0
    assert len(intervals) == 3


async def test_both_hit_same_deadline_and_stop_without_persisting() -> None:
    ctx = replace(parent_context(Route.BOTH), deadline=Deadline(time.monotonic() + 0.5))
    probe = ParallelProbe(ctx, delay=10)

    async def blocked_data(*args: object, **kwargs: object) -> None:
        try:
            await probe.wait("data", ctx.deadline)
        finally:
            probe.finish("data")

    async def blocked_knowledge(*args: object, **kwargs: object) -> None:
        try:
            await probe.wait("knowledge", ctx.deadline)
        finally:
            probe.finish("knowledge")

    ctx.evidence.find = blocked_data
    ctx.evidence.read_bundle = blocked_knowledge
    async with asyncio.timeout(3):
        output = await invoke(ctx)
    assert output.status == "failed"
    assert output.answer is None
    failures = {failure.node: failure.kind for failure in output.failures}
    assert failures["answer_data_routed"] is FailureKind.DEADLINE_EXCEEDED
    assert failures["answer_knowledge"] is FailureKind.DEADLINE_EXCEEDED
    assert all(event.is_set() for event in probe.finished.values())
    assert not ctx.evidence.committed


@pytest.mark.parametrize(
    ("wrapper", "lookup"), [(answer_data_routed, "find"), (answer_knowledge, "read_bundle")]
)
async def test_deadline_covers_snapshot_lookup(
    wrapper: Callable[..., Awaitable[Command[str]]], lookup: str
) -> None:
    ctx = replace(parent_context(Route.BOTH), deadline=Deadline(time.monotonic() + 0.1))
    stopped = asyncio.Event()

    async def blocked(*args: object, **kwargs: object) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    setattr(ctx.evidence, lookup, AsyncMock(side_effect=blocked))
    async with asyncio.timeout(2):
        output = await wrapper(routed_state(ctx), Runtime(context=ctx), {})
    assert stopped.is_set()
    assert output.update["failures"][0].kind is FailureKind.DEADLINE_EXCEEDED
    assert ctx.mcp.calls == ctx.retrieval.calls == []


async def test_external_cancellation_stops_both_without_partial_answer() -> None:
    ctx = parent_context(Route.BOTH)
    probe = ParallelProbe(ctx, delay=10)
    async with asyncio.timeout(5), asyncio.TaskGroup() as tasks:
        task = tasks.create_task(invoke(ctx))
        await probe.entered["data"].wait()
        await probe.entered["knowledge"].wait()
        task.cancel()
    assert task.cancelled()
    assert all(event.is_set() for event in probe.finished.values())
    assert not ctx.evidence.committed


async def test_assumptions_from_both_are_merged() -> None:
    ctx = parent_context(Route.BOTH)
    ctx.conversations.prepare.return_value.question = "请分析8月的经营情况"
    ParallelProbe(ctx)
    graph = topology().compile(checkpointer=InMemorySaver(serde=serializer()))
    config = {"configurable": {"thread_id": str(ctx.identity.turn_id)}}
    async with asyncio.timeout(5):
        updates = [
            item
            async for item in graph.astream(
                GraphInput(**ctx.identity.model_dump()), config, context=ctx, stream_mode="updates"
            )
        ]
    writes = {name: value for item in updates for name, value in item.items()}
    state = AgentState.model_validate((await graph.aget_state(config)).values)
    data = writes["data_agent"]["assumptions"]
    knowledge = writes["knowledge_agent"]["assumptions"]
    assert data
    assert knowledge
    assert state.assumptions == data + knowledge
    assert sum("persist_evidence" in item for item in updates) == 1


async def test_no_state_corruption_under_repeated_runs() -> None:
    template = parent_context(Route.BOTH)
    responses = list(template.llm._responses)
    mcp_results = list(template.mcp._responses)
    retrieval = template.retrieval.responses[0]
    graph = topology().compile(checkpointer=InMemorySaver(serde=serializer()))
    baseline = None
    for index in range(20):
        model = FakeChatModel(responses)
        ctx = replace(
            template,
            identity=template.identity.model_copy(update={"turn_id": uuid4()}),
            llm=model,
            mcp=FakeMcpClient(mcp_results, schema_responses=list(template.mcp._schema_responses)),
            retrieval=FakeRetrieval(retrieval),
            evidence=FakeEvidence(),
            knowledge_generation=KnowledgeGenerationService(model),
        )
        probe = ParallelProbe(ctx, first="data" if index % 2 else "knowledge")
        config = {"configurable": {"thread_id": str(ctx.identity.turn_id)}}
        async with asyncio.timeout(5):
            await graph.ainvoke(GraphInput(**ctx.identity.model_dump()), config, context=ctx)
        state = AgentState.model_validate((await graph.aget_state(config)).values)
        business = (
            state.data_evidence,
            state.knowledge_evidence,
            state.assumptions,
            state.failures,
            state.degraded_components,
            state.answer.markdown,
            state.status,
        )
        if baseline is None:
            baseline = business
        assert business == baseline
        assert state.status == "succeeded"
        assert state.failures == state.degraded_components == []
        assert all(event.is_set() for event in probe.finished.values())
        assert len(ctx.mcp.calls) == len(ctx.retrieval.calls) == 1


async def test_real_exported_spans_overlap_and_keep_parentage() -> None:
    ctx = parent_context(Route.BOTH)
    ParallelProbe(ctx, delay=0.02)
    telemetry, exporter = tracing(ctx.settings)
    try:
        with telemetry.turn(uuid4().hex, TraceMetadata()):
            callback = GraphTraceCallback()
            try:
                output = await invoke(ctx, [callback])
            finally:
                callback.close()
        telemetry.client.flush()
        spans = exporter.get_finished_spans()
        root = next(span for span in spans if span.name == "turn")
        data = next(span for span in spans if span.name == "data_agent")
        knowledge = next(span for span in spans if span.name == "knowledge_agent")
        assert output.status == "succeeded"
        assert data.parent.span_id == knowledge.parent.span_id == root.context.span_id
        assert data.context.trace_id == knowledge.context.trace_id == root.context.trace_id
        assert max(data.start_time, knowledge.start_time) < min(data.end_time, knowledge.end_time)
        for child, parent in (("execute_sql", data), ("retrieve", knowledge)):
            span = next(span for span in spans if span.name == child)
            assert span.parent.span_id == parent.context.span_id
        persistence = next(span for span in spans if span.name == "persist_evidence")
        assert persistence.start_time >= max(data.end_time, knowledge.end_time)
    finally:
        await telemetry.aclose()
