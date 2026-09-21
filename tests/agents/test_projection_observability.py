"""Projection diagnostics survive actual SDK export without exposing context."""

import asyncio
from collections.abc import Callable
from dataclasses import replace
from uuid import uuid4

import pytest
from langgraph.graph import END, START, StateGraph

from app.agents.data.state import DataAgentInput
from app.agents.knowledge.state import KnowledgeAgentInput
from app.agents.nodes.answer_knowledge import answer_knowledge_node
from app.agents.projections import to_data_input, to_knowledge_input
from app.agents.runtime import RuntimeContext
from app.agents.state import AgentState
from app.core.config_models import Settings
from app.core.masking import REDACTED, mask, safe_attributes
from app.core.observability import GraphTraceCallback, TraceMetadata
from tests.agents.knowledge_support import FakeRetrieval, ranked
from tests.agents.projection_support import routed_state
from tests.agents.support import context
from tests.observability_support import tracing


@pytest.mark.parametrize(
    ("project", "specialist"), [(to_data_input, "data"), (to_knowledge_input, "knowledge")]
)
async def test_projection_tokens_recorded(
    settings: Settings,
    project: Callable[..., DataAgentInput | KnowledgeAgentInput],
    specialist: str,
) -> None:
    ctx = context()
    state = routed_state(ctx)
    telemetry, exporter = tracing(settings)
    try:
        with telemetry.turn(uuid4().hex, TraceMetadata()):
            value = project(state, token_counter=ctx.schema_token_counter)
        telemetry.client.flush()
        span = next(s for s in exporter.get_finished_spans() if s.name == "specialist_projection")
        attrs = span.attributes
        assert attrs["langfuse.observation.metadata.projection_specialist"] == specialist
        assert attrs["langfuse.observation.metadata.projection_tokens"] == str(
            ctx.schema_token_counter.count(value.model_dump_json())
        )
        assert attrs["langfuse.observation.metadata.projection_tokenizer"] == "cl100k_base"
        exported = str([dict(s.attributes) for s in exporter.get_finished_spans()])
        for private in (state.question, state.context.summary, "SELECT 41", "o.paid_at", "营收"):
            assert private not in exported
    finally:
        await telemetry.aclose()


async def test_concurrent_projection_traces_are_isolated(settings: Settings) -> None:
    telemetry, exporter = tracing(settings)
    ctx = context()
    barrier = asyncio.Barrier(2)

    async def run(suffix: str) -> int:
        state = routed_state(ctx)
        state.question += suffix
        with telemetry.turn(uuid4().hex, TraceMetadata()):
            await barrier.wait()
            value = to_data_input(state, token_counter=ctx.schema_token_counter)
            return ctx.schema_token_counter.count(value.model_dump_json())

    try:
        expected = await asyncio.gather(run(""), run(" many extra words" * 40))
        telemetry.client.flush()
        spans = [s for s in exporter.get_finished_spans() if s.name == "specialist_projection"]
        assert sorted(
            int(s.attributes["langfuse.observation.metadata.projection_tokens"]) for s in spans
        ) == sorted(expected)
        assert len({s.context.trace_id for s in spans}) == len(expected)
    finally:
        await telemetry.aclose()


@pytest.mark.parametrize(
    ("key", "invalid"),
    [
        ("projection_specialist", "private-question"),
        ("projection_tokenizer", "private-memory"),
        ("projection_tokens", "private-sql"),
        ("projection_tokens", True),
        ("projection_tokens", -1),
        ("projection_tokens", 1.5),
    ],
)
def test_projection_metadata_rejects_invalid_scalars(key: str, invalid: object) -> None:
    assert mask({key: invalid}) == {key: REDACTED}
    attribute = "langfuse.observation.metadata." + key
    assert safe_attributes({attribute: invalid}) == {attribute: REDACTED}


def test_projection_metadata_masks_serialized_inputs() -> None:
    ctx = context()
    value = to_data_input(routed_state(ctx), token_counter=ctx.schema_token_counter)
    safe = mask({"projection_tokens": 42, "input": value.model_dump()})
    assert safe["projection_tokens"] == 42  # noqa: PLR2004 -- diagnostic sentinel.
    assert "SELECT" not in str(safe)
    assert "o.paid_at" not in str(safe)


@pytest.mark.parametrize("refused", [False, True])
async def test_compiled_knowledge_wrapper_span_parentage_and_refusal(refused: bool) -> None:
    ctx = replace(context(), retrieval=FakeRetrieval(ranked(0.1 if refused else 0.8)))
    telemetry, exporter = tracing(ctx.settings)
    graph = StateGraph(AgentState, context_schema=RuntimeContext)
    graph.add_node("answer_knowledge", answer_knowledge_node)
    graph.add_edge(START, "answer_knowledge")
    graph.add_edge("answer_knowledge", END)
    try:
        with telemetry.turn(uuid4().hex, TraceMetadata()):
            callback = GraphTraceCallback()
            try:
                await graph.compile().ainvoke(
                    routed_state(ctx), {"callbacks": [callback]}, context=ctx
                )
            finally:
                callback.close()
        telemetry.client.flush()
        spans = exporter.get_finished_spans()
        wrapper = next(s for s in spans if s.name == "answer_knowledge")
        projection = next(s for s in spans if s.name == "specialist_projection")
        assert projection.parent.span_id == wrapper.context.span_id
        assert wrapper.attributes["langfuse.observation.metadata.status"] == (
            "abstained" if refused else "succeeded"
        )
        assert "knowledge_abstention_reason" not in str([dict(s.attributes) for s in spans])
    finally:
        await telemetry.aclose()
