"""Real SDK with a local exporter; no Langfuse network or account dependency."""

import asyncio
from typing import Literal
from uuid import uuid4

from langfuse import Langfuse
from langgraph.config import get_config
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from app.core.config_models import Settings
from app.core.masking import mask, mask_otel_spans
from app.core.observability import GraphTraceCallback, Observability
from app.core.trace_export import SafeSpanExporter


def tracing(settings: Settings) -> tuple[Observability, InMemorySpanExporter]:
    service = Observability(settings)
    exporter = InMemorySpanExporter()
    service.client = Langfuse(
        public_key="pk-lf-test-" + uuid4().hex,
        secret_key="sk-lf-test-only",  # noqa: S106 -- synthetic offline credential.
        base_url="http://127.0.0.1:1",
        tracer_provider=TracerProvider(),
        span_exporter=SafeSpanExporter(exporter),
        environment="test",
        mask=lambda *, data, **kwargs: mask(data),
        mask_otel_spans=mask_otel_spans,
    )
    return service, exporter


class TraceState(BaseModel):
    """Minimal state independent of business graph inputs and services."""

    status: Literal["running", "succeeded"] = "running"


async def invoke_trace_graph(barrier: asyncio.Barrier) -> TraceState:
    """Exercise nested callbacks while two turns are both inside their child node."""

    async def prepare(state: TraceState) -> Command[str]:
        return Command(goto="answer_data")

    async def select_schema(state: TraceState) -> Command[str]:
        await barrier.wait()
        return Command(update={"status": "succeeded"}, goto=END)

    child = StateGraph(TraceState)
    child.add_node("select_schema", select_schema, destinations=(END,))
    child.add_edge(START, "select_schema")
    compiled_child = child.compile(name="trace-child")

    async def answer_data(state: TraceState) -> Command[str]:
        output = TraceState.model_validate(await compiled_child.ainvoke(state, get_config()))
        return Command(update=output.model_dump(), goto=END)

    parent = StateGraph(TraceState)
    parent.add_node("prepare", prepare, destinations=("answer_data",))
    parent.add_node("answer_data", answer_data, destinations=(END,))
    parent.add_edge(START, "prepare")
    compiled_parent = parent.compile(name="trace-parent")
    callback = GraphTraceCallback()
    try:
        return TraceState.model_validate(
            await compiled_parent.ainvoke(TraceState(), {"callbacks": [callback]})
        )
    finally:
        callback.close()
