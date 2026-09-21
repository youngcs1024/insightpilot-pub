"""Projection diagnostics survive actual SDK export without exposing context."""

import asyncio
from collections.abc import Callable
from uuid import uuid4

import pytest

from app.agents.data.state import DataAgentInput
from app.agents.knowledge.state import KnowledgeAgentInput
from app.agents.projections import to_data_input, to_knowledge_input
from app.core.config_models import Settings
from app.core.masking import REDACTED, mask, safe_attributes
from app.core.observability import TraceMetadata
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
