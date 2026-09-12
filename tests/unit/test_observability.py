"""Validate real SDK exports, graph parentage and fail-closed export behavior."""

import asyncio
import json
import threading
from time import monotonic
from unittest.mock import Mock
from uuid import uuid4

import httpx
import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode

from app.core.config_models import ObservabilitySettings, Settings
from app.core.deadline import Deadline
from app.core.observability import Observability, TraceMetadata, model_role, observe
from app.services.llm.contracts import CompletionRequest, Message
from app.services.llm.transport import LlmTransport
from tests.observability_support import invoke_trace_graph, tracing


async def test_generation_records_provider_tokens_without_messages(settings: Settings) -> None:
    service, exporter = tracing(settings)
    response = {
        "choices": [
            {"message": {"role": "assistant", "content": "private-output"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 17, "completion_tokens": 3},
    }
    async with httpx.AsyncClient(
        base_url="https://provider.invalid/v1/",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=response)),
    ) as client:
        try:
            with service.turn(uuid4().hex, TraceMetadata()), model_role("sql"):
                await LlmTransport(client).complete(
                    CompletionRequest(
                        model="test-model",
                        messages=[Message(role="user", content="private-input")],
                        temperature=0,
                        max_tokens=100,
                    ),
                    deadline=Deadline(monotonic() + 30),
                    timeout_s=5,
                    tier=None,
                )
            service.client.flush()
            generation = next(
                span for span in exporter.get_finished_spans() if span.name == "llm_completion"
            )
            attrs = dict(generation.attributes)
            assert json.loads(attrs["langfuse.observation.usage_details"]) == {
                "input": 17,
                "output": 3,
            }
            assert "private-input" not in json.dumps(attrs)
            assert "private-output" not in json.dumps(attrs)
            assert attrs["langfuse.observation.model.name"] == "test-model"
            assert "langfuse.observation.cost_details" not in attrs
        finally:
            await service.aclose()


async def test_export_mask_removes_raw_serialized_attributes(settings: Settings) -> None:
    service, exporter = tracing(settings)
    try:
        # Deliberately bypass the creation-time projection to exercise the final gate.
        with service.client.start_as_current_observation(name="test") as span:
            span._otel_span.set_attribute("langfuse.observation.input", '"private-input"')
            span._otel_span.set_attribute(
                "langfuse.observation.metadata", '{"sql":"private-sql","row_count":42}'
            )
            span._otel_span.set_attribute("unexpected", "private-field")
            span._otel_span.add_event("private-event", {"rows": "private-row"})
            span._otel_span.set_status(Status(StatusCode.ERROR, "private-status"))
        service.client.flush()
        exported = exporter.get_finished_spans()[0]
        attrs = dict(exported.attributes)
        assert "private" not in json.dumps(attrs)
        assert json.loads(attrs["langfuse.observation.metadata"])["row_count"] == 42  # noqa: PLR2004
        assert not exported.instrumentation_scope.attributes
        assert not exported.events
        assert exported.status.description is None
        assert exported.resource.attributes == {"service.name": "insightpilot"}
    finally:
        await service.aclose()


async def test_export_hook_failure_drops_batch(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    service, exporter = tracing(settings)

    def broken(*args: object, **kwargs: object) -> None:
        raise RuntimeError("mask failed")

    monkeypatch.setattr("app.core.masking._json_attribute", broken)
    try:
        with service.turn(uuid4().hex, TraceMetadata(status="running")) as root:
            root.span._otel_span.set_attribute(
                "langfuse.observation.metadata", '{"status":"running"}'
            )
        service.client.flush()
        assert not exporter.get_finished_spans()
    finally:
        await service.aclose()


async def test_disabled_observations_and_start_failure_do_not_escape(settings: Settings) -> None:
    service = Observability(settings)
    await service.start()
    with service.turn(uuid4().hex, TraceMetadata()) as root, observe("test", TraceMetadata()):
        root.update(TraceMetadata(status="succeeded"))
    assert service.client is None
    await service.aclose()


async def test_initialization_failure_is_best_effort(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    settings.observability = settings.observability.model_copy(update={"langfuse_enabled": True})
    service = Observability(settings)
    monkeypatch.setattr(service, "_create_client", Mock(side_effect=RuntimeError("unavailable")))
    await service.start()
    assert service.client is None


async def test_shutdown_runs_off_loop_and_is_bounded(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    service, _ = tracing(settings)
    client = service.client
    loop_thread = threading.get_ident()
    entered = threading.Event()
    release = threading.Event()

    def blocked() -> None:
        assert threading.get_ident() != loop_thread
        entered.set()
        release.wait(timeout=1)

    service.settings.http.shutdown_timeout_s = 0.01
    monkeypatch.setattr(client, "shutdown", blocked)
    try:
        await service.aclose()
        assert entered.is_set()
        assert service.client is None
    finally:
        release.set()
        monkeypatch.undo()
        client.shutdown()


async def test_exporter_failure_does_not_escape_business_work(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    service, exporter = tracing(settings)
    monkeypatch.setattr(exporter, "export", Mock(side_effect=RuntimeError("export unavailable")))
    try:
        with service.turn(uuid4().hex, TraceMetadata()) as root:
            root.update(TraceMetadata(status="succeeded"))
        service.client.flush()
        assert not exporter.get_finished_spans()
    finally:
        await service.aclose()


async def test_application_client_uses_safe_exporter_without_auth_probe(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    settings.observability = ObservabilitySettings(
        langfuse_enabled=True,
        langfuse_base_url="https://trace.invalid",
        langfuse_public_key="pk-lf-" + uuid4().hex,
        langfuse_secret_key="sk-lf-synthetic",  # noqa: S106 -- synthetic SDK fixture.
    )
    exporter = InMemorySpanExporter()
    factory = Mock(return_value=exporter)
    monkeypatch.setattr("app.core.observability.OTLPSpanExporter", factory)
    probe = Mock(side_effect=RuntimeError("auth check must not run"))
    monkeypatch.setattr(Langfuse, "auth_check", probe)
    service = Observability(settings)
    try:
        await service.start()
        await service.start()
        assert factory.call_count == 1
        assert factory.call_args.kwargs["timeout"] == 5  # noqa: PLR2004
        assert factory.call_args.kwargs["headers"]["x-langfuse-ingestion-version"] == "4"
        with service.turn(uuid4().hex, TraceMetadata(status="succeeded")):
            pass
        service.client.flush()
        assert not exporter.get_finished_spans()[0].instrumentation_scope.attributes
        probe.assert_not_called()
    finally:
        await service.aclose()


async def test_graph_spans_share_root_and_concurrent_requests_are_isolated(
    settings: Settings,
) -> None:
    service, exporter = tracing(settings)
    barrier = asyncio.Barrier(2)
    expected_parents = {"prepare": "turn", "answer_data": "turn", "select_schema": "answer_data"}

    async def run() -> str:
        trace_id = uuid4().hex
        with service.turn(trace_id, TraceMetadata(status="running")) as root:
            result = await invoke_trace_graph(barrier)
            assert result.status == "succeeded"
            root.update(TraceMetadata(status=result.status))
        return trace_id

    try:
        async with asyncio.timeout(5):
            ids = await asyncio.gather(run(), run())
        assert len(set(ids)) == barrier.parties
        service.client.flush()
        spans = exporter.get_finished_spans()
        assert len(spans) == len(ids) * (len(expected_parents) + 1)
        assert {format(span.context.trace_id, "032x") for span in spans} == set(ids)
        for trace_id in ids:
            group = [span for span in spans if format(span.context.trace_id, "032x") == trace_id]
            by_name = {span.name: span for span in group}
            assert len(group) == len(by_name) == len(expected_parents) + 1
            assert set(by_name) == {"turn", *expected_parents}
            # An explicit trace ID uses an SDK non-recording parent for the application root.
            turn = by_name["turn"]
            assert turn.attributes["langfuse.internal.is_app_root"] is True
            assert turn.parent is not None
            assert turn.parent.trace_id == turn.context.trace_id
            assert turn.parent.span_id not in {span.context.span_id for span in spans}
            for name, parent in expected_parents.items():
                span = by_name[name]
                assert span.parent is not None
                assert span.parent.span_id == by_name[parent].context.span_id
                assert span.parent.trace_id == span.context.trace_id
            assert all(
                span.attributes["langfuse.observation.metadata.status"] == "succeeded"
                for span in group
            )
        assert not any(span.events for span in spans)
    finally:
        await service.aclose()
