"""Real chat transactions remain authoritative when observability fails."""

import asyncio
import json
from typing import Literal
from unittest.mock import Mock
from uuid import uuid4

import pytest
import structlog
from langfuse import LangfuseSpan

from app.core.logging import setup_logging
from tests.api.chat_support import OK, Harness, asgi_stream, chat
from tests.observability_support import tracing

__all__ = ["chat"]
pytestmark = pytest.mark.integration


async def test_request_id_in_logs_and_response(
    chat: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    setup_logging(chat.app.state.settings)
    service, exporter = tracing(chat.app.state.settings)
    chat.app.state.chat.observability = service
    incoming = str(uuid4()).upper()
    expected = incoming.lower().replace("-", "")
    try:
        response = await chat.client.post(
            chat.url, json={"content": "订单数?"}, headers={"X-Request-ID": incoming}
        )
        assert response.status_code == OK
        assert response.headers["X-Request-ID"] == response.json()["trace_id"] == expected
        records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
        event = next(record for record in records if record["event"] == "turn_completed")
        assert event["request_id"] == expected
        assert event["turn_id"] == response.json()["id"]
        service.client.flush()
        spans = exporter.get_finished_spans()
        assert {format(span.context.trace_id, "032x") for span in spans} == {expected}
        root = next(span for span in spans if span.name == "turn")
        assert root.attributes["langfuse.observation.metadata.status"] == "succeeded"
        assert root.attributes["langfuse.observation.metadata.user_id"] == str(chat.user.id)
        assert root.attributes["langfuse.observation.metadata.conversation_id"] == str(chat.cid)
        persistence = next(span for span in spans if span.name == "persist_evidence")
        assert (
            "[<redacted: 1 rows × 1 cols>]" in persistence.attributes["langfuse.observation.output"]  # noqa: RUF001 -- contract notation.
        )
        assert "SELECT 42" not in json.dumps([dict(span.attributes) for span in spans])
        assert structlog.contextvars.get_contextvars() == {}
    finally:
        await service.aclose()


@pytest.mark.parametrize("operation", ["start_observation", "callback", "update", "end"])
@pytest.mark.parametrize("stream", [False, True])
async def test_langfuse_failure_does_not_fail_turn(
    chat: Harness,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    stream: bool,
) -> None:
    service, _ = tracing(chat.app.state.settings)
    chat.app.state.chat.observability = service
    target = service.client if operation == "start_observation" else LangfuseSpan
    # Root creation still succeeds; callback child observation creation fails independently.
    method = "start_observation" if operation == "callback" else operation
    monkeypatch.setattr(target, method, Mock(side_effect=RuntimeError("private-upstream-error")))
    try:
        response = await chat.client.post(
            chat.url + ("/stream" if stream else ""), json={"content": "订单数?"}
        )
        assert response.status_code == OK
        assert "private-upstream-error" not in response.text
        if stream:
            assert "event: done" in response.text
        else:
            assert response.json()["status"] == "succeeded"
        assert (await chat.stored())[-1]["status"] == "succeeded"
    finally:
        await service.aclose()


async def test_replay_does_not_create_another_turn_trace(chat: Harness) -> None:
    service, exporter = tracing(chat.app.state.settings)
    chat.app.state.chat.observability = service
    try:
        headers = {"Idempotency-Key": uuid4().hex}
        first = await chat.client.post(chat.url, json={"content": "订单数?"}, headers=headers)
        second = await chat.client.post(chat.url, json={"content": "订单数?"}, headers=headers)
        assert second.json()["trace_id"] == first.json()["trace_id"]
        assert second.headers["X-Request-ID"] != first.headers["X-Request-ID"]
        service.client.flush()
        assert len([s for s in exporter.get_finished_spans() if s.name == "turn"]) == 1
    finally:
        await service.aclose()


async def test_failed_turn_root_records_failure(chat: Harness) -> None:
    service, exporter = tracing(chat.app.state.settings)
    chat.app.state.chat.observability = service
    chat.graph.error = RuntimeError("private-business-value")
    try:
        response = await chat.client.post(chat.url, json={"content": "订单数?"})
        assert response.status_code == 500  # noqa: PLR2004
        service.client.flush()
        root = next(span for span in exporter.get_finished_spans() if span.name == "turn")
        assert root.attributes["langfuse.observation.metadata.status"] == "failed"
        assert "private-business-value" not in json.dumps(dict(root.attributes))
    finally:
        await service.aclose()


async def test_committed_success_preserved_when_result_read_fails(
    chat: Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, exporter = tracing(chat.app.state.settings)
    chat.app.state.chat.observability = service
    original = chat.app.state.chat.read

    async def read(*args: object, **kwargs: object) -> object:
        result = await original(*args, **kwargs)
        if result.status.value == "succeeded":
            raise RuntimeError("read failed")
        return result

    monkeypatch.setattr(chat.app.state.chat, "read", read)
    try:
        await chat.client.post(chat.url, json={"content": "订单数?"})
        assert (await chat.stored())[-1]["status"] == "succeeded"
        service.client.flush()
        root = next(span for span in exporter.get_finished_spans() if span.name == "turn")
        assert root.attributes["langfuse.observation.metadata.status"] == "succeeded"
    finally:
        await service.aclose()


@pytest.mark.parametrize("mode", ["disconnect", "committed"])
async def test_disconnect_trace_matches_commit_boundary(
    chat: Harness,
    monkeypatch: pytest.MonkeyPatch,
    mode: Literal["disconnect", "committed"],
) -> None:
    service, exporter = tracing(chat.app.state.settings)
    chat.app.state.chat.observability = service
    if mode == "disconnect":
        chat.graph.release.clear()
    try:
        await asgi_stream(chat, mode, monkeypatch)
        stored = (await chat.stored())[-1]
        service.client.flush()
        root = next(span for span in exporter.get_finished_spans() if span.name == "turn")
        assert root.attributes["langfuse.observation.metadata.status"] == stored["status"]
        assert stored["status"] == ("failed" if mode == "disconnect" else "succeeded")
    finally:
        await service.aclose()


async def test_cancellation_immediately_after_commit_preserves_trace_success(
    chat: Harness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, exporter = tracing(chat.app.state.settings)
    chat.app.state.chat.observability = service
    original = chat.app.state.chat._succeed

    async def committed_then_cancel(*args: object, **kwargs: object) -> None:
        await original(*args, **kwargs)
        raise asyncio.CancelledError()

    monkeypatch.setattr(chat.app.state.chat, "_succeed", committed_then_cancel)
    try:
        with pytest.raises(asyncio.CancelledError):
            await chat.client.post(chat.url, json={"content": "count"})
        assert (await chat.stored())[-1]["status"] == "succeeded"
        service.client.flush()
        root = next(span for span in exporter.get_finished_spans() if span.name == "turn")
        assert root.attributes["langfuse.observation.metadata.status"] == "succeeded"
    finally:
        await service.aclose()
