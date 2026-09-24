"""Extraction gates, bounded background work and safe failures without external I/O."""

import asyncio
import json
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from pydantic import ValidationError
from structlog.testing import capture_logs

from app.agents.contracts import TurnIdentity
from app.core import background
from app.core.config_models import Settings
from app.core.errors import LlmStructuredOutputError, LlmUnavailableError, MemoryExtractionError
from app.core.llm_config import ModelRole
from app.db.models.turn import TurnRole, TurnStatus
from app.db.session import Database
from app.schemas.memory_extraction import MemoryExtraction
from app.services.memory.extract import MemoryExtractionService, extract
from tests.auth_support import deadline
from tests.fakes.chat_model import FakeChatModel
from tests.memory_extraction_support import DURABLE, TRANSIENT, candidate, extraction_input


def identity() -> TurnIdentity:
    return TurnIdentity(user_id=uuid4(), conversation_id=uuid4(), turn_id=uuid4())


@pytest.mark.parametrize(
    "status", [TurnStatus.FAILED, TurnStatus.ABSTAINED, TurnStatus.DEGRADED, TurnStatus.RUNNING]
)
async def test_ineligible_turn_never_calls_model(status: TurnStatus) -> None:
    llm = FakeChatModel([])
    assert not (await extract(extraction_input(status=status), llm, deadline=deadline())).candidates
    assert not llm.calls


@pytest.mark.parametrize("flag", ["abstained", "degraded_components", "user_role"])
async def test_inconsistent_success_is_rejected(flag: str) -> None:
    inputs = extraction_input()
    if flag == "user_role":
        inputs.role = TurnRole.USER
    elif flag == "abstained":
        inputs.answer.abstained = True
    else:
        inputs.answer.degraded_components = ["llm"]
    llm = FakeChatModel([])
    assert not (await extract(inputs, llm, deadline=deadline())).candidates
    assert not llm.calls


@pytest.mark.parametrize("status", [TurnStatus.FAILED, TurnStatus.ABSTAINED])
async def test_failed_turn_writes_nothing(settings: Settings, status: TurnStatus) -> None:
    service = MemoryExtractionService(Mock(spec=Database), settings, FakeChatModel([]))
    service._load = AsyncMock(return_value=extraction_input(status=status))
    service._write = AsyncMock()
    await service.run(identity())
    service._write.assert_not_awaited()


async def test_abstained_turn_writes_nothing(settings: Settings) -> None:
    await test_failed_turn_writes_nothing(settings, TurnStatus.ABSTAINED)


@pytest.mark.parametrize(("confidence", "accepted"), [(0.69, False), (0.7, True), (1, True)])
async def test_low_confidence_discarded(confidence: float, accepted: bool) -> None:
    llm = FakeChatModel([MemoryExtraction(candidates=[candidate(confidence=confidence)])])
    with capture_logs() as logs:
        result = await extract(extraction_input(), llm, deadline=deadline())
    assert bool(result.candidates) is accepted
    if not accepted:
        assert any(event.get("reason") == "confidence" for event in logs)


async def test_transient_filter_not_stored() -> None:
    llm = FakeChatModel([MemoryExtraction()])
    result = await extract(extraction_input(user_message=TRANSIENT), llm, deadline=deadline())
    assert not result.candidates
    prompt = llm.calls[0].messages[0].content
    assert TRANSIENT in prompt
    assert "DO NOT STORE" in prompt
    assert "UNTRUSTED DATA" in prompt


async def test_durable_preference_stored(settings: Settings) -> None:
    llm = FakeChatModel([MemoryExtraction(candidates=[candidate()])])
    service = MemoryExtractionService(Mock(spec=Database), settings, llm)
    service._load = AsyncMock(return_value=extraction_input())
    service._write = AsyncMock()
    turn = identity()
    await service.run(turn)
    service._write.assert_awaited_once_with(turn, MemoryExtraction(candidates=[candidate()]))
    assert llm.calls[0].role is ModelRole.MEMORY_EXTRACT


@pytest.mark.parametrize(
    "quote", ["按申请时间算", "助手说的偏好", " ", "以后退款率 都按退款申请时间算"]
)
async def test_quote_not_in_message_rejected(quote: str) -> None:
    llm = FakeChatModel([MemoryExtraction(candidates=[candidate(evidence_quote=quote)])])
    with capture_logs() as logs:
        result = await extract(extraction_input(), llm, deadline=deadline())
    assert not result.candidates
    assert any(
        event.get("reason") == "evidence_quote" and event["log_level"] == "warning"
        for event in logs
    )
    assert DURABLE not in str(logs)


@pytest.mark.parametrize(
    "update",
    [
        {"evidence_quote": ""},
        {"memory_type": "world_fact"},
        {"memory_type": "region_focus"},
        {"confidence": float("nan")},
        {"confidence": 1.1},
        {"summary": "x" * 201},
        {"content": {"metric_key": "refund_rate", "patch": {"expression": "x" * 2001}}},
    ],
)
def test_invalid_candidate_schema(update: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        candidate(**update)


async def test_invalid_model_output_is_sanitized() -> None:
    value = candidate().model_dump()
    value["memory_type"] = "private-invalid-content"
    llm = FakeChatModel([json.dumps({"candidates": [value]})])
    with capture_logs() as logs, pytest.raises(LlmStructuredOutputError) as caught:
        await extract(extraction_input(), llm, deadline=deadline())
    assert "private-invalid-content" not in str(caught.value)
    assert "private-invalid-content" not in str(logs)


async def test_extraction_failure_does_not_fail_turn(settings: Settings) -> None:
    llm = FakeChatModel([LlmUnavailableError("private-upstream-detail")])
    service = MemoryExtractionService(Mock(spec=Database), settings, llm)
    inputs = extraction_input()
    original = inputs.model_copy(deep=True)
    service._load = AsyncMock(return_value=inputs)
    service._write = AsyncMock()
    counter = Mock()
    with pytest.MonkeyPatch.context() as patch, capture_logs() as logs:
        patch.setattr("app.services.memory.extract._failures", counter)
        task = background.spawn(service.run(identity()), name="extract-turn-memory")
        with pytest.raises(MemoryExtractionError):
            await task
        await asyncio.sleep(0)
    counter.add.assert_called_once_with(1)
    assert inputs == original
    service._write.assert_not_awaited()
    assert any(event["event"] == "background_task_failed" for event in logs)
    assert "private-upstream-detail" not in str(logs)


async def test_background_task_reference_retained(settings: Settings) -> None:
    service = MemoryExtractionService(Mock(spec=Database), settings, FakeChatModel([]))
    release = asyncio.Event()

    async def load(_: TurnIdentity) -> None:
        await release.wait()

    service._load = load
    task = background.spawn(service.run(identity()), name="extract-turn-memory")
    assert task in background._tasks
    release.set()
    await task
    await asyncio.sleep(0)
    assert task not in background._tasks


async def test_background_timeout_is_bounded(settings: Settings) -> None:
    service = MemoryExtractionService(Mock(spec=Database), settings, FakeChatModel([]))
    service.timeout_s = 0.01

    async def block(_: TurnIdentity) -> None:
        await asyncio.Event().wait()

    service._load = block
    with pytest.raises(MemoryExtractionError):
        await service.run(identity())


async def test_background_cancellation_is_not_a_failure(settings: Settings) -> None:
    service = MemoryExtractionService(Mock(spec=Database), settings, FakeChatModel([]))
    service._load = AsyncMock(side_effect=asyncio.CancelledError)
    with pytest.raises(asyncio.CancelledError):
        await service.run(identity())


async def test_background_uses_fresh_budget(settings: Settings) -> None:
    llm = FakeChatModel([MemoryExtraction()])
    service = MemoryExtractionService(Mock(spec=Database), settings, llm)
    service._load = AsyncMock(return_value=extraction_input())
    before = deadline().at
    await service.run(identity())
    assert llm.calls[0].deadline_at > before
