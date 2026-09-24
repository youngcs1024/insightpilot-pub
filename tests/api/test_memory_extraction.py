"""Chat responses commit first and never await background extraction."""

import asyncio
from http import HTTPStatus
from unittest.mock import Mock

import pytest

from app.agents.contracts import TurnIdentity
from app.core import background
from app.core.errors import LlmUnavailableError
from app.db.models.turn import TurnStatus
from app.repositories.memory import MemoryRepository
from app.repositories.turns import TurnRepository
from app.schemas.memory import MemoryType
from app.schemas.memory_extraction import MemoryExtraction
from tests.api.chat_support import Harness, chat
from tests.fakes.chat_model import FakeChatModel
from tests.memory_extraction_support import DURABLE, candidate

pytestmark = pytest.mark.integration
__all__ = ["chat"]


async def wait_for_memory() -> None:
    tasks = [task for task in background._tasks if task.get_name() == "extract-turn-memory"]
    if tasks:
        async with asyncio.timeout(10):
            await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.sleep(0)


async def test_commit_precedes_background_and_response_does_not_wait(
    chat: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    seen: list[TurnIdentity] = []

    async def blocked(identity: TurnIdentity) -> None:
        async with chat.database.session() as session:
            turn = await TurnRepository(session, identity.user_id).get(identity.conversation_id, identity.turn_id)
            assert turn.status is TurnStatus.SUCCEEDED
            assert turn.answer is not None
        seen.append(identity)
        entered.set()
        await release.wait()

    monkeypatch.setattr(chat.app.state.chat.memory, "run", blocked)
    try:
        async with asyncio.timeout(10):
            response = await chat.client.post(chat.url, json={"content": DURABLE}, headers={"Idempotency-Key": "memory"})
            await entered.wait()
        assert response.status_code == HTTPStatus.OK
        assert not release.is_set()
        replay = await chat.client.post(chat.url, json={"content": DURABLE}, headers={"Idempotency-Key": "memory"})
        assert replay.json()["replayed"]
        assert len(seen) == 1
    finally:
        release.set()
        await wait_for_memory()


async def test_chat_background_reads_original_message_and_persists(chat: Harness) -> None:
    chat.app.state.chat.memory.llm = FakeChatModel([MemoryExtraction(candidates=[candidate()])])
    response = await chat.client.post(chat.url, json={"content": DURABLE})
    assert response.status_code == HTTPStatus.OK
    await wait_for_memory()
    async with chat.database.session() as session:
        stored = await MemoryRepository(session, chat.user.id).list_active(MemoryType.METRIC_OVERRIDE)
    assert len(stored) == 1
    assert str(stored[0].source_turn_id) == response.json()["id"]
    assert str(stored[0].source_turn_id) != response.json()["reply_to_turn_id"]


async def test_background_failure_preserves_persisted_answer(
    chat: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat.app.state.chat.memory.llm = FakeChatModel([LlmUnavailableError()])
    counter = Mock()
    monkeypatch.setattr(background, "_failures", counter)
    response = await chat.client.post(chat.url, json={"content": DURABLE})
    assert response.status_code == HTTPStatus.OK
    await wait_for_memory()
    rows = await chat.stored()
    assert rows[1]["status"] == "succeeded"
    assert rows[1]["answer"] == response.json()["answer"]
    counter.add.assert_called_once_with(1)
