"""HTTP/SSE expose finalized preferences and preserve them through durable replay."""

from http import HTTPStatus
from uuid import uuid4

import pytest

from app.agents.contracts import TurnIdentity
from app.core.errors import DatabaseError
from app.repositories.memory import MemoryRepository
from app.schemas.memory import MemoryType, StoredMemory
from tests.api.chat_support import Harness, chat, events
from tests.memory_extraction_support import extraction_input, source_pair
from tests.memory_support import memory_input

pytestmark = pytest.mark.integration
__all__ = ["chat"]


async def seed_format(chat: Harness) -> StoredMemory:
    owner = await source_pair(chat.database, extraction_input(), owner=TurnIdentity(
        user_id=chat.user.id, conversation_id=chat.cid, turn_id=uuid4(),
    ))
    async with chat.database.session() as session, session.begin():
        return await MemoryRepository(session, owner.user_id).create(memory_input(
            owner.turn_id, kind=MemoryType.FORMAT_PREFERENCE,
            content={"prefer": "table", "decimals": 3},
        ))


async def test_http_preference_and_replay_do_not_reread_memory(chat: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    await seed_format(chat)
    response = await chat.client.post(chat.url, json={"content": "2026年8月GMV"}, headers={"Idempotency-Key": "saved-format"})
    assert response.status_code == HTTPStatus.OK
    assert response.json()["answer"]["format_preference"] == {"prefer": "table", "decimals": 3}
    async def fail(repo: MemoryRepository, types: list[MemoryType]) -> list[StoredMemory]:
        raise DatabaseError()
    monkeypatch.setattr(MemoryRepository, "active_candidates", fail)
    replay = await chat.client.post(chat.url, json={"content": "2026年8月GMV"}, headers={"Idempotency-Key": "saved-format"})
    assert replay.json()["replayed"]
    assert replay.json()["answer"] == response.json()["answer"]


async def test_sse_preference_is_committed_before_terminal_event(chat: Harness) -> None:
    await seed_format(chat)
    response = await chat.client.post(chat.url + "/stream", json={"content": "2026年8月GMV，用文字回答"})
    assert response.status_code == HTTPStatus.OK
    parsed = events(response)
    assert parsed[-1][0] == "done"
    rows = await chat.stored()
    answer = rows[-1]["answer"]
    assert answer["format_preference"] == {"prefer": "prose", "decimals": 3}
    assert any(name == "token" for name, _ in parsed)


async def test_http_memory_failure_is_visible_and_suppresses_extraction(chat: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail(repo: MemoryRepository, types: list[MemoryType]) -> list[StoredMemory]:
        raise DatabaseError()
    monkeypatch.setattr(MemoryRepository, "active_candidates", fail)
    response = await chat.client.post(chat.url, json={"content": "2026年8月GMV"})
    assert response.status_code == HTTPStatus.OK
    assert response.json()["status"] == "degraded"
    assert response.json()["answer"]["degraded_components"] == ["memory"]
    assert response.json()["answer"]["format_preference"] is None
    assert chat.app.state.chat.memory.llm.calls == []
