"""Summary scheduling never delays committed HTTP/SSE responses or replay."""

import asyncio
from http import HTTPStatus
from unittest.mock import Mock

import pytest

from app.agents.contracts import TurnIdentity
from app.core import background
from app.db.models import TurnStatus
from app.repositories.turns import TurnRepository
from tests.api.chat_support import Harness, chat

pytestmark = pytest.mark.integration
__all__ = ["chat"]


@pytest.mark.parametrize("stream", [False, True])
async def test_summary_is_after_commit_and_nonblocking(
    chat: Harness, monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    seen: list[TurnIdentity] = []

    async def blocked(identity: TurnIdentity) -> None:
        async with chat.database.session() as session:
            row = await TurnRepository(session, identity.user_id).get(
                identity.conversation_id, identity.turn_id
            )
            assert row is not None
            assert row.status is TurnStatus.SUCCEEDED
        seen.append(identity)
        entered.set()
        await release.wait()

    monkeypatch.setattr(chat.app.state.chat.summary, "run", Mock(side_effect=blocked))
    try:
        response = await chat.client.post(
            (chat.url + "/stream" if stream else chat.url),
            json={"content": "count"},
            headers={"Idempotency-Key": "summary-nonblocking"},
        )
        assert response.status_code == HTTPStatus.OK, response.text
        async with asyncio.timeout(10):
            await entered.wait()
        assert not release.is_set()
        replay = await chat.client.post(
            (chat.url + "/stream" if stream else chat.url),
            json={"content": "count"},
            headers={"Idempotency-Key": "summary-nonblocking"},
        )
        assert replay.status_code == HTTPStatus.OK
        assert len(seen) == 1
    finally:
        release.set()
        tasks = [t for t in background._tasks if t.get_name() == "summarize-conversation"]
        if tasks:
            await asyncio.gather(*tasks)
