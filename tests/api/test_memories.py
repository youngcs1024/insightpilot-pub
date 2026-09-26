"""Memory audit reads require authentication, isolation and bounded user quotas."""

# ruff: noqa: PLR2004 -- fixed threshold, row-count and pagination acceptance values.

from http import HTTPStatus
from uuid import uuid4

import pytest

from app.agents.contracts import TurnIdentity
from app.api.dependencies import get_current_user
from app.core.config_models import RateRule
from app.core.errors import DatabaseError
from app.repositories.memory import MemoryRepository
from app.schemas.auth import UserResponse
from app.schemas.memory_extraction import MemoryExtraction
from tests.api.chat_support import Harness, chat
from tests.memory_extraction_support import candidate, extraction_input, source_pair

pytestmark = pytest.mark.integration
__all__ = ["chat"]
URL = "/api/v1/memories"


async def seed(chat: Harness) -> list[str]:
    owner = TurnIdentity(user_id=chat.user.id, conversation_id=chat.cid, turn_id=uuid4())
    identity = await source_pair(chat.database, extraction_input(), owner=owner)
    outcomes = await chat.app.state.chat.memory._write(
        identity,
        MemoryExtraction(
            candidates=[
                candidate(),
                candidate(
                    content={"metric_key": "refund_rate", "patch": {"date_field": "paid_at"}}
                ),
                candidate(memory_type="terminology", content={"term": "大促", "means": "618"}),
            ]
        ),
    )
    return [str(result.memory_id) for result in outcomes]


async def test_active_default_and_history_chain(chat: Harness) -> None:
    identifiers = await seed(chat)
    response = await chat.client.get(URL)
    assert response.status_code == HTTPStatus.OK
    body = response.json()
    assert body["schema_version"] == 1
    assert body["request_id"]
    assert body["limit"] == 20
    assert body["offset"] == 0
    assert {row["id"] for row in body["items"]} == set(identifiers[1:])
    response = await chat.client.get(URL, params={"include_superseded": "true"})
    rows = response.json()["items"]
    assert len(rows) == len(identifiers)
    assert [(row["created_at"], row["id"]) for row in rows] == sorted(
        (row["created_at"], row["id"]) for row in rows
    )
    by_id = {row["id"]: row for row in rows}
    old, new = by_id[identifiers[0]], by_id[identifiers[1]]
    assert old["superseded_by"] == new["id"]
    assert old["superseded_at"]
    assert not old["is_active"]
    assert new["is_active"]
    assert all(row["source_turn_id"] and row["created_at"] and row["updated_at"] for row in rows)


async def test_filter_and_stable_pages(chat: Harness) -> None:
    identifiers = await seed(chat)
    query = {"include_superseded": "true", "memory_type": "metric_override"}
    full = (await chat.client.get(URL, params=query)).json()["items"]
    first = (await chat.client.get(URL, params={**query, "limit": 1})).json()["items"]
    second = (await chat.client.get(URL, params={**query, "limit": 1, "offset": 1})).json()["items"]
    assert first + second == full
    assert {row["id"] for row in full} == set(identifiers[:2])
    assert not (await chat.client.get(URL, params={"offset": 100})).json()["items"]


async def test_requires_real_access_authentication(chat: Harness) -> None:
    chat.app.dependency_overrides.pop(get_current_user)
    for headers in ({}, {"Authorization": "Bearer invalid"}):
        response = await chat.client.get(URL, headers=headers)
        assert response.status_code == HTTPStatus.UNAUTHORIZED
        assert response.json()["request_id"]


async def test_history_is_user_scoped(chat: Harness) -> None:
    await seed(chat)
    other = await source_pair(chat.database, extraction_input())

    async def other_user() -> UserResponse:
        return chat.user.model_copy(update={"id": other.user_id})

    chat.app.dependency_overrides[get_current_user] = other_user
    response = await chat.client.get(URL, params={"include_superseded": "true"})
    assert response.status_code == HTTPStatus.OK
    assert response.json()["items"] == []


@pytest.mark.parametrize(
    "query",
    [
        {"limit": 0},
        {"limit": 101},
        {"offset": -1},
        {"memory_type": "unknown"},
        {"include_superseded": "maybe"},
    ],
)
async def test_invalid_query_is_rejected(chat: Harness, query: dict[str, object]) -> None:
    response = await chat.client.get(URL, params=query)
    assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY
    assert response.json()["request_id"]


@pytest.mark.parametrize("reverse", [False, True])
async def test_all_quotas_apply_per_user_in_separate_bucket(chat: Harness, reverse: bool) -> None:
    rules = [RateRule(requests=1, seconds=60), RateRule(requests=10, seconds=3600)]
    chat.app.state.auth_limiter.settings.memories_rules = (
        list(reversed(rules)) if reverse else rules
    )
    assert (await chat.client.get(URL)).status_code == HTTPStatus.OK
    limited = await chat.client.get(URL)
    assert limited.status_code == HTTPStatus.TOO_MANY_REQUESTS
    assert limited.headers["retry-after"]
    assert (await chat.client.get("/api/v1/conversations")).status_code == HTTPStatus.OK
    other = await source_pair(chat.database, extraction_input())

    async def other_user() -> UserResponse:
        return chat.user.model_copy(update={"id": other.user_id})

    chat.app.dependency_overrides[get_current_user] = other_user
    assert (await chat.client.get(URL)).status_code == HTTPStatus.OK


async def test_database_failure_is_not_an_empty_success(
    chat: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fail(*args: object, **kwargs: object) -> list[object]:
        raise DatabaseError("private-memory-query")

    monkeypatch.setattr(MemoryRepository, "page", fail)
    response = await chat.client.get(URL)
    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert response.json()["code"] == "DATABASE_ERROR"
    assert response.json()["request_id"]
    assert "private-memory-query" not in response.text
