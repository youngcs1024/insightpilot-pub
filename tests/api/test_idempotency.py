"""Test-only HTTP adapter over real PostgreSQL admission; no production chat route."""

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Annotated
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import Depends, Header, Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app.application import create_app
from app.core.config_models import DatabaseSettings, Settings
from app.core.deadline import Deadline, get_deadline
from app.core.errors import ConflictError, DatabaseTimeoutError, NotFoundError
from app.db.models import Conversation, Turn, TurnStatus, User
from app.db.session import Database
from app.repositories.turns import TurnRepository
from app.services.idempotency import AdmissionResult, IdempotencyService, MessageAdmission
from scripts.migration_settings import MigrationSettings
from tests.auth_support import deadline
from tests.database_support import DatabaseStack

pytestmark = pytest.mark.integration
OK, CONFLICT, NOT_FOUND = 200, 409, 404
PAIR_SIZE = 2


@pytest.fixture
async def database(
    migrated: MigrationSettings, migration_stack: DatabaseStack
) -> AsyncIterator[Database]:
    resource = Database(
        DatabaseSettings(
            port=migration_stack.settings.db_host_port,
            app_password=migration_stack.settings.bootstrap.app_password,
        )
    )
    resource.start()
    try:
        yield resource
    finally:
        await resource.aclose()


@pytest.fixture
async def admission(database: Database) -> MessageAdmission:
    async with database.session() as session, session.begin():
        user = User(
            email=f"{uuid4().hex}@example.invalid", hashed_password=uuid4().hex, display_name="Test"
        )
        session.add(user)
        await session.flush()
        conversation = Conversation(user_id=user.id, title="Idempotency")
        session.add(conversation)
        await session.flush()
        return MessageAdmission(
            user_id=user.id,
            conversation_id=conversation.id,
            content="原始问题",
            idempotency_key="test-key",
        )


@pytest.fixture
def service(database: Database) -> IdempotencyService:
    return IdempotencyService(database, timeout_s=10)


class MessageBody(BaseModel):
    """Only public content enters the test HTTP adapter."""

    content: str = Field(min_length=1, max_length=32_000)


@pytest.fixture
async def client(
    settings: Settings, database: Database, service: IdempotencyService, admission: MessageAdmission
) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(settings, database=database)

    @app.post("/test/conversations/{conversation_id}/messages")
    async def message(
        conversation_id: UUID,
        body: MessageBody,
        response: Response,
        budget: Annotated[Deadline, Depends(get_deadline)],
        idempotency_key: Annotated[str | None, Header(min_length=1, max_length=128)] = None,
    ) -> AdmissionResult:
        result = await service.admit(
            MessageAdmission(
                user_id=admission.user_id,
                conversation_id=conversation_id,
                content=body.content,
                idempotency_key=idempotency_key,
            ),
            deadline=budget,
        )
        if result.replayed:
            response.headers["Idempotency-Replayed"] = "true"
        return result

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as resource:
        yield resource


async def post(
    client: httpx.AsyncClient, admission: MessageAdmission, content: str | None = None
) -> httpx.Response:
    return await client.post(
        f"/test/conversations/{admission.conversation_id}/messages",
        json={"content": content or admission.content},
        headers={"Idempotency-Key": admission.idempotency_key or "test-key"},
    )


async def test_repeat_key_returns_original_turn(
    client: httpx.AsyncClient, admission: MessageAdmission
) -> None:
    first, second = await post(client, admission), await post(client, admission)
    assert first.status_code == second.status_code == OK
    assert first.json()["id"] == second.json()["id"]
    assert first.json()["id"] != second.json()["reply_to_turn_id"]
    assert second.headers["Idempotency-Replayed"] == "true"
    assert second.json()["status"] == "running"
    assert second.json()["content"] == ""
    assert "Idempotency-Replayed" not in first.headers


async def test_repeat_key_different_content_conflicts(
    client: httpx.AsyncClient, admission: MessageAdmission
) -> None:
    await post(client, admission)
    response = await post(client, admission, "不同问题")
    assert response.status_code == CONFLICT
    assert response.json()["code"] == "CONFLICT"
    assert "different content" not in response.text
    assert response.json()["request_id"] == response.headers["X-Request-ID"]


async def test_concurrent_same_key_creates_one_pair(
    service: IdempotencyService, admission: MessageAdmission, database: Database
) -> None:
    async with asyncio.TaskGroup() as group:
        tasks = [group.create_task(service.admit(admission, deadline=deadline())) for _ in range(5)]
    results = [task.result() for task in tasks]
    assert len({result.id for result in results}) == 1
    assert sum(not result.replayed for result in results) == 1
    async with database.session() as session:
        assert (
            await session.scalar(
                select(func.count())
                .select_from(Turn)
                .where(Turn.conversation_id == admission.conversation_id)
            )
            == PAIR_SIZE
        )


async def test_different_key_conflicts_while_running(
    service: IdempotencyService, admission: MessageAdmission
) -> None:
    await service.admit(admission, deadline=deadline())
    with pytest.raises(ConflictError):
        await service.admit(
            admission.model_copy(update={"idempotency_key": "another"}), deadline=deadline()
        )


async def test_cross_user_isolation(
    service: IdempotencyService, admission: MessageAdmission
) -> None:
    await service.admit(admission, deadline=deadline())
    with pytest.raises(NotFoundError):
        await service.admit(admission.model_copy(update={"user_id": uuid4()}), deadline=deadline())


async def test_cross_conversation_http_isolation(
    client: httpx.AsyncClient, admission: MessageAdmission
) -> None:
    response = await post(client, admission.model_copy(update={"conversation_id": uuid4()}))
    assert response.status_code == NOT_FOUND


async def test_failed_result_replays_after_service_restart(
    service: IdempotencyService, admission: MessageAdmission, database: Database
) -> None:
    first = await service.admit(admission, deadline=deadline())
    await service.fail_deadline(
        user_id=admission.user_id, conversation_id=admission.conversation_id, turn_id=first.id
    )
    replay = await IdempotencyService(database, timeout_s=10).admit(admission, deadline=deadline())
    assert replay.replayed
    assert replay.id == first.id
    assert replay.status == TurnStatus.FAILED
    assert replay.failure_reason == "deadline_exceeded"


async def finish(database: Database, turn_id: UUID, *, age: timedelta = timedelta()) -> None:
    async with database.session() as session, session.begin():
        turn = await session.get(Turn, turn_id)
        assert turn is not None
        turn.status = TurnStatus.SUCCEEDED
        turn.content = "持久化的答案"
        turn.created_at = datetime.now(UTC) - age


async def test_completed_replay_and_cleanup_preserve_answer(
    service: IdempotencyService, admission: MessageAdmission, database: Database
) -> None:
    first = await service.admit(admission, deadline=deadline())
    await finish(database, first.id)
    await service.fail_deadline(
        user_id=admission.user_id, conversation_id=admission.conversation_id, turn_id=first.id
    )
    replay = await service.admit(admission, deadline=deadline())
    assert replay.content == "持久化的答案"
    assert replay.status == TurnStatus.SUCCEEDED


async def test_expired_key_reused_without_deleting_history(
    service: IdempotencyService, admission: MessageAdmission, database: Database
) -> None:
    first = await service.admit(admission, deadline=deadline())
    await finish(database, first.id, age=timedelta(hours=25))
    second = await service.admit(
        admission.model_copy(update={"content": "新问题"}), deadline=deadline()
    )
    assert second.id != first.id
    assert not second.replayed
    async with database.session() as session:
        prior = await session.get(Turn, first.id)
        assert prior is not None
        assert prior.idempotency_key is None
        assert prior.content == "持久化的答案"
        assert await session.get(Turn, prior.reply_to_turn_id) is not None


async def test_expired_running_key_is_not_reused(
    service: IdempotencyService, admission: MessageAdmission, database: Database
) -> None:
    first = await service.admit(admission, deadline=deadline())
    async with database.session() as session, session.begin():
        turn = await session.get(Turn, first.id)
        assert turn is not None
        turn.created_at = datetime.now(UTC) - timedelta(days=2)
    replay = await service.admit(admission, deadline=deadline())
    assert replay.id == first.id
    assert replay.replayed


async def test_no_key_creates_new_pair_after_completion(
    service: IdempotencyService, admission: MessageAdmission, database: Database
) -> None:
    request = admission.model_copy(update={"idempotency_key": None})
    first = await service.admit(request, deadline=deadline())
    await finish(database, first.id)
    second = await service.admit(request, deadline=deadline())
    assert first.id != second.id
    assert not second.replayed


async def test_cleanup_timeout_is_bounded(
    service: IdempotencyService, admission: MessageAdmission, database: Database
) -> None:
    first = await service.admit(admission, deadline=deadline())
    bounded = IdempotencyService(database, timeout_s=0.02)
    async with database.session() as session, session.begin():
        await session.execute(
            select(Conversation)
            .where(Conversation.id == admission.conversation_id)
            .with_for_update()
        )
        with pytest.raises(DatabaseTimeoutError):
            await bounded.fail_deadline(
                user_id=admission.user_id,
                conversation_id=admission.conversation_id,
                turn_id=first.id,
            )
    replay = await service.admit(admission, deadline=deadline())
    assert replay.status == TurnStatus.RUNNING


@pytest.mark.parametrize("offset_us", [-1, 0, 1])
async def test_exact_24_hour_boundary(
    service: IdempotencyService,
    admission: MessageAdmission,
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
    offset_us: int,
) -> None:
    replayed = offset_us < 0

    first = await service.admit(admission, deadline=deadline())
    await finish(database, first.id)
    async with database.session() as session:
        turn = await session.get(Turn, first.id)
        assert turn is not None
        boundary = turn.created_at + timedelta(hours=24, microseconds=offset_us)

    async def now(self: TurnRepository) -> datetime:
        return boundary

    monkeypatch.setattr(TurnRepository, "current_time", now)
    result = await service.admit(admission, deadline=deadline())
    assert result.replayed is replayed
    assert (result.id == first.id) is replayed


async def test_different_content_concurrent_requests_conflict(
    service: IdempotencyService,
    admission: MessageAdmission,
) -> None:
    async def attempt(request: MessageAdmission) -> AdmissionResult | ConflictError:
        try:
            return await service.admit(request, deadline=deadline())
        except ConflictError as exc:
            return exc

    async with asyncio.TaskGroup() as group:
        first = group.create_task(attempt(admission))
        second = group.create_task(attempt(admission.model_copy(update={"content": "other"})))
    assert sum(isinstance(task.result(), ConflictError) for task in (first, second)) == 1


async def test_deadline_cancels_fake_graph_and_finalizes_turn(
    settings: Settings,
    database: Database,
    service: IdempotencyService,
    admission: MessageAdmission,
) -> None:
    # The test adapter demonstrates the future ChatService cancellation boundary.
    # No test graph/route is installed in the production application.
    settings.http.request_timeout_s = 0.2
    app = create_app(settings, database=database)
    cancelled = asyncio.Event()

    @app.post("/test/slow")
    async def slow(budget: Annotated[Deadline, Depends(get_deadline)]) -> None:
        original = await service.admit(admission, deadline=budget)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await service.fail_deadline(
                user_id=admission.user_id,
                conversation_id=admission.conversation_id,
                turn_id=original.id,
            )
            raise

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/test/slow")
    assert response.status_code == 504  # noqa: PLR2004 -- HTTP contract under test.
    assert cancelled.is_set()
    replay = await service.admit(admission, deadline=deadline())
    assert replay.status == TurnStatus.FAILED
    assert replay.failure_reason == "deadline_exceeded"


async def test_failure_finalization_checks_user(
    service: IdempotencyService,
    admission: MessageAdmission,
) -> None:
    original = await service.admit(admission, deadline=deadline())
    with pytest.raises(NotFoundError):
        await service.fail_deadline(
            user_id=uuid4(), conversation_id=admission.conversation_id, turn_id=original.id
        )
    assert (await service.admit(admission, deadline=deadline())).status == TurnStatus.RUNNING


async def test_repository_cannot_insert_into_foreign_conversation(
    database: Database,
    admission: MessageAdmission,
) -> None:
    async with database.session() as session, session.begin():
        with pytest.raises(NotFoundError):
            await TurnRepository(session, uuid4()).create_pair(
                admission.conversation_id, "foreign", None
            )
        assert (
            await session.scalar(
                select(func.count())
                .select_from(Turn)
                .where(Turn.conversation_id == admission.conversation_id)
            )
            == 0
        )
