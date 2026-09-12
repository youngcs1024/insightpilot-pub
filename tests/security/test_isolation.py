"""Real ownership dependencies prevent cross-user reads and writes."""

from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import Depends
from pydantic import BaseModel
from sqlalchemy import update

from app.api.dependencies import get_owned_conversation
from app.application import create_app
from app.core.config_models import Settings
from app.core.security import TokenCodec
from app.db.models import Conversation, User
from app.db.session import Database
from app.repositories.conversation import ConversationRepository

pytestmark = pytest.mark.integration
NOT_FOUND, OK = 404, 200


class OwnedResult(BaseModel):
    id: UUID


@pytest.fixture
async def identities(auth_database: Database) -> tuple[UUID, UUID, UUID]:
    async with auth_database.session() as session, session.begin():
        owner = User(
            email=f"{uuid4().hex}@example.com", hashed_password=uuid4().hex, display_name="Owner"
        )
        other = User(
            email=f"{uuid4().hex}@example.com", hashed_password=uuid4().hex, display_name="Other"
        )
        session.add_all([owner, other])
        await session.flush()
        conversation = Conversation(user_id=owner.id, title="Private")
        session.add(conversation)
        await session.flush()
        return owner.id, other.id, conversation.id


async def owned_request(
    method: str, auth_database: Database, settings: Settings, identities: tuple[UUID, UUID, UUID]
) -> None:
    application = create_app(settings, database=auth_database)

    @application.api_route("/test/conversations/{conversation_id}", methods=["GET", "POST"])
    async def owned(
        conversation: Annotated[Conversation, Depends(get_owned_conversation)],
    ) -> OwnedResult:
        return OwnedResult(id=conversation.id)

    owner, other, conversation_id = identities
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://test"
    ) as client:
        for user_id, expected in ((owner, OK), (other, NOT_FOUND)):
            token, _ = TokenCodec(settings.security).create(user_id, "access")
            result = await client.request(
                method,
                f"/test/conversations/{conversation_id}",
                headers={"Authorization": f"Bearer {token}"},
            )
            assert result.status_code == expected
        absent = await client.request(
            method, f"/test/conversations/{uuid4()}", headers={"Authorization": f"Bearer {token}"}
        )
        assert absent.status_code == NOT_FOUND


async def test_user_cannot_read_other_users_conversation(
    auth_database: Database, settings: Settings, identities: tuple[UUID, UUID, UUID]
) -> None:
    await owned_request("GET", auth_database, settings, identities)


async def test_user_cannot_post_to_other_users_conversation(
    auth_database: Database, settings: Settings, identities: tuple[UUID, UUID, UUID]
) -> None:
    await owned_request("POST", auth_database, settings, identities)


async def test_repository_scoping_prevents_leak(
    auth_database: Database, identities: tuple[UUID, UUID, UUID]
) -> None:
    owner, other, conversation_id = identities
    async with auth_database.session() as session:
        assert await ConversationRepository(session, other).get(conversation_id) is None
        assert await ConversationRepository(session, owner).get(conversation_id) is not None


async def test_archived_conversation_is_hidden(
    auth_database: Database, settings: Settings, identities: tuple[UUID, UUID, UUID]
) -> None:
    owner, _, conversation_id = identities
    async with auth_database.session() as session, session.begin():
        await session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id, Conversation.user_id == owner)
            .values(archived_at=datetime.now(UTC))
        )
    async with auth_database.session() as session:
        assert await ConversationRepository(session, owner).get(conversation_id) is None
    application = create_app(settings, database=auth_database)

    @application.get("/test/conversations/{conversation_id}")
    async def owned(
        conversation: Annotated[Conversation, Depends(get_owned_conversation)],
    ) -> OwnedResult:
        return OwnedResult(id=conversation.id)

    token, _ = TokenCodec(settings.security).create(owner, "access")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://test"
    ) as client:
        response = await client.get(
            f"/test/conversations/{conversation_id}", headers={"Authorization": f"Bearer {token}"}
        )
    assert response.status_code == NOT_FOUND
