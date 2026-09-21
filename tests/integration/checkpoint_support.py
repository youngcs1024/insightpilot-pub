"""Explicit checkpoint fixtures preserving real transactions and fixture scope."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from uuid import uuid4

import pytest

from app.agents.contracts import TurnIdentity
from app.agents.runtime import RuntimeContext
from app.core.config_models import DatabaseSettings
from app.db.models import Conversation, TurnStatus, User
from app.db.session import Database
from app.repositories.turns import TurnRepository
from app.services.conversations import ConversationService
from app.services.evidence import EvidenceService
from scripts.migration_settings import MigrationSettings
from scripts.setup_checkpointer import setup_checkpointer
from tests.agents.support import context, metric_intent, sql_candidate
from tests.answer_support import data_draft
from tests.database_support import DatabaseStack
from tests.fakes.chat_model import FakeChatModel


@pytest.fixture
def checkpoint_setup(migrated: MigrationSettings) -> None:
    asyncio.run(setup_checkpointer(migrated))


@pytest.fixture
async def graph_database(
    checkpoint_setup: None, migration_stack: DatabaseStack, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[Database, DatabaseSettings]]:
    for key in ("HOST", "PORT", "USER", "PASSWORD"):
        monkeypatch.delenv("IP_MIGRATION__" + key)
    settings = DatabaseSettings(
        port=migration_stack.settings.db_host_port,
        app_password=migration_stack.settings.bootstrap.app_password,
    )
    database = Database(settings)
    database.start()
    try:
        yield database, settings
    finally:
        await database.aclose()


async def admitted(database: Database, *, identity: TurnIdentity | None = None) -> TurnIdentity:
    async with database.session() as session, session.begin():
        if identity is None:
            user = User(
                email=f"graph-{uuid4()}@example.com",
                hashed_password="synthetic",  # noqa: S106 -- isolated test user; no login performed.
                display_name="Graph user",
            )
            session.add(user)
            await session.flush()
            conversation = Conversation(user_id=user.id, title="Graph history")
            session.add(conversation)
            await session.flush()
            user_id, conversation_id = user.id, conversation.id
        else:
            user_id, conversation_id = identity.user_id, identity.conversation_id
            prior = await TurnRepository(session, user_id).get(conversation_id, identity.turn_id)
            prior.status = TurnStatus.SUCCEEDED
            prior.content = "There were 42 orders."
            await session.flush()
        assistant = await TurnRepository(session, user_id).create_pair(
            conversation_id, "2026年8月GMV", None
        )
        return TurnIdentity(user_id=user_id, conversation_id=conversation_id, turn_id=assistant.id)


def connected_context(
    database: Database, identity: TurnIdentity, *, followup: bool = False
) -> RuntimeContext:
    base = context()
    if followup:
        base = replace(
            base,
            llm=FakeChatModel(
                [
                    metric_intent(),
                    sql_candidate(),
                    data_draft(markdown="42 orders", confidence=1),
                ]
            ),
        )
    return replace(
        base,
        identity=identity,
        conversations=ConversationService(database),
        evidence=EvidenceService(database),
    )
