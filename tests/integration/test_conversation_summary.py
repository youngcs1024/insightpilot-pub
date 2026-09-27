"""Real PostgreSQL proves summary ownership, catch-up and stale-writer exclusion."""

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.agents.contracts import TurnIdentity
from app.core.config_models import Settings
from app.core.errors import NotFoundError
from app.db.models import Conversation, TurnStatus
from app.db.session import Database
from app.repositories.turns import TurnRepository
from app.schemas.summary import SummaryOutput
from app.services.conversations import ConversationService
from app.services.schema_tokens import SchemaTokenCounter
from app.services.summary import SummaryService
from tests.fakes.chat_model import FakeChatModel
from tests.memory_extraction_support import extraction_input, source_pair
from tests.shared_database import TestPostgres

pytestmark = pytest.mark.integration


@pytest.fixture
async def summary_db(migrated_db: TestPostgres) -> AsyncIterator[Database]:
    database = Database(migrated_db.app)
    database.start()
    try:
        yield database
    finally:
        await database.aclose()


async def stored(database: Database, identity: TurnIdentity) -> tuple[str | None, int]:
    async with database.session() as session:
        row = await session.scalar(select(Conversation).where(Conversation.id == identity.conversation_id))
        assert row is not None
        return row.summary, row.summary_through_seq


async def append(database: Database, identity: TurnIdentity, status: TurnStatus) -> TurnIdentity:
    async with database.session() as session, session.begin():
        assistant = await TurnRepository(session, identity.user_id).create_pair(identity.conversation_id, "八月呢？", "summary-test")
        assistant.content = "尚缺少八月信息。"
        assistant.status = status
        return identity.model_copy(update={"turn_id": assistant.id})


async def test_summary_persisted_in_conversation(summary_db: Database, settings: Settings) -> None:
    identity = await source_pair(summary_db, extraction_input())
    llm = FakeChatModel([SummaryOutput(summary="用户要求退款率按申请时间计算。")])
    service = SummaryService(ConversationService(summary_db), settings, llm, SchemaTokenCounter())
    await service.run(identity)
    assert await stored(summary_db, identity) == ("用户要求退款率按申请时间计算。", 2)
    # Reconstruct services and replay the same completed turn: no generation occurs.
    replay = FakeChatModel([])
    await SummaryService(ConversationService(summary_db), settings, replay, SchemaTokenCounter()).run(identity)
    assert not replay.calls


async def test_stale_summary_write_rejected(summary_db: Database) -> None:
    identity = await source_pair(summary_db, extraction_input())
    service = ConversationService(summary_db)
    old = await service.summary_work(identity)
    assert old is not None
    assert await service.advance_summary(identity, old, "new summary")
    assert not await service.advance_summary(identity, old, "stale summary")
    assert await stored(summary_db, identity) == ("new summary", 2)


async def test_summary_catches_up_in_order_and_skips_failed_content(summary_db: Database, settings: Settings) -> None:
    first = await source_pair(summary_db, extraction_input())
    await append(summary_db, first, TurnStatus.FAILED)
    last = await append(summary_db, first, TurnStatus.ABSTAINED)
    llm = FakeChatModel([SummaryOutput(summary="保留第一轮。"), SummaryOutput(summary="保留第一轮；八月仍需澄清。")])
    await SummaryService(ConversationService(summary_db), settings, llm, SchemaTokenCounter()).run(last)
    assert len(llm.calls) == 2
    assert '"existing_summary":"保留第一轮。"' in str(llm.calls[1].messages[1].content)
    assert '"status":"abstained"' in str(llm.calls[1].messages[1].content)
    assert await stored(summary_db, last) == ("保留第一轮；八月仍需澄清。", 6)


async def test_summary_cross_user_read_and_write_are_rejected(summary_db: Database) -> None:
    identity = await source_pair(summary_db, extraction_input())
    service = ConversationService(summary_db)
    work = await service.summary_work(identity)
    assert work is not None
    foreign = identity.model_copy(update={"user_id": uuid4()})
    with pytest.raises(NotFoundError):
        await service.summary_work(foreign)
    assert not await service.advance_summary(foreign, work, "foreign")
    assert await stored(summary_db, identity) == (None, 0)


async def test_archive_during_generation_prevents_summary_commit(summary_db: Database) -> None:
    identity = await source_pair(summary_db, extraction_input())
    service = ConversationService(summary_db)
    work = await service.summary_work(identity)
    assert work is not None
    await service.archive(identity.user_id, identity.conversation_id)
    assert not await service.advance_summary(identity, work, "too late")
    with pytest.raises(NotFoundError):
        await service.summary_work(identity)


async def test_running_gap_is_not_skipped(summary_db: Database, settings: Settings) -> None:
    first = await source_pair(summary_db, extraction_input(status=TurnStatus.RUNNING))
    llm = FakeChatModel([])
    await SummaryService(ConversationService(summary_db), settings, llm, SchemaTokenCounter()).run(first)
    assert not llm.calls
    assert await stored(summary_db, first) == (None, 0)


async def test_oversize_summary_keeps_last_committed_progress(summary_db: Database, settings: Settings) -> None:
    first = await source_pair(summary_db, extraction_input())
    last = await append(summary_db, first, TurnStatus.DEGRADED)
    llm = FakeChatModel([SummaryOutput(summary="保留已提交第一轮。"), SummaryOutput(summary=" token" * 600)])
    await SummaryService(ConversationService(summary_db), settings, llm, SchemaTokenCounter()).run(last)
    assert await stored(summary_db, first) == ("保留已提交第一轮。", 2)
