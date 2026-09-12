"""Database execution leases and scoped startup reconciliation candidates."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from app.db.models import Conversation, Turn, TurnStatus, User


class RunningIdentity(BaseModel):
    """Only IDs cross the startup discovery boundary."""

    user_id: UUID
    conversation_id: UUID
    turn_id: UUID


async def user_ids(session: AsyncSession) -> list[UUID]:
    """Enumerate identities for system startup; user data reads remain scoped."""
    return list(await session.scalars(select(User.id)))


async def running_turns(
    session: AsyncSession, user_id: UUID, before: datetime
) -> list[RunningIdentity]:
    """Only pre-startup turns for this user are reconciliation candidates."""
    rows = await session.execute(
        select(Turn.conversation_id, Turn.id)
        .join(Conversation)
        .where(
            Conversation.user_id == user_id,
            Conversation.archived_at.is_(None),
            Turn.status == TurnStatus.RUNNING,
            Turn.created_at < before,
        )
    )
    return [RunningIdentity(user_id=user_id, conversation_id=cid, turn_id=tid) for cid, tid in rows]


async def database_time(session: AsyncSession) -> datetime:
    """Use database time for the startup cutoff."""
    value: datetime = (await session.execute(select(func.clock_timestamp()))).scalar_one()
    return value


async def try_lease(connection: AsyncConnection, conversation_id: UUID) -> bool:
    """Reserve admission through finalization, independently from the graph's lock."""
    return bool(
        await connection.scalar(
            text("SELECT pg_try_advisory_lock(hashtextextended(:key, 0))"),
            {"key": "chat:" + str(conversation_id)},
        )
    )


async def release_lease(connection: AsyncConnection, conversation_id: UUID) -> None:
    """Release a session lock before returning its connection to the pool."""
    await connection.execute(
        text("SELECT pg_advisory_unlock(hashtextextended(:key, 0))"),
        {"key": "chat:" + str(conversation_id)},
    )


async def try_graph_guard(session: AsyncSession, conversation_id: UUID) -> bool:
    """A transaction guard conflicts with live/internal GraphService execution."""
    return bool(
        await session.scalar(
            text("SELECT pg_try_advisory_xact_lock(hashtextextended(:key, 0))"),
            {"key": str(conversation_id)},
        )
    )
