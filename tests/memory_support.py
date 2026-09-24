"""Memory test inputs with real application provenance and no service side effects."""

from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.memory import MemoryCreate, MemoryType
from tests import factories


async def memory_owner(session: AsyncSession) -> tuple[UUID, UUID]:
    """Persist a user, conversation and successful source turn in the caller transaction."""
    user = factories.user()
    session.add(user)
    await session.flush()
    conversation = factories.conversation(user.id)
    session.add(conversation)
    await session.flush()
    turn = factories.turn(conversation.id)
    session.add(turn)
    await session.flush()
    return user.id, turn.id


def memory_input(
    turn_id: UUID,
    *,
    kind: MemoryType = MemoryType.TERMINOLOGY,
    content: dict[str, object] | None = None,
) -> MemoryCreate:
    """Build a bounded durable preference; extraction policy is deliberately absent."""
    return MemoryCreate.model_validate(
        {
            "source_turn_id": turn_id,
            "memory_type": kind,
            "content": content if content is not None else {"term": "大促", "means": "618"},
            "summary": "Saved preference",
            "confidence": 0.9,
        }
    )


def raw_memory(user_id: UUID, turn_id: UUID) -> dict[str, object]:
    """Use raw inserts only to prove database constraints independently of Pydantic."""
    value = memory_input(turn_id)
    return {
        "id": uuid4(),
        "user_id": user_id,
        "source_turn_id": turn_id,
        "memory_type": value.memory_type,
        "content": value.content.model_dump(mode="json"),
        "summary": value.summary,
        "confidence": value.confidence,
    }
