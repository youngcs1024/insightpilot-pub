"""Write decisions inside the caller's user-locked batch transaction."""

from uuid import UUID

from app.core.errors import ConflictError
from app.repositories.memory import MemoryRepository
from app.schemas.memory import MemoryCreate
from app.schemas.memory_extraction import MemoryCandidate
from app.schemas.memory_write import WriteOutcome, WriteStatus
from app.services.memory.dedup import MemoryMatch, compare


async def write(candidate: MemoryCandidate, repo: MemoryRepository, turn_id: UUID) -> WriteOutcome:
    """Touch a duplicate or append a version; the caller owns locking and rollback."""
    existing = await repo.list_active(candidate.memory_type)
    matches = [(row, compare(candidate, row)) for row in existing]
    matches = [(row, match) for row, match in matches if match is not MemoryMatch.DISTINCT]
    if len(matches) > 1:
        raise ConflictError("Multiple active memories for one logical key")
    if matches and matches[0][1] is MemoryMatch.DUPLICATE:
        duplicate = await repo.touch(matches[0][0].id)
        return WriteOutcome(status=WriteStatus.DUPLICATE, memory_id=duplicate.id)
    value = MemoryCreate(
        source_turn_id=turn_id,
        memory_type=candidate.memory_type,
        content=candidate.content,
        summary=candidate.summary,
        confidence=candidate.confidence,
    )
    new = await repo.create(value)
    if not matches:
        return WriteOutcome(status=WriteStatus.CREATED, memory_id=new.id)
    old = matches[0][0]
    await repo.supersede(old.id, by=new.id)
    return WriteOutcome(status=WriteStatus.SUPERSEDED, memory_id=new.id, superseded_id=old.id)
