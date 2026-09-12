"""Short transactions commit evidence before any answer generation begins."""

from uuid import UUID

import structlog

from app.agents.contracts import DataEvidence, EvidenceSnapshot, TurnIdentity
from app.core.observability import record_evidence
from app.db.session import Database
from app.repositories.evidence import EvidenceRepository
from app.repositories.history import validate_turn
from app.repositories.turns import TurnRepository

logger = structlog.get_logger(__name__)


class EvidenceService:
    """Persist immutable snapshots under the existing conversation running guard."""

    def __init__(self, database: Database) -> None:
        self.database = database

    async def find(
        self, identity: TurnIdentity, snapshot_id: UUID | None = None
    ) -> EvidenceSnapshot | None:
        """Historical reads remain valid even when the turn is no longer running."""
        async with self.database.session() as session:
            return await EvidenceRepository(session, identity).find(snapshot_id)

    async def commit(self, identity: TurnIdentity, data: DataEvidence) -> EvidenceSnapshot:
        """An identical retry returns its committed ID; changed content conflicts."""
        async with self.database.session() as session, session.begin():
            turns = TurnRepository(session, identity.user_id)
            await turns.lock_conversation(identity.conversation_id)
            await validate_turn(turns, identity)
            snapshot = await EvidenceRepository(session, identity).insert(data)
        logger.info(
            "evidence_committed", turn_id=str(identity.turn_id), snapshot_id=str(snapshot.id)
        )
        record_evidence(data)
        return snapshot
