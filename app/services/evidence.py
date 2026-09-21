"""Short transactions commit evidence before any answer generation begins."""

from uuid import UUID

import structlog

from app.agents.contracts import DataEvidence, EvidenceBundle, EvidenceSnapshot, TurnIdentity
from app.core.observability import record_evidence
from app.db.session import Database
from app.repositories.evidence import EvidenceRepository
from app.repositories.history import validate_turn
from app.repositories.turns import TurnRepository
from app.schemas.knowledge import KnowledgeEvidence

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

    async def read_bundle(self, identity: TurnIdentity) -> EvidenceBundle:
        """Read only committed snapshots for an owned turn."""
        async with self.database.session() as session:
            return await EvidenceRepository(session, identity).read_bundle()

    async def commit_bundle(
        self, identity: TurnIdentity, data: DataEvidence | None, knowledge: KnowledgeEvidence | None
    ) -> EvidenceBundle:
        """Commit both outputs atomically before any answer generation starts."""
        async with self.database.session() as session, session.begin():
            turns = TurnRepository(session, identity.user_id)
            await turns.lock_conversation(identity.conversation_id)
            await validate_turn(turns, identity)
            repo = EvidenceRepository(session, identity)
            bundle = EvidenceBundle(
                data=await repo.insert(data) if data is not None else None,
                knowledge=await repo.insert_knowledge(knowledge) if knowledge is not None else None,
            )
        if data is not None:
            record_evidence(data)
        logger.info(
            "evidence_bundle_committed",
            turn_id=str(identity.turn_id),
            has_data=bundle.data is not None,
            has_knowledge=bundle.knowledge is not None,
        )
        return bundle
