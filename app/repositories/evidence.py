"""Owned snapshot reads and immutable inserts; services own commit boundaries."""

import hashlib
import json
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.contracts import DataEvidence, EvidenceBundle, EvidenceSnapshot, KnowledgeSnapshot, TurnIdentity
from app.core.errors import ConflictError, NotFoundError
from app.db.models.evidence import DataEvidenceRecord, KnowledgeEvidenceRecord
from app.repositories.turns import TurnRepository
from app.schemas.knowledge import KnowledgeEvidence


def fingerprint(data: DataEvidence | KnowledgeEvidence) -> str:
    """Canonical content identity excludes persistence-generated fields."""
    return hashlib.sha256(
        json.dumps(
            data.model_dump(mode="json"), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


class EvidenceRepository:
    """Every access filters user_id and validates conversation ownership."""

    def __init__(self, session: AsyncSession, identity: TurnIdentity) -> None:
        self.session = session
        self.identity = identity

    async def find(self, snapshot_id: UUID | None = None) -> EvidenceSnapshot | None:
        """Read frozen evidence without touching any external source."""
        identity = self.identity
        if (
            await TurnRepository(self.session, identity.user_id).get(
                identity.conversation_id, identity.turn_id
            )
            is None
        ):
            raise NotFoundError()
        query = select(DataEvidenceRecord).where(
            DataEvidenceRecord.user_id == identity.user_id,
            DataEvidenceRecord.assistant_turn_id == identity.turn_id,
        )
        if snapshot_id is not None:
            query = query.where(DataEvidenceRecord.id == snapshot_id)
        record = await self.session.scalar(query)
        return (
            None
            if record is None
            else EvidenceSnapshot(id=record.id, data=DataEvidence.model_validate(record.payload))
        )

    async def insert(self, data: DataEvidence) -> EvidenceSnapshot:
        """Caller holds the conversation lock, serializing concurrent identical writes."""
        previous = await self.find()
        if previous is not None:
            if fingerprint(previous.data) != fingerprint(data):
                raise ConflictError("evidence differs from committed snapshot")
            return previous
        record = DataEvidenceRecord(
            user_id=self.identity.user_id,
            assistant_turn_id=self.identity.turn_id,
            content_sha256=fingerprint(data),
            payload=data.model_dump(mode="json"),
        )
        self.session.add(record)
        await self.session.flush()
        return EvidenceSnapshot(id=record.id, data=data)


    async def find_knowledge(self, snapshot_id: UUID | None = None) -> KnowledgeSnapshot | None:
        """Validate turn ownership even when it has no knowledge snapshot."""
        identity = self.identity
        if await TurnRepository(self.session, identity.user_id).get(
            identity.conversation_id, identity.turn_id
        ) is None:
            raise NotFoundError()
        query = select(KnowledgeEvidenceRecord).where(
            KnowledgeEvidenceRecord.user_id == identity.user_id,
            KnowledgeEvidenceRecord.assistant_turn_id == identity.turn_id,
        )
        if snapshot_id is not None:
            query = query.where(KnowledgeEvidenceRecord.id == snapshot_id)
        record = await self.session.scalar(query)
        return None if record is None else KnowledgeSnapshot(
            id=record.id, knowledge=KnowledgeEvidence.model_validate(record.payload)
        )

    async def insert_knowledge(self, knowledge: KnowledgeEvidence) -> KnowledgeSnapshot:
        """Reuse identical snapshots; never overwrite original source text."""
        previous = await self.find_knowledge()
        if previous is not None:
            if fingerprint(previous.knowledge) != fingerprint(knowledge):
                raise ConflictError("knowledge differs from committed snapshot")
            return previous
        record = KnowledgeEvidenceRecord(
            user_id=self.identity.user_id,
            assistant_turn_id=self.identity.turn_id,
            content_sha256=fingerprint(knowledge),
            payload=knowledge.model_dump(mode="json"),
        )
        self.session.add(record)
        await self.session.flush()
        return KnowledgeSnapshot(id=record.id, knowledge=knowledge)

    async def read_bundle(self) -> EvidenceBundle:
        """Read both kinds without consulting either external evidence source."""
        return EvidenceBundle(data=await self.find(), knowledge=await self.find_knowledge())
