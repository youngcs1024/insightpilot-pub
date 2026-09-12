"""Owned snapshot reads and immutable inserts; services own commit boundaries."""

import hashlib
import json
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.contracts import DataEvidence, EvidenceSnapshot, TurnIdentity
from app.core.errors import ConflictError, NotFoundError
from app.db.models.evidence import DataEvidenceRecord
from app.repositories.turns import TurnRepository


def fingerprint(data: DataEvidence) -> str:
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
