"""Versioned public conversation, turn and SSE contracts."""

from datetime import datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticCustomError

from app.agents.contracts import Answer, EvidenceRefs, EvidenceSnapshot, KnowledgeSnapshot
from app.agents.failures import FailureKind
from app.db.models import TurnRole, TurnStatus
from app.schemas.auth import AuthResponse
from app.schemas.metric_resolution import MetricClarification


class ChatResponse(AuthResponse):
    """Every chat response is versioned and request-correlated."""

    model_config = ConfigDict(from_attributes=True)
    schema_version: Literal[1] = 1


class ConversationCreate(BaseModel):
    """An omitted title is left empty for future automatic naming."""

    title: str = Field(default="", max_length=200)


class ConversationResponse(ChatResponse):
    """Owned conversation metadata."""

    id: UUID
    title: str
    created_at: datetime
    updated_at: datetime
    archived_at: datetime | None


class ConversationPage(ChatResponse):
    """Stable descending creation order."""

    items: list[ConversationResponse]
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)


class MessageRequest(BaseModel):
    """Unescaped user text; preserve SQL and comparison operators."""

    content: str = Field(min_length=1, max_length=32_000)


class TurnResponse(ChatResponse):
    """Durable assistant result or user message, without internal diagnostics."""

    id: UUID
    conversation_id: UUID
    reply_to_turn_id: UUID | None
    seq: int
    role: TurnRole
    content: str
    status: TurnStatus
    failure_reason: FailureKind | None
    trace_id: str | None
    latency_ms: int | None
    answer: Answer | None = None
    clarification: MetricClarification | None = None
    evidence_refs: EvidenceRefs = Field(default_factory=EvidenceRefs)
    replayed: bool = False

    @model_validator(mode="after")
    def separate_clarification(self) -> Self:
        """Clarification responses have prose but no analytical evidence."""
        if self.clarification is not None and (
            (self.answer is not None and (
                not self.answer.abstained or self.answer.claims or self.answer.citations
                or self.answer.sql or self.answer.assumptions
                or self.answer.evidence_refs != EvidenceRefs()
            ))
            or self.evidence_refs.data_snapshot_id is not None
            or self.evidence_refs.knowledge_snapshot_id is not None
        ):
            raise PydanticCustomError("turn_clarification", "Clarification cannot contain analysis")
        return self


class TurnPage(ChatResponse):
    """Ordered persisted messages."""

    items: list[TurnResponse]
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)


class EvidenceResponse(ChatResponse):
    """Historical snapshots; an existing turn may have no evidence."""

    turn_id: UUID
    data: EvidenceSnapshot | None
    knowledge: KnowledgeSnapshot | None = None


class TokenEvent(BaseModel):
    """A delta of already validated and committed answer text."""

    delta: str


class HeartbeatEvent(ChatResponse):
    """A keepalive contains no uncommitted answer text."""


class ErrorEvent(BaseModel):
    """Terminal stream failures contain only a code and fixed safe message."""

    code: str
    message: str
