"""Bounded extraction contracts, distinct from persisted memory provenance."""

from typing import Literal

from pydantic import Field

from app.agents.contracts import Answer
from app.db.models.turn import TurnRole, TurnStatus
from app.schemas.mcp import Contract
from app.schemas.memory import MemoryPayload


class MemoryCandidate(MemoryPayload):
    """A proposed user preference still needs confidence and exact-quote gates."""

    evidence_quote: str = Field(min_length=1, max_length=32_000)


class MemoryExtraction(Contract):
    """Zero candidates is the normal successful extraction result."""

    schema_version: Literal[1] = 1
    candidates: list[MemoryCandidate] = Field(default_factory=list, max_length=16)


class MemoryExtractionInput(Contract):
    """Detached committed-turn projection; no ORM session crosses the boundary."""

    schema_version: Literal[1] = 1
    role: TurnRole
    status: TurnStatus
    user_message: str = Field(min_length=1, max_length=32_000)
    answer: Answer
