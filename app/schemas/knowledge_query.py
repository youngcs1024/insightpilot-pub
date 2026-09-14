"""Bounded knowledge-only context and typed query interpretation outcomes."""

from enum import StrEnum
from typing import Literal, Self
from uuid import UUID

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from app.schemas.mcp import Contract
from app.schemas.model_runtime import Text
from app.schemas.retrieval import KnowledgeTimeScope


class KnowledgeHistoryTurn(Contract):
    """Caller-projected topics, never raw messages, SQL or data evidence."""

    schema_version: Literal[1] = 1
    turn_id: UUID
    question: str = Field(min_length=1, max_length=300)
    answer_summary: str = Field(default="", max_length=300)
    time_scope: KnowledgeTimeScope


class KnowledgeClarificationKind(StrEnum):
    """Ambiguity is separate from unavailable services and absent evidence."""

    PERIOD_UNRESOLVED = "period_unresolved"
    REFERENCE_UNRESOLVED = "reference_unresolved"


class KnowledgeClarification(Contract):
    """User-safe next action, without model or exception prose."""

    schema_version: Literal[1] = 1
    kind: KnowledgeClarificationKind
    message: str = Field(min_length=1, max_length=1000)


class KnowledgeTimeResolution(Contract):
    """Only the time node writes this intermediate, before history resolution."""

    schema_version: Literal[1] = 1
    scope: KnowledgeTimeScope | None = None
    assumptions: list[Text] = Field(default_factory=list, max_length=24)
    needs_history: bool = False
    year_inferred: bool = False
    clarification: KnowledgeClarification | None = None


class KnowledgeRewrite(Contract):
    """The model selects supplied antecedents; it cannot manufacture calendar bounds."""

    schema_version: Literal[1] = 1
    standalone: Text
    referenced_turn_ids: list[UUID] = Field(default_factory=list, max_length=3)
    unresolved_references: list[Text] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def unique_references(self) -> Self:
        """One antecedent may only appear once."""
        if len(set(self.referenced_turn_ids)) != len(self.referenced_turn_ids):
            raise PydanticCustomError("knowledge_references", "Duplicate antecedents")
        return self
