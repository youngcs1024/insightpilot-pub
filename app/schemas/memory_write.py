"""Versioned results for committed memory write decisions."""

from enum import StrEnum
from typing import Literal, Self
from uuid import UUID

from pydantic import model_validator
from pydantic_core import PydanticCustomError

from app.schemas.mcp import Contract


class WriteStatus(StrEnum):
    """Observable outcomes, never inferred from log prose."""

    CREATED = "created"
    DUPLICATE = "duplicate"
    SUPERSEDED = "superseded"


class WriteOutcome(Contract):
    """The retained/new memory and, for replacement only, its predecessor."""

    schema_version: Literal[1] = 1
    status: WriteStatus
    memory_id: UUID
    superseded_id: UUID | None = None

    @model_validator(mode="after")
    def valid_predecessor(self) -> Self:
        """Require a distinct predecessor precisely when a replacement was written."""
        if (self.status is WriteStatus.SUPERSEDED) != (self.superseded_id is not None) or (
            self.superseded_id == self.memory_id
        ):
            raise PydanticCustomError("memory_write_outcome", "Invalid memory predecessor")
        return self
