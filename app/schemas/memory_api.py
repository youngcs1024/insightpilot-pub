"""Bounded, request-correlated memory history responses."""

from typing import Literal

from pydantic import Field

from app.schemas.auth import AuthResponse
from app.schemas.memory import StoredMemory


class MemoryPage(AuthResponse):
    """Owned stored versions with provenance and explicit supersession pointers."""

    schema_version: Literal[1] = 1
    items: list[StoredMemory] = Field(max_length=100)
    limit: int = Field(ge=1, le=100)
    offset: int = Field(ge=0)
