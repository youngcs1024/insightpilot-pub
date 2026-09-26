"""Typed two-stage memory reads, selection decisions and bounded provenance."""

from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import Field

from app.schemas.mcp import Contract
from app.schemas.memory import StoredMemory
from app.schemas.metric_resolution import MetricKey, MetricPatches


class MemoryStage(StrEnum):
    PREPARE = "prepare"
    FINALIZE = "finalize"


class MemoryReason(StrEnum):
    SELECTED = "selected"
    WRONG_USER = "wrong_user"
    INACTIVE = "inactive"
    WRONG_STAGE = "wrong_stage"
    WRONG_ROUTE = "wrong_route"
    WRONG_METRIC = "wrong_metric"
    TERM_ABSENT = "term_absent"
    EXPLICIT_REGION = "explicit_region"
    UNKNOWN_REGION = "unknown_region"
    EXPLICIT_PATCH = "explicit_patch"
    COUNT_LIMIT = "count_limit"
    TOKEN_LIMIT = "token_limit"


class MemoryReadRequest(Contract):
    """Versioned MemoryReadRequest boundary for conservative retrieval."""

    schema_version: Literal[1] = 1
    user_id: UUID
    question: str = Field(max_length=32_000)
    stage: MemoryStage
    data_route: bool = False
    clarify: bool = False
    metric_keys: list[MetricKey] = Field(default_factory=list, max_length=6)
    explicit_patch: MetricPatches = Field(default_factory=MetricPatches)
    region_mentioned: bool | None = None


class MemoryDecision(Contract):
    """Versioned MemoryDecision boundary for conservative retrieval."""

    schema_version: Literal[1] = 1
    memory_id: UUID
    reason: MemoryReason
    score: float = Field(ge=0, le=1)


class MemorySelection(Contract):
    """Versioned MemorySelection boundary for conservative retrieval."""

    schema_version: Literal[1] = 1
    selected: list[StoredMemory] = Field(default_factory=list, max_length=5)
    decisions: list[MemoryDecision] = Field(default_factory=list)
    tokens: int = Field(default=0, ge=0, le=600)
    failed: bool = False
    failure_code: str | None = None
