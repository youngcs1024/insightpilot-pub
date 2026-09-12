"""Advisory result observations, independent of SQL failure and retry contracts."""

from enum import StrEnum
from typing import Literal

from pydantic import Field

from app.schemas.mcp import Contract


class SanityFlag(StrEnum):
    """Nonfatal observations retained in evidence; never correction eligibility."""

    EMPTY_RESULT = "empty_result"
    ALL_NULL = "all_null"
    SINGLE_NULL_SCALAR = "single_null_scalar"
    TRUNCATED = "truncated"
    SUSPICIOUS_ZERO = "suspicious_zero"
    EXTREME_MAGNITUDE = "extreme_magnitude"
    NEGATIVE_MONEY = "negative_money"
    CARDINALITY_SPIKE = "cardinality_spike"


class SanityCheckResult(Contract):
    """A failed check preserves partial observations without failing the query."""

    schema_version: Literal[1] = 1
    flags: list[SanityFlag] = Field(default_factory=list, max_length=8)
    check_failed: bool = False
