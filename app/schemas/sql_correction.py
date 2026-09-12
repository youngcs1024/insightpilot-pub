"""Typed correction decisions, independent of model explanation text."""

from enum import StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from app.schemas.mcp import Contract

MAX_CORRECTIONS = 2


class CorrectionStatus(StrEnum):
    """A new validation/execution failure resets pending validation to idle."""

    IDLE = "idle"
    PENDING_VALIDATION = "pending_validation"
    TERMINAL = "terminal"


class CorrectionRoute(StrEnum):
    """Destinations for the Step 2.11 assembler; no topology is installed here."""

    CORRECT_SQL = "correct_sql"
    VALIDATE_SQL = "validate_sql"
    PACKAGE_FAILURE = "package_failure"
    NO_CORRECTION = "no_correction"


class CorrectionDecision(StrEnum):
    """The model must explicitly decline when it cannot preserve meaning."""

    CORRECTED = "corrected"
    CANNOT_CORRECT = "cannot_correct"


class CorrectionStopReason(StrEnum):
    """Safe diagnostic categories, never substring matches on error prose."""

    BUDGET_EXHAUSTED = "budget_exhausted"
    IDENTICAL_SQL = "identical_sql"
    MODEL_DECLINED = "model_declined"
    SEMANTICS_UNPROVEN = "semantics_unproven"
    OPERATION_FAILED = "operation_failed"


class SqlCorrectionOutput(Contract):
    """Bounded structured response; refusal cannot also supply executable SQL."""

    schema_version: Literal[1] = 1
    decision: CorrectionDecision
    sql: str = Field(default="", max_length=32000)

    @model_validator(mode="after")
    def consistent_decision(self) -> Self:
        """Require exactly the SQL presence indicated by the typed decision."""
        if bool(self.sql.strip()) != (self.decision is CorrectionDecision.CORRECTED):
            raise PydanticCustomError("correction_decision", "SQL does not match decision")
        return self
