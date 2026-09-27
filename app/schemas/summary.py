"""Detached application summary work and bounded model output."""

from pydantic import Field

from app.db.models.turn import TurnStatus
from app.schemas.mcp import Contract


class SummaryOutput(Contract):
    """Token limits are checked with the runtime counter before persistence."""

    summary: str = Field(min_length=1, max_length=32000)


class SummaryWork(Contract):
    """One chronological assistant/user pair and the compare-and-set precondition."""

    expected_seq: int = Field(ge=0)
    covered_seq: int = Field(gt=0)
    existing_summary: str
    latest_user: str
    latest_answer: str
    status: TurnStatus
