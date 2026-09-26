"""Input to the single current-turn data intent interpretation."""

from typing import Literal

from pydantic import Field

from app.schemas.mcp import Contract
from app.schemas.memory import TerminologyProjection
from app.schemas.metric_resolution import MetricKey


class MetricIntentInput(Contract):
    """Versioned MetricIntentInput boundary for conservative retrieval."""

    schema_version: Literal[1] = 1
    question: str = Field(min_length=1, max_length=32_000)
    data_intent: str = Field(default="", max_length=32_000)
    metric_hints: list[MetricKey] = Field(default_factory=list, max_length=6)
    terminology: list[TerminologyProjection] = Field(default_factory=list, max_length=5)
