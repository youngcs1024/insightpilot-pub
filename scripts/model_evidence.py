"""Versioned raw measurements for the dedicated model acceptance workload."""

import statistics
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, NonNegativeInt, model_validator
from pydantic_core import PydanticCustomError

from app.schemas.model_runtime import ModelMetadata, ModelResponse, RerankResult, Score

RERANK_P50_SECONDS = 2.0
PAIR_COUNT = 50
GROUP_COUNT = 5
CANDIDATE_COUNT = 20
CONTROL_COUNT = 2
DEFAULT_BATCH = 16
EMBED_LENGTH = 512
RERANK_LENGTH = 320
Seconds = Annotated[FiniteFloat, Field(ge=0)]


class Evidence(BaseModel):
    """Old or silently extended artifacts cannot count as current acceptance."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


class Provenance(Evidence):
    """Operator-observed deployment identity, not a claim made by HTTP readiness."""

    source_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    image_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    gpu_uuid: str = Field(pattern=r"^GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$")


class CallMeasurement(Evidence):
    """Retain effective settings even when the server recovers with a smaller batch."""

    request_id: str = Field(min_length=1)
    metadata: ModelMetadata
    client_seconds: Seconds
    server_ms: NonNegativeInt
    queue_ms: NonNegativeInt
    inference_ms: NonNegativeInt

    @classmethod
    def from_response(cls, result: ModelResponse, elapsed: float) -> Self:
        """Copy only safe response metadata, never embedded source text or vectors."""
        return cls(
            request_id=result.request_id,
            metadata=result.metadata.model_copy(deep=True),
            client_seconds=elapsed,
            server_ms=result.ms,
            queue_ms=result.queue_ms,
            inference_ms=result.inference_ms,
        )


class RerankMeasurement(Evidence):
    """Ordered synthetic scores paired with their own effective execution settings."""

    call: CallMeasurement
    scores: list[Score] = Field(min_length=1, max_length=256)

    @classmethod
    def from_response(cls, result: RerankResult, elapsed: float) -> Self:
        """Detach the scored response from mutable client objects."""
        return cls(
            call=CallMeasurement.from_response(result, elapsed), scores=list(result.scores)
        )


class Benchmark(Evidence):
    """Raw evidence is authoritative; acceptance and percentiles are recomputed."""

    schema_version: Literal[2] = 2
    provenance: Provenance
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata: ModelMetadata
    embedding: CallMeasurement
    relevance: RerankMeasurement
    precision_calls: list[RerankMeasurement] = Field(min_length=5, max_length=5)
    latency_calls: list[RerankMeasurement] = Field(min_length=50, max_length=50)

    @model_validator(mode="after")
    def complete_workload(self) -> Self:
        """Every score has an unambiguous position in the fixed workload."""
        if (
            len(self.relevance.scores) != CONTROL_COUNT
            or any(len(item.scores) != PAIR_COUNT // GROUP_COUNT for item in self.precision_calls)
            or any(len(item.scores) != CANDIDATE_COUNT for item in self.latency_calls)
        ):
            raise PydanticCustomError("incomplete_workload", "Incomplete model workload")
        return self

    @property
    def scores(self) -> list[float]:
        """Undo query grouping to recover the original fifty-pair order."""
        return [
            self.precision_calls[index % GROUP_COUNT].scores[index // GROUP_COUNT]
            for index in range(PAIR_COUNT)
        ]

    @property
    def p50_s(self) -> float:
        """Median of all fifty client wall times, including SSH transport."""
        return statistics.median(item.call.client_seconds for item in self.latency_calls)

    @property
    def p95_s(self) -> float:
        """Nearest-rank 95th percentile for the fixed fifty-request workload."""
        return sorted(item.call.client_seconds for item in self.latency_calls)[47]

    @property
    def stable_settings(self) -> bool:
        """A recovered or reconfigured call cannot masquerade as the ready configuration."""
        calls = [self.embedding, self.relevance.call]
        calls.extend(item.call for item in (*self.precision_calls, *self.latency_calls))
        return all(item.metadata == self.metadata for item in calls)

    @property
    def accepted(self) -> bool:
        """The default workload, positive control and precision-specific latency must pass."""
        defaults = (
            self.metadata.embed_batch == DEFAULT_BATCH
            and self.metadata.rerank_batch == DEFAULT_BATCH
            and self.metadata.embed_max_length == EMBED_LENGTH
            and self.metadata.rerank_max_length == RERANK_LENGTH
        )
        return (
            defaults
            and self.stable_settings
            and self.relevance.scores[0] > self.relevance.scores[1]
            and (self.metadata.precision == "fp32" or self.p50_s <= RERANK_P50_SECONDS)
        )
