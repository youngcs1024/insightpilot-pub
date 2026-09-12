"""Portable typed evidence and capacity arithmetic; no GPU imports or live I/O."""

import math
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field, FiniteFloat

from app.core.errors import InsightPilotError
from app.core.settings_base import ConfigModel

GIB = 1024**3
MIN_COMPARISON = 2
PROJECT_CEILING = 13_000_000_000
Bytes = Annotated[int, Field(ge=0)]
Vector = Annotated[list[FiniteFloat], Field(min_length=1024, max_length=1024)]


class Status(StrEnum):
    """Missing measurements are distinct from unsuccessful measurements."""

    PASSED = "passed"
    FAILED = "failed"
    PENDING = "pending"


class FailureKind(StrEnum):
    """Stable failure categories, never inferred from error prose."""

    PREREQUISITE = "prerequisite"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    TRANSPORT = "transport"
    CUDA = "cuda"
    OOM = "oom"
    HEADROOM = "headroom"
    OUTPUT = "output"
    SAMPLING = "sampling"
    ISOLATION = "isolation"


class ProbeError(InsightPilotError):
    """A bounded probe failed; the category is suitable for retry decisions."""

    code = "CAPACITY_PROBE_FAILED"

    def __init__(self, kind: FailureKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind


class CapacityResult(ConfigModel):
    """A local-only verdict based on measured bytes and current allocation."""

    status: Status
    docker_memtotal_bytes: Bytes
    project_peak_bytes: Bytes | None
    other_peak_bytes: Bytes | None
    headroom_bytes: Bytes = GIB
    required_bytes: Bytes | None = None
    recommended_allocation_bytes: Bytes | None = None
    ceiling_bytes: Bytes = PROJECT_CEILING


def assess_capacity(
    memtotal: int, project_peak: int | None, other_peak: int | None
) -> CapacityResult:
    """Keep the decimal project ceiling independent of binary Docker headroom."""
    result = CapacityResult(
        status=Status.PENDING,
        docker_memtotal_bytes=memtotal,
        project_peak_bytes=project_peak,
        other_peak_bytes=other_peak,
    )
    if project_peak is None or other_peak is None:
        return result
    required = project_peak + other_peak + GIB
    result.required_bytes = required
    result.recommended_allocation_bytes = math.ceil(required / GIB) * GIB
    result.status = (
        Status.PASSED if required <= memtotal and project_peak <= PROJECT_CEILING else Status.FAILED
    )
    return result


class ResourceIdentity(ConfigModel):
    """Stable identity/state only; uptime text changes even without intervention."""

    kind: Literal["container", "network", "volume"]
    id: str
    name: str
    project: str = ""
    state: str = ""


def verify_unchanged(before: list[ResourceIdentity], after: list[ResourceIdentity]) -> None:
    """Require every external resource to retain its original identity and state."""
    identities = {(item.kind, item.id): item for item in after}
    if any(identities.get((item.kind, item.id)) != item for item in before):
        raise ProbeError(FailureKind.ISOLATION, "An external resource changed during the probe.")


class MemorySample(ConfigModel):
    """Raw cgroup usage and cache-adjusted working set must not be conflated."""

    elapsed_s: Annotated[float, Field(ge=0)]
    container_id: str
    project: str
    usage_bytes: Bytes
    working_set_bytes: Bytes
    limit_bytes: Bytes


class TextBatch(ConfigModel):
    """Bounded diagnostic embedding request."""

    texts: Annotated[
        list[Annotated[str, Field(min_length=1, max_length=32000)]],
        Field(min_length=1, max_length=16),
    ]


class Pair(ConfigModel):
    """A fixed query/passage pair for capacity and numeric comparison."""

    query: str = Field(min_length=1, max_length=32000)
    passage: str = Field(min_length=1, max_length=32000)


class PairBatch(ConfigModel):
    """One rerank operation is twenty pairs at most."""

    pairs: Annotated[list[Pair], Field(min_length=1, max_length=20)]


class ModelIdentity(ConfigModel):
    """Actual model settings travel with every inference response."""

    embed_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    rerank_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    precision: Literal["fp16", "fp32"]
    embed_max_length: Literal[512] = 512
    rerank_max_length: Literal[320] = 320


class Stage(ConfigModel):
    """Synchronized server timings plus CUDA/cgroup high water marks."""

    name: str
    seconds: Annotated[FiniteFloat, Field(ge=0)]
    allocated_peak_bytes: Bytes
    reserved_peak_bytes: Bytes
    cgroup_peak_bytes: Bytes
    forward_batch_sizes: list[Annotated[int, Field(ge=1, le=16)]] = Field(default_factory=list)


class EmbedResponse(ConfigModel):
    """Validate dense and learned-sparse output before storing it."""

    identity: ModelIdentity
    dense: list[Vector]
    sparse: list[
        Annotated[
            dict[Annotated[int, Field(ge=0)], Annotated[FiniteFloat, Field(ge=0)]],
            Field(min_length=1),
        ]
    ]
    stage: Stage


class RerankResponse(ConfigModel):
    """Normalized scores stay in request order."""

    identity: ModelIdentity
    scores: list[Annotated[FiniteFloat, Field(ge=0, le=1)]]
    stage: Stage


class ReadyResponse(ConfigModel):
    """Successful warmup, not a live TCP listener, establishes readiness."""

    identity: ModelIdentity
    stages: list[Stage]
    device: str


class ErrorResponse(ConfigModel):
    """No exception prose, credentials or input text crosses HTTP."""

    kind: FailureKind


def percentile(values: list[float], fraction: float) -> float:
    """Use documented nearest-rank percentiles without adding a scientific dependency."""
    if not values or not 0 < fraction <= 1:
        raise ProbeError(FailureKind.OUTPUT, "Percentile needs observations and a valid fraction.")
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def ranks(values: list[float]) -> list[float]:
    """Average ties for the Spearman comparison."""
    ordered = sorted(values)
    return [
        (ordered.index(value) + len(ordered) - ordered[::-1].index(value) + 1) / 2
        for value in values
    ]


def spearman(left: list[float], right: list[float]) -> float:
    """Reject degenerate/nonfinite comparisons rather than manufacturing correlation."""
    if (
        len(left) != len(right)
        or len(left) < MIN_COMPARISON
        or not all(map(math.isfinite, left + right))
    ):
        raise ProbeError(FailureKind.OUTPUT, "Comparison needs matching finite observations.")
    a, b = ranks(left), ranks(right)
    a = [x - sum(a) / len(a) for x in a]
    b = [x - sum(b) / len(b) for x in b]
    denominator = math.sqrt(sum(x * x for x in a) * sum(x * x for x in b))
    if denominator == 0:
        raise ProbeError(FailureKind.OUTPUT, "Constant scores cannot establish rank correlation.")
    return sum(x * y for x, y in zip(a, b, strict=True)) / denominator
