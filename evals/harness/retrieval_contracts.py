"""Reproducible Suite B inputs, observations and public selection artifacts."""

from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import Field

from app.retrieval.config import FilterConfig, RetrievalConfig
from app.schemas.ingestion import Digest
from app.schemas.mcp import Contract
from app.schemas.retrieval import KnowledgeTimeScope, ObservedRetrieval
from evals.harness.ir_metrics import IRMetrics
from scripts.model_evidence import Provenance

ROOT = Path(__file__).resolve().parents[1] / "datasets/retrieval"


class Split(StrEnum):
    """Partitions assigned by semantic group before any tuning."""

    DEVELOPMENT = "development"
    FROZEN = "frozen"


class Arm(StrEnum):
    """Stable tie order; FP32 is a control, never a selectable ninth strategy."""

    A = "A"
    A_RERANK = "A-rerank"
    B = "B"
    B_RERANK = "B-rerank"
    C = "C"
    C_RERANK = "C-rerank"
    D = "D"
    D_RERANK = "D-rerank"
    D_FP32 = "D-fp32"


def arm_config(arm: Arm, filtering: FilterConfig | None = None) -> RetrievalConfig:
    """Only arm membership and explicit development-selected thresholds vary."""
    return RetrievalConfig(
        use_sparse_learned=arm in {Arm.B, Arm.B_RERANK, Arm.D, Arm.D_RERANK, Arm.D_FP32},
        use_bm25=arm in {Arm.C, Arm.C_RERANK, Arm.D, Arm.D_RERANK, Arm.D_FP32},
        use_rerank=arm in {Arm.A_RERANK, Arm.B_RERANK, Arm.C_RERANK, Arm.D_RERANK, Arm.D_FP32},
        record_arm_scores=True,
        filtering=filtering or FilterConfig(),
    )


class RetrievalCase(Contract):
    """Question and explicit business time, with no labels in generation inputs."""

    id: str = Field(pattern=r"^r-\d{3}$")
    text: str = Field(min_length=1, max_length=32000)
    time_scope: KnowledgeTimeScope
    semantic_group_id: str = Field(min_length=1)
    split: Split
    category: str = Field(min_length=1)
    notes: str = Field(min_length=1)
    answerable: bool = True


class Queries(Contract):
    """The complete preassigned query inventory."""

    queries: list[RetrievalCase]


class Judgment(Contract):
    """Explicit original-text review, never a model score converted to a label."""

    chunk_id: UUID
    grade: int = Field(strict=True, ge=0, le=3)
    why: str = Field(min_length=1)


class Judgments(Contract):
    """One query's exhaustive or pooled judgments."""

    query_id: str
    judgments: list[Judgment] = Field(min_length=8)


class JudgmentPartition(Contract):
    """Physical separation prevents the tuner from opening frozen labels."""

    split: Split
    reviewed_by: Literal["agent_authorized_by_user"] = "agent_authorized_by_user"
    records: list[Judgments]


class ChunkReview(Contract):
    """Metadata for checking temporal grades independently of retrieval output."""

    chunk_id: UUID
    source_path: str
    content_sha256: Digest
    effective_from: date | None
    effective_to: date | None


class DatasetManifest(Contract):
    """Hashes bind the dataset to source/chunk identities without machine paths."""

    schema_version: Literal[1] = 1
    corpus_version: Digest
    chunking_version: Digest
    queries_sha256: Digest
    judgments_sha256: dict[Split, Digest]
    chunk_ids: list[UUID]
    chunks: list[ChunkReview]
    review_policy: Literal["agent_authorized_by_user"] = "agent_authorized_by_user"


class Dataset(Contract):
    """Only the selected partition's labels reach the evaluator."""

    manifest: DatasetManifest
    cases: list[RetrievalCase]
    labels: list[Judgments]
    identity: Digest


class Attempt(Contract):
    """Every attempt survives failures; payloads also support exact precision replay."""

    query_id: str
    arm: Arm
    observed: ObservedRetrieval | None = None
    failure_code: str | None = None


class Measurements(Contract):
    """Raw phase artifact, kept private; all input and deployment identities are explicit."""

    schema_version: Literal[1] = 1
    client_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_dirty: bool
    dataset_identity: Digest
    corpus_version: Digest
    split: Split
    server: Provenance
    attempts: list[Attempt]
    selection_commit: str | None = None
    selection_identity: Digest | None = None


class ScoredAttempt(Contract):
    """Invalid attempts have no fabricated numerical success."""

    query_id: str
    arm: Arm
    answerable: bool
    ranking: IRMetrics | None = None
    filtered: IRMetrics | None = None
    abstained: bool | None = None
    wrong_version_count: int = Field(default=0, ge=0)
    failure_code: str | None = None


class Selection(Contract):
    """Written and committed after development, before opening frozen labels."""

    schema_version: Literal[1] = 1
    dataset_identity: Digest
    development_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    arm: Arm
    config: RetrievalConfig
    rationale: str = Field(min_length=1)


class Percentiles(Contract):
    """No samples means unknown; never a zero-time successful measurement."""

    count: int = Field(ge=0)
    p50: float | None
    p95: float | None


class ArmSummary(Contract):
    """Answerable IR means and negative controls have independent denominators."""

    arm: Arm
    attempted: int
    valid: int
    answerable: int
    ranking: IRMetrics
    filtered: IRMetrics
    negative_count: int
    correct_abstentions: int
    false_evidence: int
    latency_ms: dict[str, Percentiles]


class AblationReport(Contract):
    """Derived from complete raw attempts; no trusted saved passed boolean."""

    measurements: Measurements
    selection: Selection | None
    selection_commit: str | None = None
    results: list[ScoredAttempt]
    summaries: list[ArmSummary]
    evidence_valid: bool
    issues: list[str]
