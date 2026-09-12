"""Static ground truth and run artifacts never cross into model inputs."""

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from app.agents.contracts import ResolvedMetricBinding
from app.core.errors import ValidationError
from app.core.llm_config import ModelRoleSettings
from app.schemas.mcp import Contract, PolicyReason, QueryResultPayload
from app.services.periods import Period
from data.seed.contracts import Manifest

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "datasets/nl2sql/cases.yaml"
REFERENCE_TIME = datetime.fromisoformat("2026-09-09T12:00:00+08:00")


class EvaluationError(ValidationError):
    """Invalid evaluation evidence, distinct from an incorrect model answer."""

    code = "EVALUATION_INVALID"


class Comparison(StrEnum):
    """Closed result comparison policies."""

    SCALAR = "scalar_numeric"
    SET = "set_unordered"
    ORDERED = "ordered"
    EMPTY = "empty"


class ExpectedBinding(Contract):
    """Complete per-metric semantics, independently authored from business rules."""

    metric_key: str
    period: Period
    filters: list[str]
    date_field: str


class Case(Contract):
    """One ordinary question or fixed validator adversary."""

    id: str = Field(pattern=r"^nl2sql-[a-z0-9-]+$")
    question: str = Field(min_length=1, max_length=32000)
    traps: list[Literal["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"]] = Field(
        default_factory=list
    )
    expected_bindings: list[ExpectedBinding] = Field(default_factory=list)
    canonical_sql: str = Field(default="", max_length=32000)
    comparison: Comparison = Comparison.SCALAR
    tolerance: float = Field(default=0.001, ge=0, le=1)
    adversarial_sql: str = Field(default="", max_length=32000)
    expected_reasons: list[PolicyReason] = Field(default_factory=list)
    anchor_rows: list[list[str | int | float | bool | None]] | None = None

    @model_validator(mode="after")
    def distinct_paths(self) -> Self:
        """A safety probe can never be mistaken for an executable oracle."""
        if self.adversarial_sql:
            if self.canonical_sql or self.expected_bindings or not self.expected_reasons:
                raise EvaluationError("Invalid adversarial case")
        elif not self.canonical_sql or not self.expected_bindings or self.expected_reasons:
            raise EvaluationError("Ordinary cases require SQL and semantic expectations")
        keys = [binding.metric_key for binding in self.expected_bindings]
        if len(keys) != len(set(keys)):
            raise EvaluationError("Duplicate expected metric")
        return self


class Options(Contract):
    """Validated CLI configuration; no environment or credential duplication."""

    suite: Literal["nl2sql"] = "nl2sql"
    repeats: int = Field(default=1, ge=1, le=20)
    report: Path = ROOT / "reports"
    threshold_result_accuracy: float | None = Field(default=None, ge=0, le=1)
    seed_manifest: Path = ROOT.parent / "data/seed/out/manifest.json"


class Observation(Contract):
    """Detached graph measurements, including semantics on failed SQL paths."""

    bindings: list[ResolvedMetricBinding] = Field(default_factory=list)
    result: QueryResultPayload | None = None
    sql: str = ""
    corrections: int = Field(default=0, ge=0, le=2)
    failure_code: str | None = None


class CaseResult(Contract):
    """Every attempt is retained, including errors and clarification."""

    case_id: str
    repeat: int = Field(ge=1)
    traps: list[str]
    adversarial: bool
    execution_match: bool = False
    result_accuracy: bool = False
    metric_resolution_accuracy: bool = False
    unsafe_sql_blocked: bool = False
    evidence_valid: bool = True
    failure_code: str | None = None
    policy_reasons: list[PolicyReason] = Field(default_factory=list)
    observation: Observation | None = None
    canonical_result: QueryResultPayload | None = None


class ConfigSnapshot(Contract):
    """Explicit public allowlist, never a dump of application Settings."""

    model: str
    sql_role: ModelRoleSettings
    prompt_hashes: dict[str, str]
    catalog_versions: dict[str, int]
    catalog_hash: str
    schema_hash: str
    dataset_hash: str
    seed_manifest: Manifest
    git_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_dirty: bool
    reference_time: AwareDatetime = REFERENCE_TIME


class Score(Contract):
    """An absent denominator is unknown, never a perfect score."""

    passed: int = Field(ge=0)
    total: int = Field(ge=0)

    @property
    def rate(self) -> float | None:
        """Return the measured rate only when cases exist."""
        return self.passed / self.total if self.total else None


class Summary(Contract):
    """Separate denominators for business questions and validator probes."""

    execution_match: Score
    result_accuracy: Score
    metric_resolution_accuracy: Score
    unsafe_sql_block_rate: Score


class Report(Contract):
    """Versioned artifact containing every measured attempt and configuration."""

    schema_version: Literal[1] = 1
    run_id: str
    created_at: AwareDatetime
    config: ConfigSnapshot
    repeats: int = Field(ge=1)
    expected_attempts: int = Field(gt=0)
    results: list[CaseResult]
    summary: Summary
    per_trap: dict[str, Summary]
    per_repeat: dict[int, Summary]
    evidence_valid: bool
