"""Suite C ground truth, measurements and provenance are separate typed contracts."""

import hashlib
from enum import StrEnum
from pathlib import Path
from typing import Final, Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from app.agents.contracts import Route, RouteDecision, RoutingContext
from app.core.llm_config import ModelRoleSettings
from app.core.routing import RoutingStrategy
from app.schemas.mcp import Contract
from evals.harness.contracts import EvaluationError, Score

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "evals/datasets/routing/cases.yaml"
SELECTION = CASES.with_name("selected.yaml")
ROUTES = tuple(Route)
MIN_REPEATS = 3
MAX_BOTH_MISROUTE = 0.05
CONCURRENCY: Final = 4


class Split(StrEnum):
    """Semantic groups are assigned before any quality measurement."""

    DEVELOPMENT = "development"
    FROZEN = "frozen"


class Case(Contract):
    """Only question and context may cross into the classifier."""

    id: str = Field(pattern=r"^route-\d{3}$")
    question: str = Field(min_length=1, max_length=32000)
    expected: Route
    difficulty: Literal["easy", "medium", "hard"]
    rationale: str = Field(min_length=1)
    semantic_group_id: str = Field(min_length=1)
    split: Split
    deliberately_ambiguous: bool
    clarification_kind: Literal["ambiguous", "out_of_scope"] | None = None
    reviewed_by: Literal["agent"]
    routing_context: RoutingContext = Field(default_factory=RoutingContext)

    @model_validator(mode="after")
    def clarification_ground_truth(self) -> Self:
        """Separate ambiguity from deliberately unsupported actions."""
        if (self.expected is Route.CLARIFY) != (self.clarification_kind is not None):
            raise EvaluationError("Clarification cases require an explicit ground-truth kind")
        return self


class Options(Contract):
    """Live evaluation is explicit and ordinary tests never invoke it."""

    suite: Literal["routing"] = "routing"
    repeats: int = Field(default=3, ge=1, le=20)
    report: Path = ROOT / "evals/reports"
    split: Split = Split.DEVELOPMENT
    threshold_accuracy: float = Field(default=0.90, ge=0, le=1)


class Observation(Contract):
    """None is a rules-only abstention or failure, never an invented clarify result."""

    decision: RouteDecision | None = None
    prefilter_hit: bool = False
    tokens: int | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0)
    failure_code: str | None = None


class Attempt(Contract):
    """Retain every case/arm/repeat, including provider failures."""

    case_id: str
    arm: RoutingStrategy
    repeat: int = Field(ge=1)
    observation: Observation


class Snapshot(Contract):
    """Public allowlist: credentials and transport URLs are never serialized."""

    git_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    source_dirty: bool
    dataset_hash: str
    source_hash: str
    prompt_hashes: dict[str, str]
    capability_hashes: list[str]
    model: ModelRoleSettings
    min_confidence: float = Field(ge=0, le=1)
    timeout_s: float = Field(gt=0)
    production_strategy: RoutingStrategy
    concurrency: Literal[4] = CONCURRENCY

    def fingerprint(self) -> str:
        """Bind evaluated inputs while allowing a subsequent selection-only commit."""
        return hashlib.sha256(
            self.model_dump_json(exclude={"git_sha", "source_dirty"}).encode()
        ).hexdigest()


class Selection(Contract):
    """Committed development choice required before frozen inference."""

    schema_version: Literal[1] = 1
    arm: Literal[RoutingStrategy.HYBRID, RoutingStrategy.LLM_ONLY]
    development_run_id: str
    development_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    development_report_hash: str
    fingerprint: str
    rule: Literal["accuracy_first_no_misroute_regression"] = "accuracy_first_no_misroute_regression"


class Measurements(Contract):
    """Raw run artifact, with a fixed denominator and post-run source recheck."""

    schema_version: Literal[1] = 1
    run_id: str
    created_at: AwareDatetime
    split: Split
    repeats: int = Field(ge=1, le=20)
    case_ids: list[str]
    config: Snapshot
    attempts: list[Attempt]
    selection: Selection | None = None
    selection_commit: str | None = None
    complete: bool = True


class ClassScore(Contract):
    """Precision and recall have independent, explicit denominators."""

    precision: Score
    recall: Score


class Metrics(Contract):
    """Non-decisions are separately counted outside the four predicted classes."""

    accuracy: Score
    misroute: Score
    both_misroute: Score
    clarification_precision: Score
    out_of_scope_recall: Score
    prefilter_hit_rate: Score
    prefilter_accuracy: Score
    per_class: dict[Route, ClassScore]
    confusion_matrix: list[list[int]]
    no_decision_by_class: dict[Route, int]
    mean_tokens: float | None
    token_measurements: int
    mean_latency_ms: float


class Distribution(Contract):
    """Sample standard deviation over repeats; one repeat cannot estimate variance."""

    mean: float | None
    sample_stddev: float | None


class ArmReport(Contract):
    """Overall metrics, per-repeat metrics and their variation."""

    overall: Metrics
    per_repeat: dict[int, Metrics]
    variation: dict[str, Distribution]


class Report(Contract):
    """Derived metrics remain attached to their complete raw evidence."""

    measurements: Measurements
    arms: dict[RoutingStrategy, ArmReport]
    evidence_valid: bool
    identical_arm_accuracy: bool
