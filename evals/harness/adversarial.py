"""Authored SQL probes shared by Suite A, MCP regressions and the offline gate."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import sqlglot
import yaml  # type: ignore[import-untyped]
from pydantic import Field, TypeAdapter, ValidationError, model_validator
from sqlglot import exp

from app.core.errors import SchemaMetadataError
from app.core.sql_policy.sql_validator import SQLValidator
from app.schemas.mcp import Contract, PolicyReason, ValidationStatus
from data.seed.schema_metadata_loader import check_keys
from evals.harness.contracts import Case, EvaluationError, Score

CASES = Path(__file__).resolve().parents[1] / "datasets/nl2sql/adversarial.yaml"
MIN_REJECTIONS = 17
MIN_REWRITES = 2


class AdversarialCase(Contract):
    """One expected policy rejection or semantics-preserving outer cap."""

    id: str = Field(pattern=r"^nl2sql-(unsafe|rewrite)-[0-9]{3}$")
    sql: str = Field(min_length=1, max_length=32000)
    expected_reasons: list[PolicyReason] = Field(default_factory=list)
    database_read_only: bool = False
    max_rows: int = Field(default=1000, ge=1, le=5000)
    expected_limit: int | None = Field(default=None, ge=2, le=5001)

    @model_validator(mode="after")
    def one_outcome(self) -> "AdversarialCase":
        """A safe rewrite cannot silently enter the rejection denominator."""
        rejecting = self.id.startswith("nl2sql-unsafe-")
        if rejecting != bool(self.expected_reasons) or rejecting == (
            self.expected_limit is not None
        ):
            raise EvaluationError("Ambiguous adversarial SQL expectation")
        if not rejecting and (self.database_read_only or self.expected_limit != self.max_rows + 1):
            raise EvaluationError("Invalid safe rewrite expectation")
        return self

    def suite_a_case(self) -> Case:
        """Only true rejections become Suite A unsafe-SQL probes."""
        if not self.expected_reasons:
            raise EvaluationError("Safe rewrite cannot count as a blocked attack")
        return Case(
            id=self.id,
            question="固定危险 SQL 校验探针",
            adversarial_sql=self.sql,
            expected_reasons=list(self.expected_reasons),
        )


class AdversarialResult(Contract):
    """A measured validator outcome, including a mismatched reason or cap."""

    case_id: str
    passed: bool
    status: ValidationStatus
    reasons: list[PolicyReason]
    limit_applied: bool


class AdversarialReport(Contract):
    """Offline evidence with a stable denominator and source identity."""

    run_id: str
    created_at: datetime
    dataset_hash: str
    results: list[AdversarialResult]
    blocked: Score
    rewrites: Score


def load_adversaries(path: Path = CASES) -> list[AdversarialCase]:
    """Reject malformed, duplicate or incomplete authored safety suites."""
    try:
        source = path.read_text(encoding="utf-8")
        node = yaml.compose(source)
        if node is not None:
            check_keys(node)
        cases = TypeAdapter(list[AdversarialCase]).validate_python(yaml.safe_load(source))
    except (OSError, yaml.YAMLError, ValidationError, SchemaMetadataError) as exc:
        raise EvaluationError("Invalid adversarial SQL dataset") from exc
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise EvaluationError("Duplicate adversarial SQL case")
    if (
        sum(bool(case.expected_reasons) for case in cases) < MIN_REJECTIONS
        or sum(case.expected_limit is not None for case in cases) < MIN_REWRITES
    ):
        raise EvaluationError("Incomplete adversarial SQL dataset")
    return cases


def _outer_limit(sql: str) -> int | None:
    """Inspect the rewritten AST instead of matching SQL text."""
    parsed = sqlglot.parse_one(sql, dialect="postgres")
    limit = parsed.args.get("limit")
    expression = limit.expression if isinstance(limit, exp.Limit) else None
    if isinstance(expression, exp.Literal) and expression.is_int:
        return int(expression.this)
    return None


def evaluate(cases: list[AdversarialCase], path: Path = CASES) -> AdversarialReport:
    """Run every fixed probe without opening a model or database connection."""
    validator = SQLValidator()
    results = []
    for case in cases:
        outcome = validator.validate(case.sql, result_cap=case.max_rows)
        if case.expected_reasons:
            passed = outcome.status is not ValidationStatus.VALID and (
                outcome.reasons == case.expected_reasons
            )
        else:
            passed = (
                outcome.status is ValidationStatus.VALID
                and outcome.limit_applied
                and _outer_limit(outcome.rewritten_sql) == case.expected_limit
            )
        results.append(
            AdversarialResult(
                case_id=case.id,
                passed=passed,
                status=outcome.status,
                reasons=outcome.reasons,
                limit_applied=outcome.limit_applied,
            )
        )
    rejecting = [r for r, c in zip(results, cases, strict=True) if c.expected_reasons]
    rewriting = [r for r, c in zip(results, cases, strict=True) if not c.expected_reasons]
    return AdversarialReport(
        run_id=uuid4().hex,
        created_at=datetime.now(UTC),
        dataset_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
        results=results,
        blocked=Score(passed=sum(r.passed for r in rejecting), total=len(rejecting)),
        rewrites=Score(passed=sum(r.passed for r in rewriting), total=len(rewriting)),
    )


def exit_code(report: AdversarialReport, threshold: float) -> int:
    """Missing attempts and any failed rewrite invalidate the offline gate."""
    rejecting = [r for r in report.results if r.case_id.startswith("nl2sql-unsafe-")]
    rewriting = [r for r in report.results if r.case_id.startswith("nl2sql-rewrite-")]
    return int(
        len(report.results) != report.blocked.total + report.rewrites.total
        or len({result.case_id for result in report.results}) != len(report.results)
        or len(rejecting) != report.blocked.total
        or len(rewriting) != report.rewrites.total
        or sum(r.passed for r in rejecting) != report.blocked.passed
        or sum(r.passed for r in rewriting) != report.rewrites.passed
        or report.blocked.total < MIN_REJECTIONS
        or report.rewrites.total < MIN_REWRITES
        or report.blocked.rate is None
        or report.blocked.rate < threshold
        or report.rewrites.rate != 1.0
    )
