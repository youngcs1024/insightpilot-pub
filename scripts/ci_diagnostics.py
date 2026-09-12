"""SHA/run-bound diagnostic records; raw workflow outcomes remain the acceptance gate."""

import argparse
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, get_args

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from scripts.ci_dependencies import Assessment
from scripts.ci_evidence import CheckEvidence
from scripts.ci_junit import CaseOutcome, read_junit
from scripts.ci_partitions import PARTITIONS
from scripts.ci_policy import Plan, Result, Results
from scripts.ci_result import StepResult, case_label, job_description, migration_assessment
from scripts.ci_types import TypeReport
from scripts.ci_types import describe as describe_types

Stage = Literal[
    "quality",
    "unit",
    "integration",
    "storage",
    "migrations",
    "coverage",
    "api",
    "mcp",
    "model-runtime",
    "model-tunnel",
]
MAX_DETAILS = 50
ARTIFACT_STEPS = (
    "coverage_upload",
    "test_upload",
    "types_upload",
    "repair_upload",
    "quality_logs",
    "unit_download",
    "integration_download",
    "storage_download",
    "diagnostics_download",
)


class DiagnosticKind(StrEnum):
    """Separate originating failures from unexecuted work and broken evidence transfer."""

    COLLECTION_ERROR = "collection_error"
    TEST_FAILURE = "test_failure"
    NOT_RUN = "not_run"
    UPSTREAM_BLOCKED = "upstream_blocked"
    ARTIFACT_ERROR = "artifact_error"


class RunIdentity(BaseModel):
    """Bind every diagnostic to one immutable commit and workflow attempt."""

    tested_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    run_id: str = Field(pattern=r"^[0-9]+$")
    run_attempt: str = Field(pattern=r"^[0-9]+$")


class DiagnosticRecord(RunIdentity):
    """Only bounded identities, counts and typed outcomes enter the final summary."""

    stage: Stage
    steps: dict[Annotated[str, Field(max_length=100)], StepResult] = Field(max_length=100)
    counts: dict[CaseOutcome, Annotated[int, Field(ge=0)]] = Field(default_factory=dict)
    affected: list[Annotated[str, Field(max_length=300)]] = Field(
        default_factory=list, max_length=MAX_DETAILS
    )
    report_valid: bool | None = None
    assessment: CheckEvidence | None = None
    dependencies: Assessment | None = None
    types: TypeReport | None = None


def record(request: DiagnosticRecord, report: Path | None) -> DiagnosticRecord:
    """Project JUnit without copying SQL, failure prose or captured output."""
    if report is None:
        return request
    parsed = read_junit(report)
    cases = parsed.cases
    if request.stage == "migrations":
        cases = [case for case in cases if case.classname == "tests.integration.test_migrations"]
    assessment = request.assessment
    if request.stage == "migrations":
        assessment = migration_assessment(
            parsed, request.steps.get("tests", StepResult(outcome=Result.SKIPPED)).outcome
        )
    return request.model_copy(
        update={
            "counts": dict(Counter(case.outcome for case in cases)),
            "affected": [
                ".".join(case.identity)[:300]
                for case in cases
                if case.outcome != CaseOutcome.PASSED
            ][:MAX_DETAILS],
            "report_valid": parsed.valid,
            "assessment": assessment,
        }
    )


def execution_category(value: DiagnosticRecord) -> DiagnosticKind | None:
    """Classify execution using structured outcomes, never exception text or filenames."""
    if value.stage in ("quality", *PARTITIONS):
        quality = value.stage == "quality"
        step = value.steps.get("collection" if quality else "tests")
        if step is not None and step.outcome == Result.FAILURE:
            return DiagnosticKind.COLLECTION_ERROR if quality else DiagnosticKind.TEST_FAILURE
        return None if step and step.outcome == Result.SUCCESS else DiagnosticKind.NOT_RUN
    if value.stage == "migrations":
        if not value.counts:
            return DiagnosticKind.NOT_RUN
        if any(count for kind, count in value.counts.items() if kind != CaseOutcome.PASSED):
            return DiagnosticKind.TEST_FAILURE
    return None


def evidence_categories(value: DiagnosticRecord) -> list[DiagnosticKind]:
    """Keep independent transfer failures visible alongside the original failure."""
    categories = []
    if any(
        step.outcome in (Result.FAILURE, Result.CANCELLED)
        for name in ARTIFACT_STEPS
        if (step := value.steps.get(name)) is not None
    ):
        categories.append(DiagnosticKind.ARTIFACT_ERROR)
    if value.assessment is not None:
        if any(result != Result.SUCCESS for result in value.assessment.upstream.values()):
            categories.append(DiagnosticKind.UPSTREAM_BLOCKED)
        elif not value.assessment.evidence_valid:
            categories.append(DiagnosticKind.ARTIFACT_ERROR)
    return categories


def render(record: DiagnosticRecord) -> str:
    """Explain independent step results without softening a job failure."""
    rows = [f"### {record.stage}"]
    categories = evidence_categories(record)
    if (category := execution_category(record)) is not None:
        categories.insert(0, category)
    if categories:
        rows.append("Categories: " + ", ".join(dict.fromkeys(categories)))
    rows.extend(
        f"- {case_label(name)}: {step.outcome.value}" for name, step in record.steps.items()
    )
    if record.report_valid is not None:
        rows.append(
            f"JUnit structurally valid: {record.report_valid}; counts: "
            + ", ".join(f"{kind.value}={count}" for kind, count in record.counts.items())
        )
    rows.extend(f"- Affected: {case_label(name)}" for name in record.affected)
    if record.assessment is not None:
        rows.append(record.assessment.describe())
    if record.dependencies is not None:
        detail = record.dependencies
        rows.append(
            f"Dependency audit: {detail.status}; raw exit: {detail.raw_exit}; "
            f"scoped permits: {len(detail.permitted)}; blocking: {len(detail.blocking)}."
        )
        rows.extend(
            f"- {item.error.code.value}: {case_label(item.module)} in "
            f"{case_label(item.location.file)}"
            for item in detail.blocking[:MAX_DETAILS]
        )
    if record.types is not None:
        rows.append(describe_types(record.types))
    return "\n".join(rows)


def aggregate(
    directory: Path,
    *,
    run: RunIdentity,
    expected: tuple[str, ...] = (),
    results: Results | None = None,
    plan: Plan | None = None,
) -> str:
    """Reject stale, malformed and duplicated stage evidence in diagnostic displays."""
    records: dict[str, DiagnosticRecord] = {}
    invalid: set[str] = set()
    rows = ["\n## Stage diagnostics (raw job results above remain authoritative)"]
    for path in sorted(directory.rglob("*.json")):
        try:
            value = DiagnosticRecord.model_validate_json(path.read_text())
        except (OSError, ValueError, ValidationError):
            rows.append(f"Unavailable diagnostic: {case_label(path.name)} (invalid evidence).")
            continue
        if (value.tested_sha, value.run_id, value.run_attempt) != (
            run.tested_sha,
            run.run_id,
            run.run_attempt,
        ):
            rows.append(f"Unavailable diagnostic: {value.stage} (SHA/run/attempt mismatch).")
            continue
        if value.stage in records:
            invalid.add(value.stage)
        records[value.stage] = value
    for stage, value in records.items():
        rows.append(
            f"Unavailable diagnostic: {stage} (duplicate evidence)."
            if stage in invalid
            else render(value)
        )
    rows.extend(
        missing_record_description(stage, results, plan)
        for stage in expected
        if stage not in records
    )
    if not records:
        rows.append("No valid stage records; inspect setup, reporting and artifact upload logs.")
    return "\n\n".join(rows)


def missing_record_description(stage: str, results: Results | None, plan: Plan | None) -> str:
    """A blocked job cannot upload a record; retain its upstream cause in the summary."""
    job = "integration" if stage == "migrations" else stage
    if results is not None and plan is not None and job in Results.model_fields:
        result = getattr(results, job).result
        if result in (Result.SKIPPED, Result.CANCELLED):
            return f"{stage}: not_run/upstream_blocked; {job_description(job, plan, results)}."
    return f"Unavailable diagnostic: {stage} (artifact_error: missing current-run evidence)."


def read_types(path: Path) -> TypeReport:
    """Missing reporting evidence must not be displayed as successful type checking."""
    try:
        return TypeReport.model_validate_json(path.read_text())
    except (OSError, ValueError, ValidationError):
        return TypeReport(report_valid=False)


def read_dependencies(path: Path) -> Assessment:
    """Keep unavailable dependency evidence distinct from a successful audit."""
    try:
        return Assessment.model_validate_json(path.read_text())
    except (OSError, ValueError, ValidationError):
        return Assessment(status="invalid_evidence", raw_exit=None)


def main() -> None:
    """Produce a bounded artifact using explicit workflow arguments only."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=get_args(Stage))
    parser.add_argument("--tested-sha", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-attempt", required=True)
    parser.add_argument("--steps", default="{}")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--plan")
    parser.add_argument("--needs")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--assessment", type=Path)
    parser.add_argument("--dependencies", type=Path)
    parser.add_argument("--types", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.directory is not None:
        plan = Plan.model_validate_json(args.plan) if args.plan is not None else None
        expected = (
            ()
            if plan is None
            else (
                *(
                    ("quality", "unit", "integration", "storage", "migrations", "coverage")
                    if plan.checks
                    else ()
                ),
                *plan.images,
            )
        )
        with args.summary.open("a") as stream:
            stream.write(
                aggregate(
                    args.directory,
                    run=RunIdentity(
                        tested_sha=args.tested_sha,
                        run_id=args.run_id,
                        run_attempt=args.run_attempt,
                    ),
                    expected=expected,
                    results=Results.model_validate_json(args.needs) if args.needs else None,
                    plan=plan,
                )
                + "\n"
            )
        return
    if args.stage is None or args.output is None:
        parser.error("recording requires --stage and --output")
    value = DiagnosticRecord(
        tested_sha=args.tested_sha,
        run_id=args.run_id,
        run_attempt=args.run_attempt,
        stage=args.stage,
        steps=TypeAdapter(dict[str, StepResult]).validate_json(args.steps),
    )
    if args.assessment is not None:
        try:
            value.assessment = CheckEvidence.model_validate_json(args.assessment.read_text())
        except (OSError, ValueError, ValidationError):
            value.assessment = CheckEvidence(checks_passed=None, evidence_valid=False)
    if args.dependencies is not None:
        value.dependencies = read_dependencies(args.dependencies)
    if args.types is not None:
        value.types = read_types(args.types)
    value = record(value, args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(value.model_dump_json(indent=2) + "\n")
    with args.summary.open("a") as stream:
        stream.write(render(value) + "\n")


if __name__ == "__main__":
    main()
