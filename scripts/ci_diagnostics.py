"""SHA/run-bound diagnostic records; raw workflow outcomes remain the acceptance gate."""

import argparse
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, get_args

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from scripts.ci_collection import CollectionReport
from scripts.ci_dependencies import Assessment
from scripts.ci_evidence import CheckEvidence, CoverageState
from scripts.ci_junit import CaseOutcome, read_junit
from scripts.ci_partitions import PARTITIONS
from scripts.ci_policy import ALL_IMAGES, Plan, Result, Results, failures
from scripts.ci_result import (
    QUALITY_CHECKS,
    StepResult,
    case_label,
    job_description,
    migration_assessment,
)
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
    "dependencies_upload",
    "repair_upload",
    "quality_logs",
    "unit_download",
    "integration_download",
    "storage_download",
    "diagnostics_download",
)


class DiagnosticKind(StrEnum):
    """Separate originating failures from unexecuted work and broken evidence transfer."""

    QUALITY_FAILURE = "quality_failure"
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
    collection: CollectionReport | None = None


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


def quality_category(value: DiagnosticRecord) -> DiagnosticKind | None:
    """Collection and independent quality failures retain distinct categories."""
    collection = value.steps.get("collection")
    if collection is not None and collection.outcome == Result.FAILURE:
        return DiagnosticKind.COLLECTION_ERROR
    if any(
        (check := value.steps.get(name)) is not None and check.outcome == Result.FAILURE
        for name in QUALITY_CHECKS
    ):
        return DiagnosticKind.QUALITY_FAILURE
    if collection is None or collection.outcome != Result.SUCCESS:
        return DiagnosticKind.NOT_RUN
    return None


def execution_category(value: DiagnosticRecord) -> DiagnosticKind | None:
    """Classify execution using structured outcomes, never exception text or filenames."""
    if value.stage == "quality":
        return quality_category(value)
    if value.stage in PARTITIONS:
        step = value.steps.get("tests")
        if step is not None and step.outcome == Result.FAILURE:
            return DiagnosticKind.TEST_FAILURE
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
        if value.assessment.artifact_error:
            categories.append(DiagnosticKind.ARTIFACT_ERROR)
    tests = value.steps.get("tests")
    if (
        value.report_valid is False
        and tests is not None
        and tests.outcome in (Result.SUCCESS, Result.FAILURE)
    ):
        categories.append(DiagnosticKind.ARTIFACT_ERROR)
    return categories


def render(record: DiagnosticRecord) -> str:
    """Explain independent step results without softening a job failure."""
    rows = [f"### {record.stage}"]
    categories = evidence_categories(record)
    if (category := execution_category(record)) is not None:
        categories.insert(0, category)
    if record.stage == "quality" and any(
        (step := record.steps.get(name)) is not None and step.outcome is Result.FAILURE
        for name in QUALITY_CHECKS
        if name != "collection"
    ):
        categories.append(DiagnosticKind.QUALITY_FAILURE)
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
    if record.collection is not None:
        collection_detail = record.collection
        rows.append(
            f"Collection: raw exit={collection_detail.raw_exit}; collected={collection_detail.collected}; "
            f"failed collectors={collection_detail.failed_count}."
        )
        rows.extend(f"- Collector: {case_label(node)}" for node in collection_detail.failed_nodes)
    if record.types is not None:
        rows.append(describe_types(record.types))
    return "\n".join(rows)


class DiagnosticAssessment(BaseModel):
    """Evidence validity is an additional gate; it cannot soften raw job failures."""

    summary: str
    valid: bool
    raw_passed: bool

    @property
    def accepted(self) -> bool:
        """Both authoritative outcomes and current-run evidence must pass."""
        return self.valid and self.raw_passed


def stage_job(stage: str) -> str:
    """Resolve migration reuse and individual image evidence to their parent job."""
    if stage in ALL_IMAGES:
        return "build"
    return "integration" if stage == "migrations" else stage


def successful_record(value: DiagnosticRecord) -> bool:
    """Reject incomplete or contradictory evidence accompanying a successful job."""
    required = {
        "quality": ("setup", *QUALITY_CHECKS, "report"),
        "coverage": ("setup", "coverage"),
        "migrations": ("setup", "tests", "migrations_report"),
    }.get(value.stage, ("setup", "tests", "test_report"))
    if value.stage in ALL_IMAGES:
        required = ("setup", "build", "smoke")
    if any(
        (step := value.steps.get(name)) is None or step.outcome is not Result.SUCCESS
        for name in required
    ) or any(step.outcome in (Result.FAILURE, Result.CANCELLED) for step in value.steps.values()):
        return False
    return successful_payload(value)


def successful_payload(value: DiagnosticRecord) -> bool:
    """Check the relevant structured measurements rather than just step labels."""
    if value.stage == "quality":
        return (
            value.collection is not None
            and value.collection.accepted
            and value.types is not None
            and value.types.accepted
            and value.dependencies is not None
            and value.dependencies.status == "passed"
            and not value.dependencies.blocking
        )
    if value.stage == "coverage":
        return (
            value.assessment is not None
            and value.assessment.accepted
            and set(value.assessment.upstream) == set(PARTITIONS)
            and set(value.assessment.partitions) == set(PARTITIONS)
            and all(state is CoverageState.VALID for state in value.assessment.partitions.values())
        )
    if value.stage == "migrations":
        return (
            value.assessment is not None
            and value.assessment.accepted
            and value.assessment.upstream == {"integration": Result.SUCCESS}
            and value.report_valid is True
        )
    if value.stage in PARTITIONS:
        return (
            value.report_valid is True
            and value.counts.get(CaseOutcome.PASSED, 0) > 0
            and all(
                count == 0 for kind, count in value.counts.items() if kind != CaseOutcome.PASSED
            )
        )
    return True


def load_records(
    directory: Path, run: RunIdentity
) -> tuple[dict[str, DiagnosticRecord], list[str]]:
    """Validate every downloaded record, including stale extras and duplicates."""
    records: dict[str, DiagnosticRecord] = {}
    problems = []
    duplicated: set[str] = set()
    for path in sorted(directory.rglob("*.json")):
        try:
            value = DiagnosticRecord.model_validate_json(path.read_text())
        except (OSError, ValueError, ValidationError):
            problems.append(f"Unavailable diagnostic: {case_label(path.name)} (invalid evidence).")
            continue
        if (value.tested_sha, value.run_id, value.run_attempt) != (
            run.tested_sha,
            run.run_id,
            run.run_attempt,
        ):
            problems.append(f"Unavailable diagnostic: {value.stage} (SHA/run/attempt mismatch).")
            continue
        if value.stage in records:
            duplicated.add(value.stage)
        records[value.stage] = value
    for stage in duplicated:
        records.pop(stage)
        problems.append(f"Unavailable diagnostic: {stage} (duplicate evidence).")
    return records, problems


def assess_diagnostics(
    directory: Path,
    *,
    run: RunIdentity,
    expected: tuple[str, ...] = (),
    results: Results | None = None,
    plan: Plan | None = None,
) -> DiagnosticAssessment:
    """Fail closed on missing required evidence while explaining legitimate absence."""
    records, problems = load_records(directory, run)
    rows = ["\n## Stage diagnostics (raw job results remain authoritative)", *problems]
    valid = not problems
    for stage in expected:
        job = getattr(results, stage_job(stage)).result if results is not None else None
        value = records.get(stage)
        if value is None:
            rows.append(missing_record_description(stage, results, plan))
            valid = valid and job in (Result.SKIPPED, Result.CANCELLED)
        elif job is Result.SUCCESS and not successful_record(value):
            valid = False
            rows.append(f"Unavailable diagnostic: {stage} (contradicts raw job success).")
    if plan is not None:
        valid = valid and plan.tested_sha == run.tested_sha
        unexpected = set(records) - set(expected)
        if unexpected:
            valid = False
            rows.append("Unexpected stage evidence: " + ", ".join(sorted(unexpected)))
    rows.extend(render(value) for value in records.values())
    if not records:
        rows.append("No valid stage records; inspect setup, reporting and artifact upload logs.")
    raw_passed = plan is None or (results is not None and not failures(plan, results))
    assessment = DiagnosticAssessment(summary="\n\n".join(rows), valid=valid, raw_passed=raw_passed)
    assessment.summary += f"\n\nDiagnostic acceptance: {'PASS' if assessment.accepted else 'FAIL'}."
    return assessment


def aggregate(
    directory: Path,
    *,
    run: RunIdentity,
    expected: tuple[str, ...] = (),
    results: Results | None = None,
    plan: Plan | None = None,
) -> str:
    """Keep the human-readable projection available to diagnostic callers."""
    return assess_diagnostics(
        directory, run=run, expected=expected, results=results, plan=plan
    ).summary


def missing_record_description(stage: str, results: Results | None, plan: Plan | None) -> str:
    """A blocked job cannot upload a record; retain its upstream cause in the summary."""
    job = stage_job(stage)
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
    parser.add_argument("--collection", type=Path)
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
        if plan is None or args.needs is None:
            parser.error("aggregation requires --plan and --needs")
        assessment = assess_diagnostics(
            args.directory,
            run=RunIdentity(
                tested_sha=args.tested_sha, run_id=args.run_id, run_attempt=args.run_attempt
            ),
            expected=expected,
            results=Results.model_validate_json(args.needs),
            plan=plan,
        )
        with args.summary.open("a") as stream:
            stream.write(assessment.summary + "\n")
        print(assessment.summary)
        raise SystemExit(0 if assessment.accepted else 1)
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
    if args.collection is not None:
        try:
            value.collection = CollectionReport.model_validate_json(args.collection.read_text())
        except (OSError, ValueError, ValidationError):
            value.collection = None
    value = record(value, args.report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(value.model_dump_json(indent=2) + "\n")
    with args.summary.open("a") as stream:
        stream.write(render(value) + "\n")


if __name__ == "__main__":
    main()
