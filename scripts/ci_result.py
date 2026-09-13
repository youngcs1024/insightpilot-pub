"""Fail-closed final gate and migration evidence extracted without rerunning tests."""

import argparse
from collections import Counter
from html import escape
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from app.core.errors import InsightPilotError
from scripts.ci_evidence import CheckEvidence
from scripts.ci_junit import CaseOutcome, FileState, JunitReport, file_state, read_junit
from scripts.ci_partitions import PARTITIONS
from scripts.ci_policy import Plan, Result, Results, failures

MIGRATION_CASES = (
    "test_registry_migration_roundtrip",
    "test_upgrade_head_from_empty",
    "test_downgrade_to_base",
    "test_autogenerate_produces_empty_diff[app]",
    "test_autogenerate_produces_empty_diff[business]",
    "test_checkpoint_tables_excluded",
    "test_real_autogenerate_file_is_empty[app]",
    "test_real_autogenerate_file_is_empty[business]",
)
QUALITY_CHECKS = (
    "contracts",
    "structure",
    "corpus",
    "collection",
    "lint",
    "format",
    "types",
    "dependencies",
)
QUALITY_COMMANDS = {
    "structure": ".venv/bin/python -m scripts.check_test_structure",
    "contracts": ".venv/bin/python -m scripts.check_deployment_contracts",
    "corpus": ".venv/bin/python -m scripts.corpus_stats",
    "collection": (
        '.venv/bin/pytest tests --collect-only --no-cov -m "not gpu and not external" '
        "-p tests.ci_collection_plugin --collection-report quality-collection.json"
    ),
    "lint": "make lint-check ENV=test",
    "format": "make format-check ENV=test",
    "types": "make typecheck-report ENV=test",
    "dependencies": ".venv/bin/python -m scripts.ci_dependencies --output dependency-evidence",
}
MAX_DIAGNOSTICS = 50
MAX_CASE_NAME = 300


class StepResult(BaseModel):
    """Read raw outcome, never a possibly softened step conclusion."""

    outcome: Result


class QualitySteps(BaseModel):
    """Missing steps remain visible and cannot satisfy the quality gate."""

    setup: StepResult | None = None
    structure: StepResult | None = None
    contracts: StepResult | None = None
    corpus: StepResult | None = None
    collection: StepResult | None = None
    lint: StepResult | None = None
    format: StepResult | None = None
    types: StepResult | None = None
    dependencies: StepResult | None = None


class QualityDetails(BaseModel):
    """Bound diagnostic outcomes to the current commit, independent of job results."""

    tested_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    steps: QualitySteps


class QualityOutputs(BaseModel):
    """GitHub transports the typed diagnostic document as a JSON string."""

    quality_details: str = ""


class QualityDiagnosticJob(BaseModel):
    """Parse optional diagnostics separately so they cannot alter the raw gate."""

    outputs: QualityOutputs = Field(default_factory=QualityOutputs)


class QualityNeeds(BaseModel):
    """Only the quality job's diagnostic output is consumed here."""

    quality: QualityDiagnosticJob


def emit(message: str, summary: Path | None) -> None:
    """Publish the same readable result to logs and the optional job summary."""
    print(message)
    if summary is not None:
        with summary.open("a") as stream:
            stream.write(message + "\n")


def quality_summary(
    steps_json: str, summary: Path, *, output: Path | None = None, tested_sha: str = ""
) -> bool:
    """Require setup and every independent check to have actually succeeded."""
    try:
        steps = QualitySteps.model_validate_json(steps_json)
        if output is not None:
            details = QualityDetails(tested_sha=tested_sha, steps=steps)
            with output.open("a") as stream:
                stream.write(f"quality_details={details.model_dump_json()}\n")
    except ValidationError:
        emit("Quality: FAIL — invalid step outcomes", summary)
        return False
    rows = [
        "## Quality checks\n",
        "| Check | Outcome | Log | Reproduce |",
        "| --- | --- | --- | --- |",
    ]
    passed = True
    for name in ("setup", *QUALITY_CHECKS):
        step: StepResult | None = getattr(steps, name)
        outcome = step.outcome.value if step else "not executed / missing"
        command = QUALITY_COMMANDS.get(name, "make install ENV=test")
        log = f"quality-{name}.log" if name in QUALITY_COMMANDS else "setup step log"
        rows.append(f"| {name} | {outcome} | {log} | `{command}` |")
        passed = passed and step is not None and step.outcome == Result.SUCCESS
    emit("\n".join(rows), summary)
    return passed


def quality_diagnostics(needs_json: str, result: Result, tested_sha: str) -> str:
    """Optional metadata explains failures but can never override raw job results."""
    try:
        raw = QualityNeeds.model_validate_json(needs_json).quality.outputs.quality_details
        details = QualityDetails.model_validate_json(raw)
    except ValidationError:
        return "Quality detail unavailable (missing or invalid output); inspect the quality job."
    if details.tested_sha != tested_sha:
        return "Quality detail unavailable (SHA mismatch); inspect the quality job."
    problems = [
        name
        for name in ("setup", *QUALITY_CHECKS)
        if (step := getattr(details.steps, name)) is None or step.outcome != Result.SUCCESS
    ]
    if result == Result.SUCCESS and problems:
        return "Quality detail unavailable (contradicts raw job success); raw gate unchanged."
    if not problems:
        return (
            "All reported quality checks succeeded."
            if result == Result.SUCCESS
            else "Quality checks reported success; inspect reporting/upload or job interruption."
        )
    return "Quality checks requiring attention: " + ", ".join(
        f"quality / {name}" for name in problems
    )


def case_label(name: str) -> str:
    """Bound and escape identities, including newlines and parameterized input text."""
    name = " ".join(name.split())
    if len(name) > MAX_CASE_NAME:
        name = name[: MAX_CASE_NAME - 3] + "..."
    return "<code>" + escape(name) + "</code>"


def report_diagnostics(report: JunitReport) -> list[str]:
    """List actionable identities without ever copying failure messages or bodies."""
    rows = [f"JUnit artifact: {report.file_state.value}."]
    rows.extend(f"- Report defect: {problem.value}" for problem in report.problems)
    counts = Counter(case.identity for case in report.cases)
    diagnostics = [
        case
        for case in report.cases
        if case.outcome != CaseOutcome.PASSED or counts[case.identity] > 1
    ]
    if diagnostics:
        rows.append("\nAffected testcases:\n")
    for case in diagnostics[:MAX_DIAGNOSTICS]:
        category = "duplicate" if counts[case.identity] > 1 else case.outcome.value
        identity = ".".join(part for part in case.identity if part)
        rows.append(f"- {category}: {case_label(identity)}")
    if len(diagnostics) > MAX_DIAGNOSTICS:
        rows.append(f"- Omitted {len(diagnostics) - MAX_DIAGNOSTICS} additional entries.")
    return rows


def execution_description(outcome: Result) -> str:
    """Separate unexecuted work from failed or cancelled work with partial artifacts."""
    if outcome == Result.SKIPPED:
        return "not executed; no acceptance claimed"
    if outcome == Result.CANCELLED:
        return "cancelled or interrupted; partial evidence is diagnostic only"
    if outcome == Result.FAILURE:
        return "failed (including step timeout); artifacts cannot override the raw outcome"
    return "completed successfully; evidence still requires validation"


def partition_summary(report: Path, coverage: Path, outcome: Result, summary: Path) -> bool:
    """Require executed tests, valid evidence and no skipped required cases."""
    parsed = read_junit(report)
    counts = Counter(case.outcome for case in parsed.cases)
    coverage_state = file_state(coverage)
    passed = (
        outcome == Result.SUCCESS
        and parsed.valid
        and counts[CaseOutcome.PASSED] == len(parsed.cases)
        and coverage_state == FileState.PRESENT
    )
    rows = [
        f"Tests: {'PASS' if passed else 'FAIL'}; outcome={outcome.value}; "
        f"total={len(parsed.cases)}, passed={counts[CaseOutcome.PASSED]}, "
        f"failures={counts[CaseOutcome.FAILURE]}, errors={counts[CaseOutcome.ERROR]}, "
        f"skipped={counts[CaseOutcome.SKIPPED]}, invalid={counts[CaseOutcome.INVALID]}",
        f"Execution: {execution_description(outcome)}.",
        f"Coverage artifact: {coverage_state.value} (content is validated by coverage.py).",
        *report_diagnostics(parsed),
        "\nFull diagnostics remain in this job's test log and evidence artifact.",
    ]
    emit("\n".join(rows), summary)
    return passed


def migration_summary(report: Path, summary: Path, outcome: Result = Result.SUCCESS) -> bool:
    """Reuse the same strict parser; partial runs cannot complete migration acceptance."""
    parsed = read_junit(report)
    cases = [case for case in parsed.cases if case.classname == "tests.integration.test_migrations"]
    failed = [case for case in cases if case.outcome != CaseOutcome.PASSED]
    counts = Counter(case.name for case in cases)
    missing = [name for name in MIGRATION_CASES if not counts[name]]
    assessment = migration_assessment(parsed, outcome)
    passed = assessment.accepted
    rows = [
        "## Migration acceptance (integration run)\n",
        f"Passed: {passed}.",
        assessment.describe(),
        f"Execution: {execution_description(outcome)}.",
        f"Migration cases: {len(cases) - len(failed)}/{len(cases)} passed; "
        f"required={len(MIGRATION_CASES)}. Overall integration acceptance: {outcome.value}.",
        *report_diagnostics(parsed.model_copy(update={"cases": cases})),
    ]
    rows.extend(f"- Missing: {case_label(name)}" for name in missing)
    if not cases:
        rows.append(
            "not_run: no migration execution evidence; no migration assertion failure claimed."
        )
    rows.append("Required cases: " + ", ".join(MIGRATION_CASES))
    emit("\n".join(rows), summary)
    return passed


def migration_assessment(parsed: JunitReport, outcome: Result) -> CheckEvidence:
    """Missing required instances are incomplete evidence, not failed assertions."""
    cases = [case for case in parsed.cases if case.classname == "tests.integration.test_migrations"]
    counts = Counter(case.name for case in cases)
    return CheckEvidence(
        checks_passed=all(case.outcome == CaseOutcome.PASSED for case in cases) if cases else None,
        evidence_valid=parsed.valid and all(counts[name] == 1 for name in MIGRATION_CASES),
        upstream={"integration": outcome},
    )


def collection_blocker(results: Results) -> str | None:
    """Other quality failures do not block tests after successful collection."""
    collection = results.quality.outputs.collection_outcome
    if collection == Result.SUCCESS and results.quality.result != Result.CANCELLED:
        return None
    cause = "collection_error" if collection == Result.FAILURE else "not_run"
    return (
        f"not_run/upstream_blocked: quality / collection "
        f"({cause}; outcome={collection or 'missing'}; quality={results.quality.result.value})"
    )


def job_description(name: str, plan: Plan, results: Results) -> str:
    """Explain failure propagation without relaxing the result predicate."""
    result: Result = getattr(results, name).result
    if result != Result.SKIPPED:
        return result.value
    selected = name == "changes" or (bool(plan.images) if name == "build" else plan.checks)
    if not selected:
        return "planned omission"
    if name in PARTITIONS and results.changes.result == Result.SUCCESS:
        return collection_blocker(results) or "unexpected skip"
    dependencies: dict[str, tuple[str, ...]] = {
        "changes": (),
        "quality": ("changes",),
        # Collection is the quality prerequisite, handled above independently of lint/types.
        "build": ("changes",),
        "coverage": ("changes", *PARTITIONS),
    }
    dependencies.update(dict.fromkeys(PARTITIONS, ("changes", "quality")))
    blockers = [
        parent for parent in dependencies[name] if getattr(results, parent).result != Result.SUCCESS
    ]
    return "blocked by " + ", ".join(blockers) if blockers else "unexpected skip"


def final_result(
    plan_json: str, needs_json: str, summary: Path | None = None, *, tested_sha: str
) -> bool:
    """Invalid outputs or absent required job results cannot produce success."""
    try:
        plan = Plan.model_validate_json(plan_json)
        results = Results.model_validate_json(needs_json)
    except (ValidationError, InsightPilotError):
        emit("CI result failed: missing or invalid selection/results", summary)
        return False
    if plan.tested_sha != tested_sha:
        emit("CI result failed: selection SHA does not match tested commit", summary)
        return False
    failed = failures(plan, results)
    rows = ["## CI result\n", "| Job | Outcome |", "| --- | --- |"]
    rows.extend(
        f"| {name} | {job_description(name, plan, results)} |" for name in Results.model_fields
    )
    rows.append(f"\nCI result: {'FAIL ' + ', '.join(failed) if failed else 'PASS'}")
    if plan.checks:
        collection = results.quality.outputs.collection_outcome
        rows.append(f"\nRaw collection_outcome: {collection or 'missing'}.")
        if collection == Result.FAILURE:
            rows.append("collection_error: ordinary tests could not be collected.")
        rows.append("\n" + quality_diagnostics(needs_json, results.quality.result, tested_sha))
    emit("\n".join(rows), summary)
    return not failed


def main() -> None:
    """Run a gate using explicit arguments, without inherited application settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    result = commands.add_parser("result")
    result.add_argument("--plan", required=True)
    result.add_argument("--needs", required=True)
    result.add_argument("--tested-sha", required=True)
    result.add_argument("--summary", type=Path)
    quality = commands.add_parser("quality")
    quality.add_argument("--steps", required=True)
    quality.add_argument("--summary", type=Path, required=True)
    quality.add_argument("--output", type=Path)
    quality.add_argument("--tested-sha", default="")
    migrations = commands.add_parser("migrations")
    migrations.add_argument("--report", type=Path, required=True)
    migrations.add_argument("--outcome", type=Result, required=True)
    migrations.add_argument("--summary", type=Path, required=True)
    tests = commands.add_parser("tests")
    tests.add_argument("--report", type=Path, required=True)
    tests.add_argument("--coverage", type=Path, required=True)
    tests.add_argument("--outcome", type=Result, required=True)
    tests.add_argument("--summary", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "result":
        passed = final_result(
            arguments.plan, arguments.needs, arguments.summary, tested_sha=arguments.tested_sha
        )
    elif arguments.command == "quality":
        passed = quality_summary(
            arguments.steps,
            arguments.summary,
            output=arguments.output,
            tested_sha=arguments.tested_sha,
        )
    elif arguments.command == "tests":
        passed = partition_summary(
            arguments.report, arguments.coverage, arguments.outcome, arguments.summary
        )
    else:
        passed = migration_summary(arguments.report, arguments.summary, arguments.outcome)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
