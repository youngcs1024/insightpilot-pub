"""Current-run diagnostics explain independent failures without weakening the raw gate."""

from pathlib import Path

import pytest

from scripts.ci_diagnostics import (
    DiagnosticRecord,
    RunIdentity,
    aggregate,
    read_types,
    record,
    render,
)
from scripts.ci_evidence import CheckEvidence
from scripts.ci_policy import Reason, Result, Results, full_plan
from scripts.ci_result import MIGRATION_CASES, StepResult, migration_summary
from scripts.ci_types import parse_output
from tests.tooling.support import needs_for

SHA = "a" * 40


def evidence(**updates: object) -> DiagnosticRecord:
    return DiagnosticRecord.model_validate(
        {
            "tested_sha": SHA,
            "run_id": "123",
            "run_attempt": "2",
            "stage": "unit",
            "steps": {"tests": {"outcome": "failure"}},
            **updates,
        }
    )


def joined(path: Path) -> str:
    return aggregate(
        path, run=RunIdentity(tested_sha=SHA, run_id="123", run_attempt="2"), expected=("unit",)
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tested_sha", "b" * 40),
        ("run_id", "456"),
        ("run_attempt", "1"),
    ],
)
def test_stale_run_or_commit_is_never_displayed_as_current(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    (tmp_path / "unit.json").write_text(evidence(**{field: value}).model_dump_json())
    assert "mismatch" in joined(tmp_path)
    assert "missing current-run evidence" in joined(tmp_path)
    assert "### unit" not in joined(tmp_path)


def test_multiple_failures_and_cancellation_keep_their_step_outcomes(tmp_path: Path) -> None:
    (tmp_path / "unit.json").write_text(evidence().model_dump_json())
    (tmp_path / "quality.json").write_text(
        evidence(
            stage="quality",
            steps={
                "lint": StepResult(outcome=Result.FAILURE),
                "types": StepResult(outcome=Result.FAILURE),
                "format": StepResult(outcome=Result.SUCCESS),
                "upload": StepResult(outcome=Result.CANCELLED),
            },
        ).model_dump_json()
    )
    summary = joined(tmp_path)
    assert "### unit" in summary
    assert "### quality" in summary
    assert "<code>types</code>: failure" in summary
    assert "<code>upload</code>: cancelled" in summary


def test_missing_malformed_and_duplicate_records_are_explicit(tmp_path: Path) -> None:
    assert "missing current-run evidence" in joined(tmp_path)
    (tmp_path / "bad.json").write_text("{")
    assert "invalid evidence" in joined(tmp_path)
    for filename in ("one.json", "two.json"):
        (tmp_path / filename).write_text(evidence().model_dump_json())
    assert "duplicate evidence" in joined(tmp_path)
    assert "### unit" not in joined(tmp_path)


def test_migrations_do_not_claim_failure_from_unrelated_api_tests(tmp_path: Path) -> None:
    xml = tmp_path / "results.xml"
    cases = "".join(
        f'<testcase classname="tests.integration.test_migrations" name="{name}"/>'
        for name in MIGRATION_CASES
    )
    xml.write_text(
        "<testsuite>"
        + cases
        + '<testcase classname="tests.api" name="broken"><failure>secret</failure>'
        "</testcase></testsuite>"
    )
    summary = tmp_path / "summary"
    assert not migration_summary(xml, summary, Result.FAILURE)
    assert "7/7 passed" in summary.read_text()
    assert "tests.api.broken" not in summary.read_text()
    detail = record(evidence(stage="migrations"), xml)
    assert not detail.affected
    assert "secret" not in detail.model_dump_json()


def test_affected_cases_are_bounded_and_never_include_failure_bodies(tmp_path: Path) -> None:
    xml = tmp_path / "results.xml"
    xml.write_text(
        "<testsuite>"
        + "".join(
            f'<testcase name="t{index}"><failure>private SQL</failure></testcase>'
            for index in range(60)
        )
        + "</testsuite>"
    )
    result = record(evidence(), xml)
    assert len(result.affected) == 50  # noqa: PLR2004 -- bounded summary contract.
    assert "private SQL" not in result.model_dump_json()


@pytest.mark.parametrize("upstream", [Result.FAILURE, Result.CANCELLED, Result.SKIPPED])
def test_passing_checks_do_not_hide_upstream_blockers(tmp_path: Path, upstream: Result) -> None:
    value = evidence(
        stage="coverage",
        assessment=CheckEvidence(
            checks_passed=True,
            evidence_valid=True,
            upstream={"integration": upstream},
        ),
    )
    (tmp_path / "coverage.json").write_text(value.model_dump_json())
    summary = joined(tmp_path)
    assert "Checks: PASS" in summary
    assert f"integration={upstream.value}" in summary
    assert "acceptance: FAIL" in summary


@pytest.mark.parametrize(("checks", "valid"), [(False, True), (None, False), (True, False)])
def test_failed_or_missing_measurements_cannot_pass(checks: bool | None, valid: bool) -> None:
    result = CheckEvidence(checks_passed=checks, evidence_valid=valid)
    assert not result.accepted


@pytest.mark.parametrize("field", ["tested_sha", "run_id", "run_attempt"])
def test_type_details_cannot_cross_commit_or_run_boundary(tmp_path: Path, field: str) -> None:
    types = parse_output(
        b'{"severity":"error","file":"private.py","message":"hidden detail","code":"var-annotated"}',
        1,
    )
    changed = "b" * 40 if field == "tested_sha" else "999"
    value = evidence(types=types, **{field: changed})
    (tmp_path / "types.json").write_text(value.model_dump_json())
    assert "hidden detail" not in joined(tmp_path)
    assert "mismatch" in joined(tmp_path)


def test_missing_type_artifact_is_not_reported_as_success(tmp_path: Path) -> None:
    assert not read_types(tmp_path / "absent.json").accepted
    path = tmp_path / "invalid.json"
    path.write_text("{")
    assert not read_types(path).accepted


def test_collection_blocked_stages_are_not_reported_as_artifact_failures(tmp_path: Path) -> None:
    plan = full_plan(SHA, Reason.MANUAL)
    needs = needs_for(plan)
    needs["quality"] = {"result": "failure", "outputs": {"collection_outcome": "failure"}}
    for name in ("unit", "integration", "storage"):
        needs[name]["result"] = "skipped"
    summary = aggregate(
        tmp_path,
        run=RunIdentity(tested_sha=SHA, run_id="123", run_attempt="2"),
        expected=("unit", "integration", "storage", "migrations"),
        results=Results.model_validate(needs),
        plan=plan,
    )
    assert "collection_error" in summary
    assert "migrations: not_run/upstream_blocked" in summary
    assert "artifact_error" not in summary
    assert "assertion" not in summary


@pytest.mark.parametrize("stage", ["unit", "integration", "storage", "migrations"])
def test_no_execution_has_no_assertion_failure(tmp_path: Path, stage: str) -> None:
    value = record(
        evidence(stage=stage, steps={"tests": {"outcome": "skipped"}}), tmp_path / "absent"
    )
    summary = render(value)
    assert "not_run" in summary
    assert "test_failure" not in summary
    if stage == "migrations":
        assert value.assessment.checks_passed is None
        assert not value.assessment.accepted
        assert "Checks: unavailable" in summary


@pytest.mark.parametrize(
    "step",
    ["coverage_upload", "test_upload", "unit_download", "integration_download", "storage_download"],
)
@pytest.mark.parametrize("outcome", ["failure", "cancelled"])
def test_actual_artifact_transfer_errors_remain_visible(step: str, outcome: str) -> None:
    value = evidence(steps={"tests": {"outcome": "failure"}, step: {"outcome": outcome}})
    summary = render(value)
    assert "test_failure" in summary
    assert "artifact_error" in summary


@pytest.mark.parametrize("child", ["", "<failure/>"])
def test_partial_migration_evidence_is_separate_from_assertions(tmp_path: Path, child: str) -> None:
    report = tmp_path / "partial.xml"
    report.write_text(
        '<testsuite><testcase classname="tests.integration.test_migrations" '
        f'name="{MIGRATION_CASES[0]}">{child}</testcase></testsuite>'
    )
    value = record(evidence(stage="migrations"), report)
    assert value.assessment.checks_passed is (not child)
    assert not value.assessment.evidence_valid
    assert not value.assessment.accepted
    assert ("test_failure" in render(value)) is bool(child)


def test_quality_collection_is_the_original_failure() -> None:
    value = evidence(stage="quality", steps={"collection": {"outcome": "failure"}})
    summary = render(value)
    assert "collection_error" in summary
    assert "test_failure" not in summary


@pytest.mark.parametrize("check", ["structure", "lint", "format", "types", "dependencies"])
def test_independent_quality_failures_have_an_explicit_category(check: str) -> None:
    value = evidence(
        stage="quality",
        steps={"collection": {"outcome": "success"}, check: {"outcome": "failure"}},
    )
    assert "quality_failure" in render(value)
    assert "collection_error" not in render(value)


def test_upstream_and_invalid_evidence_are_both_reported() -> None:
    value = evidence(
        stage="coverage",
        assessment=CheckEvidence(
            checks_passed=None, evidence_valid=False, upstream={"unit": Result.FAILURE}
        ),
    )
    summary = render(value)
    assert "upstream_blocked" in summary
    assert "artifact_error" in summary


@pytest.mark.parametrize("outcome", ["success", "failure"])
@pytest.mark.parametrize("content", [None, "<broken"])
def test_executed_tests_require_valid_junit(
    tmp_path: Path, outcome: str, content: str | None
) -> None:
    path = tmp_path / "junit.xml"
    if content is not None:
        path.write_text(content)
    value = record(evidence(steps={"tests": {"outcome": outcome}}), path)
    assert "artifact_error" in render(value)


def test_dependency_upload_failure_is_explicit() -> None:
    value = evidence(
        stage="quality",
        steps={
            "collection": {"outcome": "success"},
            "dependencies_upload": {"outcome": "failure"},
        },
    )
    assert "artifact_error" in render(value)


def test_collection_and_other_quality_failures_are_reported_together() -> None:
    value = evidence(
        stage="quality",
        steps={name: {"outcome": "failure"} for name in ("collection", "lint", "format")},
    )
    summary = render(value)
    assert "collection_error" in summary
    assert "quality_failure" in summary
    assert "<code>lint</code>: failure" in summary
    assert "<code>format</code>: failure" in summary
