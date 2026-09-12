"""Partial evidence and softened outcomes cannot create successful CI gates."""

import json
from pathlib import Path

import pytest
import yaml

from scripts.ci_policy import Reason, Result, full_plan
from scripts.ci_result import (
    MIGRATION_CASES,
    QUALITY_CHECKS,
    QualityDetails,
    QualitySteps,
    final_result,
    migration_summary,
    partition_summary,
    quality_summary,
)
from tests.tooling.support import condition_holds, needs_for

ROOT = Path(__file__).resolve().parents[2]


def test_quality_outputs_report_all_failures_with_reproduction_commands(tmp_path: Path) -> None:
    sha = "a" * 40
    steps = {name: {"outcome": "success"} for name in ("setup", *QUALITY_CHECKS)}
    for name in ("format", "types"):
        steps[name] = {"outcome": "failure"}
    output = tmp_path / "output"
    summary = tmp_path / "summary"
    assert not quality_summary(json.dumps(steps), summary, output=output, tested_sha=sha)
    details = QualityDetails.model_validate_json(output.read_text().split("=", 1)[1])
    assert details.tested_sha == sha
    assert "quality-format.log" in summary.read_text()
    assert "make typecheck-report ENV=test" in summary.read_text()
    plan = full_plan(sha, Reason.MANUAL)
    needs = needs_for(plan)
    needs["quality"] = {
        "result": "failure",
        "outputs": {"collection_outcome": "success", "quality_details": details.model_dump_json()},
    }
    assert not final_result(plan.model_dump_json(), json.dumps(needs), summary, tested_sha=sha)
    assert "quality / format, quality / types" in summary.read_text()
    assert "| unit | success |" in summary.read_text()
    assert "| build | success |" in summary.read_text()


@pytest.mark.parametrize("problem", ["missing", "invalid", "sha", "contradiction", "wrong_type"])
def test_auxiliary_quality_data_cannot_make_failed_run_pass(tmp_path: Path, problem: str) -> None:
    sha = "a" * 40
    plan = full_plan(sha, Reason.MANUAL)
    needs = needs_for(plan)
    needs["quality"]["result"] = "failure"
    details = QualityDetails(
        tested_sha="b" * 40 if problem == "sha" else sha,
        steps=QualitySteps.model_validate(
            {name: {"outcome": "success"} for name in ("setup", *QUALITY_CHECKS)}
        ),
    )
    if problem != "missing":
        raw = "not json" if problem == "invalid" else details.model_dump_json()
        needs["quality"]["outputs"] = {
            "collection_outcome": "success",
            "quality_details": 42 if problem == "wrong_type" else raw,
        }
    summary = tmp_path / "summary"
    assert not final_result(plan.model_dump_json(), json.dumps(needs), summary, tested_sha=sha)
    assert "FAIL quality" in summary.read_text()
    if problem == "contradiction":
        assert "reporting/upload" in summary.read_text()
    else:
        assert "unavailable" in summary.read_text()


def test_auxiliary_data_does_not_replace_successful_raw_gate(tmp_path: Path) -> None:
    sha = "a" * 40
    plan = full_plan(sha, Reason.MANUAL)
    needs = needs_for(plan)
    needs["quality"]["outputs"] = {
        "collection_outcome": "success",
        "quality_details": QualityDetails(tested_sha=sha, steps=QualitySteps()).model_dump_json(),
    }
    summary = tmp_path / "summary"
    assert final_result(plan.model_dump_json(), json.dumps(needs), summary, tested_sha=sha)
    assert "contradicts raw job success" in summary.read_text()


@pytest.mark.parametrize("first", ["success", "failure", "cancelled", "skipped", None])
def test_quality_raw_outcomes_cannot_be_softened(tmp_path: Path, first: str | None) -> None:
    steps = {name: {"outcome": "success"} for name in ("setup", *QUALITY_CHECKS)}
    if first is None:
        del steps["lint"]
    else:
        steps["lint"] = {"outcome": first, "conclusion": "success"}
    summary = tmp_path / "summary.md"
    assert quality_summary(json.dumps(steps), summary) is (first == "success")
    for name in QUALITY_CHECKS:
        assert f"| {name} |" in summary.read_text()


@pytest.mark.parametrize("outcome", ["failure", "cancelled", "skipped"])
def test_failed_setup_cannot_pass_quality(tmp_path: Path, outcome: str) -> None:
    steps = {"setup": {"outcome": outcome}}
    assert not quality_summary(json.dumps(steps), tmp_path / "summary.md")


@pytest.mark.parametrize("steps", ["{}", "not json", '{"lint":{"outcome":"unknown"}}'])
def test_invalid_or_missing_quality_results_fail(tmp_path: Path, steps: str) -> None:
    assert not quality_summary(steps, tmp_path / "summary.md")


@pytest.mark.parametrize("failed_check", QUALITY_CHECKS)
def test_failure_does_not_hide_remaining_quality_checks(tmp_path: Path, failed_check: str) -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    checks = [
        step for step in workflow["jobs"]["quality"]["steps"] if step.get("id") in QUALITY_CHECKS
    ]
    outcomes = {"setup": {"outcome": "success"}}
    executed = []
    for step in checks:
        assert condition_holds(step["if"], {"steps.setup.outcome": "success"})
        executed.append(step["id"])
        outcomes[step["id"]] = {"outcome": "failure" if step["id"] == failed_check else "success"}
    assert set(executed) == set(QUALITY_CHECKS)
    assert not quality_summary(json.dumps(outcomes), tmp_path / "summary.md")


@pytest.mark.parametrize("missing", MIGRATION_CASES)
def test_each_required_migration_instance_must_exist(tmp_path: Path, missing: str) -> None:
    cases = "".join(
        f'<testcase classname="tests.integration.test_migrations" name="{name}"/>'
        for name in MIGRATION_CASES
        if name != missing
    )
    report, summary = tmp_path / "report.xml", tmp_path / "summary.md"
    report.write_text(f"<testsuites><testsuite>{cases}</testsuite></testsuites>")
    assert not migration_summary(report, summary)
    assert f"Missing: <code>{missing}</code>" in summary.read_text()


@pytest.mark.parametrize("problem", ["duplicate", "single", "similar", "malformed", "extra_failed"])
def test_migration_evidence_cannot_be_partial_or_misidentified(
    tmp_path: Path, problem: str
) -> None:
    cases = [
        f'<testcase classname="tests.integration.test_migrations" name="{name}"/>'
        for name in MIGRATION_CASES
    ]
    if problem == "duplicate":
        cases.append(cases[0])
    if problem == "single":
        cases = cases[:1]
    if problem == "similar":
        cases = [case.replace('test_migrations"', 'test_migrations_fake"') for case in cases]
    if problem == "extra_failed":
        cases.append(
            '<testcase classname="tests.integration.test_migrations" name="extra"><error/></testcase>'
        )
    xml = "<testsuites><testsuite>" + "".join(cases) + "</testsuite></testsuites>"
    report, summary = tmp_path / "report.xml", tmp_path / "summary.md"
    report.write_text("<broken" if problem == "malformed" else xml)
    assert not migration_summary(report, summary)


def test_final_summary_distinguishes_failure_and_blocked_jobs(tmp_path: Path) -> None:
    plan = full_plan("a" * 40, Reason.MANUAL)
    needs = needs_for(plan)
    needs["quality"]["result"] = "failure"
    for name in ("unit", "integration", "storage", "coverage", "build"):
        needs[name]["result"] = "skipped"
    summary = tmp_path / "summary.md"
    assert not final_result(
        plan.model_dump_json(), json.dumps(needs), summary, tested_sha=plan.tested_sha
    )
    assert "| quality | failure |" in summary.read_text()
    assert "| unit | unexpected skip |" in summary.read_text()
    assert "| coverage | blocked by unit, integration, storage |" in summary.read_text()


@pytest.mark.parametrize("outcome", ["skipped", "cancelled"])
def test_unexpected_skip_and_cancel_remain_failures(tmp_path: Path, outcome: str) -> None:
    plan = full_plan("a" * 40, Reason.MANUAL)
    needs = needs_for(plan)
    needs["unit"]["result"] = outcome
    summary = tmp_path / "summary.md"
    assert not final_result(
        plan.model_dump_json(), json.dumps(needs), summary, tested_sha=plan.tested_sha
    )
    assert ("unexpected skip" if outcome == "skipped" else "cancelled") in summary.read_text()


@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled", "skipped"])
@pytest.mark.parametrize(
    "problem", ["none", "missing", "malformed", "empty", "failure", "error", "skipped", "coverage"]
)
def test_partition_requires_complete_passing_evidence(
    tmp_path: Path, outcome: str, problem: str
) -> None:
    report, coverage = tmp_path / "report.xml", tmp_path / ".coverage"
    coverage.write_bytes(b"coverage fixture")
    child = f"<{problem}/>" if problem in {"failure", "error", "skipped"} else ""
    xml = f'<testsuite><testcase name="case">{child}</testcase></testsuite>'
    if problem == "empty":
        xml = "<testsuite/>"
    if problem != "missing":
        report.write_text("<broken" if problem == "malformed" else xml)
    if problem == "coverage":
        coverage = tmp_path / "absent"
    assert partition_summary(report, coverage, Result(outcome), tmp_path / "summary") is (
        outcome == "success" and problem == "none"
    )


def test_selection_must_belong_to_tested_commit(tmp_path: Path) -> None:
    plan = full_plan("a" * 40, Reason.MANUAL)
    assert not final_result(
        plan.model_dump_json(),
        json.dumps(needs_for(plan)),
        tmp_path / "summary",
        tested_sha="b" * 40,
    )


def test_quality_failure_does_not_mislabel_independent_successes(tmp_path: Path) -> None:
    plan = full_plan("a" * 40, Reason.MANUAL)
    needs = needs_for(plan)
    needs["quality"]["result"] = "failure"
    summary = tmp_path / "summary"
    assert not final_result(
        plan.model_dump_json(), json.dumps(needs), summary, tested_sha=plan.tested_sha
    )
    for name in ("unit", "integration", "storage", "coverage", "build"):
        assert f"| {name} | success |" in summary.read_text()


@pytest.mark.parametrize("check", QUALITY_CHECKS)
def test_every_quality_check_is_mandatory_even_with_softened_conclusion(
    tmp_path: Path, check: str
) -> None:
    steps = {name: {"outcome": "success"} for name in ("setup", *QUALITY_CHECKS)}
    steps[check] = {"outcome": "failure", "conclusion": "success"}
    assert not quality_summary(json.dumps(steps), tmp_path / "summary")
    steps.pop(check)
    assert not quality_summary(json.dumps(steps), tmp_path / "missing-summary")


@pytest.mark.parametrize(
    "xml",
    [
        '<wrong><testcase name="case"/></wrong>',
        '<testsuites><testcase name="case"/></testsuites>',
        '<testsuite><wrapper><testcase name="case"/></wrapper></testsuite>',
        '<testsuite><unknown/><testcase name="case"/></testsuite>',
        '<testsuite><properties><error/></properties><testcase name="case"/></testsuite>',
        '<testsuite><system-out><failure/></system-out><testcase name="case"/></testsuite>',
        '<testsuite><system-err><skipped/></system-err><testcase name="case"/></testsuite>',
        '<testsuite><testcase name="case"/><testcase name="case"/></testsuite>',
        '<testsuite><testcase name="case"><failure/><error/></testcase></testsuite>',
        '<testsuite><testcase name="case"><error/><error/></testcase></testsuite>',
        '<testsuite><testcase name="case"><skipped/><failure/></testcase></testsuite>',
        '<testsuite><testcase name="case"><properties><error/></properties></testcase></testsuite>',
        '<testsuite><error/><testcase name="case"/></testsuite>',
        "<testsuite><testcase/></testsuite>",
        '<testsuite tests="2"><testcase name="case"/></testsuite>',
        '<testsuite failures="1"><testcase name="case"/></testsuite>',
        '<testsuite tests="-1"><testcase name="case"/></testsuite>',
        '<testsuite tests="not-an-int"><testcase name="case"/></testsuite>',
    ],
)
def test_invalid_junit_cannot_pass_either_reporter(tmp_path: Path, xml: str) -> None:
    report, coverage = tmp_path / "report.xml", tmp_path / ".coverage"
    report.write_text(xml)
    coverage.write_bytes(b"diagnostic coverage")
    assert not partition_summary(report, coverage, Result.SUCCESS, tmp_path / "tests-summary")
    assert not migration_summary(report, tmp_path / "migration-summary")
    assert "Report defect:" in (tmp_path / "tests-summary").read_text()


def test_multiple_result_nodes_do_not_create_negative_pass_count(tmp_path: Path) -> None:
    report, coverage, summary = (
        tmp_path / "report.xml",
        tmp_path / ".coverage",
        tmp_path / "summary",
    )
    coverage.write_bytes(b"diagnostic coverage")
    report.write_text(
        '<testsuite><testcase name="broken"><failure/><error/></testcase></testsuite>'
    )
    assert not partition_summary(report, coverage, Result.FAILURE, summary)
    text = summary.read_text()
    assert "total=1, passed=0, failures=0, errors=0, skipped=0, invalid=1" in text
    assert "invalid: <code>broken</code>" in text


def test_realistic_double_failure_report_lists_root_causes_without_bodies(tmp_path: Path) -> None:
    report, coverage, summary = (
        tmp_path / "report.xml",
        tmp_path / ".coverage",
        tmp_path / "summary",
    )
    report.write_text("""<testsuites><testsuite tests="3" failures="2" errors="0" skipped="0">
<testcase classname="tests.security.test_mcp_containers" name="test_runtime_api_environment_has_only_allowlisted_keys"><failure message="private-error">private-traceback</failure></testcase>
<testcase classname="tests.security.test_trust_boundary" name="test_api_container_env_has_no_business_credential"><failure>private-traceback</failure></testcase>
<testcase classname="tests.unit" name="passing"/>
</testsuite></testsuites>""")
    coverage.write_bytes(b"diagnostic coverage")
    assert not partition_summary(report, coverage, Result.FAILURE, summary)
    text = summary.read_text()
    assert "total=3, passed=1, failures=2" in text
    assert "test_runtime_api_environment_has_only_allowlisted_keys" in text
    assert "test_api_container_env_has_no_business_credential" in text
    assert "private" not in text


def test_collection_error_is_a_named_error_not_an_empty_run(tmp_path: Path) -> None:
    report, coverage, summary = (
        tmp_path / "report.xml",
        tmp_path / ".coverage",
        tmp_path / "summary",
    )
    report.write_text(
        '<testsuite tests="1" errors="1"><testcase classname="" name="tests/unit/test_import.py"><error>private-import-error</error></testcase></testsuite>'
    )
    assert not partition_summary(report, coverage, Result.FAILURE, summary)
    text = summary.read_text()
    assert "total=1, passed=0, failures=0, errors=1" in text
    assert "error: <code>tests/unit/test_import.py</code>" in text
    assert "Coverage artifact: missing" in text


@pytest.mark.parametrize("state", ["missing", "empty", "unreadable"])
def test_report_file_states_are_distinct(tmp_path: Path, state: str) -> None:
    report = tmp_path / "report.xml"
    if state == "empty":
        report.write_bytes(b"")
    elif state == "unreadable":
        report.mkdir()
    summary = tmp_path / "summary"
    assert not partition_summary(report, tmp_path / ".coverage", Result.FAILURE, summary)
    assert f"JUnit artifact: {state}" in summary.read_text()


@pytest.mark.parametrize("state", ["missing", "empty", "unreadable"])
def test_coverage_file_states_are_distinct(tmp_path: Path, state: str) -> None:
    report, coverage, summary = (
        tmp_path / "report.xml",
        tmp_path / ".coverage",
        tmp_path / "summary",
    )
    report.write_text('<testsuite><testcase name="passing"/></testsuite>')
    if state == "empty":
        coverage.write_bytes(b"")
    elif state == "unreadable":
        coverage.mkdir()
    assert not partition_summary(report, coverage, Result.SUCCESS, summary)
    assert f"Coverage artifact: {state}" in summary.read_text()


@pytest.mark.parametrize("outcome", [Result.FAILURE, Result.CANCELLED, Result.SKIPPED])
def test_partial_successful_xml_cannot_override_raw_execution(
    tmp_path: Path, outcome: Result
) -> None:
    report, coverage, summary = (
        tmp_path / "report.xml",
        tmp_path / ".coverage",
        tmp_path / "summary",
    )
    cases = "".join(
        f'<testcase classname="tests.integration.test_migrations" name="{name}"/>'
        for name in MIGRATION_CASES
    )
    report.write_text(f"<testsuite>{cases}</testsuite>")
    coverage.write_bytes(b"diagnostic coverage")
    assert not partition_summary(report, coverage, outcome, summary)
    assert not migration_summary(report, tmp_path / "migration-summary", outcome)
    assert f"outcome={outcome.value}" in summary.read_text()
    assert "JUnit artifact: present" in summary.read_text()


def test_distinct_parameter_instances_and_nested_suites_are_valid(tmp_path: Path) -> None:
    report, coverage = tmp_path / "report.xml", tmp_path / ".coverage"
    report.write_text("""<testsuites tests="2"><testsuite tests="2" errors="0" failures="0" skipped="0">
<testsuite tests="1"><testcase classname="tests.example" name="case[a]"/></testsuite>
<testcase classname="tests.example" name="case[b]"/>
</testsuite></testsuites>""")
    coverage.write_bytes(b"diagnostic coverage")
    assert partition_summary(report, coverage, Result.SUCCESS, tmp_path / "summary")


def test_diagnostics_are_bounded_escaped_and_never_include_exception_bodies(tmp_path: Path) -> None:
    report, coverage, summary = (
        tmp_path / "report.xml",
        tmp_path / ".coverage",
        tmp_path / "summary",
    )
    suffix = "x" * 400
    cases = "".join(
        f'<testcase name="case-{index}-&lt;script&gt;{suffix}"><failure>private-body</failure></testcase>'
        for index in range(55)
    )
    report.write_text(f"<testsuite>{cases}</testsuite>")
    coverage.write_bytes(b"diagnostic coverage")
    assert not partition_summary(report, coverage, Result.FAILURE, summary)
    text = summary.read_text()
    assert text.count("- failure: ") == 50  # noqa: PLR2004 -- approved display cap.
    assert "Omitted 5 additional entries" in text
    assert "&lt;script&gt;" in text
    assert "<script>" not in text
    assert "private-body" not in text
    assert "x" * 301 not in text
    assert "..." in text
