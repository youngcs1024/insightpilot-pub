"""Required current-run evidence cannot silently disappear behind green job labels."""

import json
from pathlib import Path

import pytest

from scripts.ci_diagnostics import DiagnosticRecord, RunIdentity, assess_diagnostics, main
from scripts.ci_policy import Reason, Result, Results, classify, full_plan
from tests.tooling.support import needs_for

SHA = "a" * 40
RUN = RunIdentity(tested_sha=SHA, run_id="123", run_attempt="2")


def passing_unit() -> DiagnosticRecord:
    return DiagnosticRecord(
        **RUN.model_dump(),
        stage="unit",
        steps={name: {"outcome": "success"} for name in ("setup", "tests", "test_report")},
        counts={"passed": 1},
        report_valid=True,
    )


@pytest.mark.parametrize("defect", ["missing", "malformed", "duplicate", "stale", "contradictory"])
def test_green_jobs_cannot_override_invalid_diagnostics(tmp_path: Path, defect: str) -> None:
    value = passing_unit()
    if defect == "stale":
        value.run_attempt = "1"
    if defect == "contradictory":
        value.steps["tests"].outcome = Result.FAILURE
    if defect != "missing":
        (tmp_path / "unit.json").write_text(
            "{" if defect == "malformed" else value.model_dump_json()
        )
    if defect == "duplicate":
        (tmp_path / "duplicate.json").write_text(value.model_dump_json())
    plan = full_plan(SHA, Reason.MANUAL)
    result = assess_diagnostics(
        tmp_path,
        run=RUN,
        expected=("unit",),
        plan=plan,
        results=Results.model_validate(needs_for(plan)),
    )
    assert not result.valid
    assert not result.accepted


def test_valid_record_passes_and_raw_failure_still_blocks(tmp_path: Path) -> None:
    (tmp_path / "unit.json").write_text(passing_unit().model_dump_json())
    plan = full_plan(SHA, Reason.MANUAL)
    needs = needs_for(plan)
    for outcome in ("success", "failure"):
        needs["unit"]["result"] = outcome
        result = assess_diagnostics(
            tmp_path,
            run=RUN,
            expected=("unit",),
            plan=plan,
            results=Results.model_validate(needs),
        )
        assert result.valid
        assert result.accepted is (outcome == "success")


def test_planned_omission_needs_no_artifacts(tmp_path: Path) -> None:
    plan = classify(["README.md"], baseline="b" * 40, tested_sha=SHA)
    assert assess_diagnostics(
        tmp_path, run=RUN, plan=plan, results=Results.model_validate(needs_for(plan))
    ).accepted


def test_cli_rejects_missing_current_run_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = full_plan(SHA, Reason.MANUAL)
    monkeypatch.setattr(
        "sys.argv",
        [
            "ci_diagnostics",
            "--directory",
            str(tmp_path),
            "--tested-sha",
            SHA,
            "--run-id",
            RUN.run_id,
            "--run-attempt",
            RUN.run_attempt,
            "--plan",
            plan.model_dump_json(),
            "--needs",
            json.dumps(needs_for(plan)),
            "--summary",
            str(tmp_path / "summary"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1


@pytest.mark.parametrize("field", ["report_valid", "counts", "steps"])
def test_success_label_requires_actual_test_evidence(tmp_path: Path, field: str) -> None:
    value = passing_unit().model_dump(mode="json")
    value[field] = False if field == "report_valid" else {}
    (tmp_path / "unit.json").write_text(json.dumps(value))
    plan = full_plan(SHA, Reason.MANUAL)
    assert not assess_diagnostics(
        tmp_path,
        run=RUN,
        expected=("unit",),
        plan=plan,
        results=Results.model_validate(needs_for(plan)),
    ).accepted


def test_collection_blocker_allows_missing_records_but_never_accepts(tmp_path: Path) -> None:
    plan = full_plan(SHA, Reason.MANUAL)
    needs = needs_for(plan)
    needs["quality"] = {"result": "failure", "outputs": {"collection_outcome": "failure"}}
    for name in ("unit", "integration", "storage"):
        needs[name]["result"] = "skipped"
    result = assess_diagnostics(
        tmp_path,
        run=RUN,
        expected=("unit", "integration", "storage", "migrations"),
        plan=plan,
        results=Results.model_validate(needs),
    )
    assert result.valid
    assert not result.accepted
    assert "artifact_error" not in result.summary
