"""Development permissions never silence production imports or malformed tool evidence."""

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts.ci_dependencies import TIMEOUT_SECONDS, assess, run


def project(root: Path, *, declared: bool = True) -> Path:
    manifest = '[dependency-groups]\ntest = ["coverage==7.16.0"]\n'
    (root / "pyproject.toml").write_text(
        manifest if declared else "[dependency-groups]\ntest = []\n"
    )
    return root / "deptry.json"


def issue(
    file: str = "scripts/ci_coverage.py", module: str = "coverage", code: str = "DEP004"
) -> dict[str, object]:
    return {
        "error": {"code": code, "message": "ignored prose"},
        "module": module,
        "location": {"file": file, "line": 6, "column": 1},
    }


def test_permit_retains_original_issue_and_exit(tmp_path: Path) -> None:
    report = project(tmp_path)
    report.write_text(json.dumps([issue(), issue(file=str(tmp_path / "scripts/ci_coverage.py"))]))
    result = assess(report, 1, tmp_path)
    assert result.status == "passed"
    assert result.raw_exit == 1
    assert result.permitted
    assert not result.blocking


@pytest.mark.parametrize(
    ("file", "module", "code"),
    [
        ("app/service.py", "coverage", "DEP004"),
        ("scripts/other.py", "coverage", "DEP004"),
        ("scripts/ci_coverage.py", "pytest", "DEP004"),
        ("scripts/ci_coverage.py", "coverage", "DEP001"),
        ("scripts/ci_coverage.py", "coverage", "DEP002"),
        ("scripts/ci_coverage.py", "coverage", "DEP003"),
        ("../scripts/ci_coverage.py", "coverage", "DEP004"),
    ],
)
def test_scope_never_expands_to_other_imports(
    tmp_path: Path, file: str, module: str, code: str
) -> None:
    report = project(tmp_path)
    report.write_text(json.dumps([issue(file, module, code)]))
    result = assess(report, 1, tmp_path)
    assert result.status == "blocked"
    assert not result.permitted


@pytest.mark.parametrize(
    ("raw_exit", "payload"),
    [
        (0, [issue()]),
        (1, []),
        (1, [issue(code="DEP999")]),
    ],
)
def test_contradictory_or_unknown_reports_fail(
    tmp_path: Path, raw_exit: int, payload: object
) -> None:
    report = project(tmp_path)
    report.write_text(json.dumps(payload))
    assert assess(report, raw_exit, tmp_path).status == "invalid_evidence"


def test_missing_damaged_and_undeclared_reports_fail(tmp_path: Path) -> None:
    report = project(tmp_path, declared=False)
    assert assess(report, 0, tmp_path).status == "invalid_evidence"
    report.write_text("{")
    assert assess(report, 1, tmp_path).status == "invalid_evidence"
    report.write_text(json.dumps([issue()]))
    assert assess(report, 1, tmp_path).status == "blocked"
    assert assess(report, 2, tmp_path).status == "tool_error"


@pytest.mark.parametrize(
    "error", [OSError("missing tool"), subprocess.TimeoutExpired("deptry", 90)]
)
def test_tool_failure_retains_failed_assessment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=error))
    output = tmp_path / "evidence"
    result = run(tmp_path, output)
    assert result.status == "tool_error"
    assert (output / "assessment.json").is_file()


def test_clean_report_and_mixed_failures_are_distinct(tmp_path: Path) -> None:
    report = project(tmp_path)
    report.write_text("[]")
    assert assess(report, 0, tmp_path).status == "passed"
    report.write_text(json.dumps([issue(), issue(file="app/service.py")]))
    result = assess(report, 1, tmp_path)
    assert result.status == "blocked"
    assert result.permitted
    assert result.blocking


def test_runner_preserves_raw_report_and_uses_locked_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project(tmp_path)
    output = tmp_path / "evidence"

    def deptry(command: list[str], **options: object) -> subprocess.CompletedProcess[bytes]:
        assert command[0] == str(tmp_path / ".venv/bin/deptry")
        assert options["cwd"] == tmp_path
        assert options["timeout"] == TIMEOUT_SECONDS
        assert not options["check"]
        (output / "deptry.json").write_text(json.dumps([issue()]))
        return subprocess.CompletedProcess(command, 1, b"raw output", b"raw diagnostic")

    monkeypatch.setattr(subprocess, "run", deptry)
    result = run(tmp_path, output)
    assert result.status == "passed"
    assert result.raw_exit == 1
    assert (output / "deptry.json").is_file()
    assert (output / "deptry.log").read_bytes() == b"raw outputraw diagnostic"
    with pytest.raises(FileExistsError):
        run(tmp_path, output)
