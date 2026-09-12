"""Native mypy evidence remains diagnostic and never overrides the process outcome."""

import json
import subprocess
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts.ci_types import (
    MAX_ISSUES,
    TypeIssue,
    TypeReport,
    annotation,
    describe,
    main,
    parse_output,
    run,
)


def message(**updates: object) -> bytes:
    return json.dumps(
        {
            "file": "sample.py",
            "line": 3,
            "column": 0,
            "end_line": 3,
            "end_column": 6,
            "severity": "error",
            "code": "var-annotated",
            "message": "Need an annotation",
            "hint": None,
            **updates,
        }
    ).encode()


@pytest.mark.parametrize(
    ("raw_exit", "payload", "accepted", "valid"),
    [
        (0, b"", True, True),
        (0, message(severity="note", code=None), True, True),
        (1, message(), False, True),
        (0, message(), False, False),
        (1, b"", False, False),
        (1, message(severity="note"), False, False),
        (0, b"broken JSON", False, False),
        (1, message() + b"\nbroken JSON", False, False),
        (1, message(line="3"), False, False),
        (2, b"", False, True),
    ],
)
def test_raw_exit_and_json_must_agree(
    raw_exit: int, payload: bytes, accepted: bool, valid: bool
) -> None:
    report = parse_output(payload, raw_exit)
    assert report.accepted is accepted
    assert report.report_valid is valid
    assert report.raw_exit == raw_exit


def test_limit_prioritizes_errors_without_losing_counts() -> None:
    payload = b"\n".join([message(severity="note")] * (MAX_ISSUES + 1) + [message()])
    report = parse_output(payload, 1)
    assert len(report.issues) == MAX_ISSUES
    assert report.issues[0].severity == "error"
    assert report.errors == 1
    assert report.notes == MAX_ISSUES + 1
    assert "further diagnostics" in describe(report)


def test_github_properties_and_message_are_escaped(tmp_path: Path) -> None:
    path = tmp_path / "sample,part.py"
    path.write_text("value = 1\n")
    issue = TypeIssue.model_validate_json(
        message(
            file=path.name,
            line=1,
            message="100%\r\n::error::fake",
            code="code,unsafe",
        )
    )
    value = annotation(issue, tmp_path)
    assert value is not None
    assert "file=sample%2Cpart.py,line=1,col=1" in value
    assert "100%25%0D%0A::error::fake" in value
    assert "\n" not in value
    assert "code%2Cunsafe" in value
    assert "\n::error::fake" not in describe(parse_output(message(message=issue.message), 1))


@pytest.mark.parametrize(
    "updates",
    [
        {"file": None},
        {"line": -1},
        {"line": None},
        {"file": "../outside.py"},
    ],
)
def test_unknown_or_external_locations_do_not_get_invented_annotations(
    tmp_path: Path, updates: dict[str, object]
) -> None:
    (tmp_path / "sample.py").write_text("value = 1\n")
    assert annotation(TypeIssue.model_validate_json(message(**updates)), tmp_path) is None


def test_missing_column_keeps_line_only_and_summary_escapes_markup(tmp_path: Path) -> None:
    (tmp_path / "sample.py").write_text("value = 1\n")
    issue = TypeIssue.model_validate_json(message(column=-1, message="<script>bad</script>"))
    value = annotation(issue, tmp_path)
    assert value is not None
    assert ",col=" not in value
    report = parse_output(message(message="<script>bad</script>"), 1)
    assert "&lt;script&gt;" in describe(report)
    assert "<script>" not in describe(report)


def test_runner_invokes_one_complete_check_and_keeps_both_streams(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = Mock(return_value=subprocess.CompletedProcess([], 1, message(), b"provider diagnostic"))
    monkeypatch.setattr(subprocess, "run", tool)
    output = tmp_path / "evidence"
    report = run(tmp_path, output)
    assert not report.accepted
    tool.assert_called_once()
    assert tool.call_args.args[0] == [sys.executable, "-m", "mypy", "--output=json"]
    assert tool.call_args.kwargs["cwd"] == tmp_path
    assert (output / "stdout.jsonl").read_bytes() == message()
    assert (output / "stderr.log").read_bytes() == b"provider diagnostic"
    assert (output / "report.json").is_file()
    with pytest.raises(FileExistsError):
        run(tmp_path, output)


@pytest.mark.parametrize(
    "error",
    [
        OSError("unavailable"),
        subprocess.TimeoutExpired("mypy", 180, output=b"partial", stderr=b"interrupted"),
    ],
)
def test_interruption_cannot_publish_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    monkeypatch.setattr(subprocess, "run", Mock(side_effect=error))
    output = tmp_path / "evidence"
    report = run(tmp_path, output)
    assert not report.accepted
    assert report.raw_exit is None
    assert (output / "report.json").is_file()
    if isinstance(error, subprocess.TimeoutExpired):
        assert (output / "stderr.log").read_bytes() == b"interrupted"
        assert report.status == "timeout"
    else:
        assert (output / "stderr.log").read_bytes() == b"unavailable"
        assert report.status == "tool_error"


@pytest.mark.parametrize(
    ("raw_exit", "payload", "expected_exit"),
    [(0, b"", 0), (1, message(), 1), (0, message(), 1), (0, b"invalid", 1), (2, b"", 1)],
)
def test_cli_preserves_failures_even_with_a_successful_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_exit: int,
    payload: bytes,
    expected_exit: int,
) -> None:
    monkeypatch.setattr(sys, "argv", ["ci_types", "--output-dir", str(tmp_path / "evidence")])
    monkeypatch.setattr("scripts.ci_types.run", Mock(return_value=parse_output(payload, raw_exit)))
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == expected_exit


@pytest.mark.parametrize(
    ("source", "expected_exit"),
    [("value: int = 1\n", 0), ('value: int = "wrong"\n', 1)],
    ids=["success", "assignment-error"],
)
def test_real_locked_mypy_cli_and_report_agree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    source: str,
    expected_exit: int,
) -> None:
    manifest = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())
    assert f"mypy=={version('mypy')}" in manifest["dependency-groups"]["dev"]
    (tmp_path / "sample.py").write_text(source)
    (tmp_path / "pyproject.toml").write_text(
        '[tool.mypy]\nfiles = ["sample.py"]\nstrict = true\n'
        'python_version = "3.12"\nincremental = false\n'
    )
    output = tmp_path / "evidence"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["ci_types", "--output-dir", str(output)])
    with pytest.raises(SystemExit) as exit_info:
        main()
    report = TypeReport.model_validate_json((output / "report.json").read_text())
    raw = (output / "stdout.jsonl").read_bytes()
    issues = [TypeIssue.model_validate_json(line) for line in raw.splitlines() if line.strip()]
    assert exit_info.value.code == report.raw_exit == expected_exit
    assert report.status == "completed"
    assert report.report_valid
    assert report.accepted is (expected_exit == 0)
    assert report.errors == expected_exit
    assert report.notes == 0
    assert report.issues == issues
    assert (output / "stderr.log").read_bytes() == b""
    console = capsys.readouterr().out
    assert f"Types: {'PASS' if expected_exit == 0 else 'FAIL'}" in console
    if expected_exit:
        assert len(issues) == 1
        issue = issues[0]
        assert issue.severity == "error"
        assert issue.code == "assignment"
        assert issue.file == "sample.py"
        assert issue.line == 1
        assert "::error file=sample.py,line=1," in console
    else:
        assert not issues
        assert "::error" not in console
