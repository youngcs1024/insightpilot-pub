"""Run the locked mypy once and retain structured evidence without weakening its exit."""

import argparse
import subprocess
import sys
from html import escape
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

TIMEOUT_SECONDS = 180
MAX_ISSUES = 50
MAX_MESSAGE = 1000


class TypeIssue(BaseModel):
    """Native mypy JSON coordinates may be missing or negative when unavailable."""

    model_config = ConfigDict(strict=True)

    file: str | None = None
    line: int | None = None
    column: int | None = None
    code: str | None = None
    severity: Literal["error", "note"]
    message: str


class TypeReport(BaseModel):
    """A bounded display never replaces full raw output or the original exit code."""

    status: Literal["completed", "timeout", "tool_error"] = "completed"
    raw_exit: int | None = None
    report_valid: bool = False
    errors: int = Field(default=0, ge=0)
    notes: int = Field(default=0, ge=0)
    issues: list[TypeIssue] = Field(default_factory=list, max_length=MAX_ISSUES)

    @property
    def accepted(self) -> bool:
        """Only a completed, consistent, error-free successful invocation may pass."""
        return (
            self.status == "completed"
            and self.raw_exit == 0
            and self.report_valid
            and self.errors == 0
        )


def parse_output(stdout: bytes, raw_exit: int) -> TypeReport:
    """Parse every JSON line, prioritize errors, and fail closed on partial corruption."""
    report = TypeReport(raw_exit=raw_exit, report_valid=True)
    errors: list[TypeIssue] = []
    notes: list[TypeIssue] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            issue = TypeIssue.model_validate_json(line)
        except ValidationError:
            report.report_valid = False
            continue
        issue.message = issue.message[:MAX_MESSAGE]
        if issue.severity == "error":
            report.errors += 1
            target = errors
        else:
            report.notes += 1
            target = notes
        if len(target) < MAX_ISSUES:
            target.append(issue)
    report.issues = (errors + notes)[:MAX_ISSUES]
    if raw_exit not in {0, 1}:
        report.status = "tool_error"
    elif (raw_exit == 0) != (report.errors == 0):
        report.report_valid = False
    return report


def command_escape(value: str, *, property_value: bool = False) -> str:
    """Escape GitHub command data before encoding property separators."""
    value = value.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    if property_value:
        value = value.replace(":", "%3A").replace(",", "%2C")
    return value


def annotation(issue: TypeIssue, root: Path) -> str | None:
    """Only annotate an actual repository location; unknown coordinates stay in logs."""
    if not issue.file or issue.line is None or issue.line < 1:
        return None
    path = Path(issue.file)
    try:
        relative = (
            (path if path.is_absolute() else root / path).resolve().relative_to(root.resolve())
        )
    except (OSError, ValueError):
        return None
    if not (root / relative).is_file():
        return None
    fields = f"file={command_escape(relative.as_posix(), property_value=True)},line={issue.line}"
    if issue.column is not None and issue.column >= 0:
        fields += f",col={issue.column + 1}"
    level = "error" if issue.severity == "error" else "notice"
    title = command_escape(f"mypy [{issue.code or 'no-code'}]", property_value=True)
    return f"::{level} {fields},title={title}::{command_escape(issue.message)}"


def describe(report: TypeReport) -> str:
    """Render bounded, escaped detail inside the current run's quality summary."""
    rows = [
        f"Types: {'PASS' if report.accepted else 'FAIL'}; status={report.status}; "
        f"raw exit={report.raw_exit}; report valid={report.report_valid}; "
        f"errors={report.errors}, notes={report.notes}."
    ]
    for issue in report.issues:
        location = issue.file or "location unavailable"
        if issue.line is not None and issue.line > 0:
            location += f":{issue.line}"
            if issue.column is not None and issue.column >= 0:
                location += f":{issue.column + 1}"
        rows.append(
            f"- <code>{summary_escape(location)}</code> "
            f"[{summary_escape(issue.code or 'no-code')}] {summary_escape(issue.message)}"
        )
    omitted = report.errors + report.notes - len(report.issues)
    if omitted:
        rows.append(f"- {omitted} further diagnostics retained in raw output.")
    return "\n".join(rows)


def summary_escape(value: str) -> str:
    """Keep source-derived text on one log line and outside HTML markup."""
    return escape(value).replace("\r", "&#13;").replace("\n", "&#10;")


def run(root: Path, output: Path) -> TypeReport:
    """The Make entrypoint selects this interpreter and working directory through uv."""
    output.mkdir(parents=True, exist_ok=False)
    stdout, stderr = b"", b""
    try:
        # Selected interpreter, fixed module/flags, no shell.
        result = subprocess.run(
            [sys.executable, "-m", "mypy", "--output=json"],
            cwd=root,
            capture_output=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        stdout, stderr = error.stdout or b"", error.stderr or b""
        report = TypeReport(status="timeout")
    except OSError as error:
        stderr = str(error).encode()
        report = TypeReport(status="tool_error")
    else:
        stdout, stderr = result.stdout, result.stderr
        report = parse_output(stdout, result.returncode)
    (output / "stdout.jsonl").write_bytes(stdout)
    (output / "stderr.log").write_bytes(stderr)
    (output / "report.json").write_text(report.model_dump_json(indent=2) + "\n")
    print(describe(report))
    for issue in report.issues:
        rendered = annotation(issue, root)
        if rendered is not None:
            print(rendered)
    return report


def main() -> None:
    """Preserve the raw failure even when diagnostics and uploads are successful."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(Path.cwd(), args.output_dir.resolve())
    raise SystemExit(0 if report.accepted else 1)


if __name__ == "__main__":
    main()
