"""Audit all imports with precise development-tool permissions and retained raw evidence."""

import argparse
import subprocess
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

TIMEOUT_SECONDS = 90


class Rule(StrEnum):
    """Unknown deptry rules require review instead of implicit acceptance."""

    MISSING = "DEP001"
    UNUSED = "DEP002"
    TRANSITIVE = "DEP003"
    DEVELOPMENT = "DEP004"


class DependencyError(BaseModel):
    """Only the rule identifier controls policy."""

    code: Rule


class Location(BaseModel):
    """deptry can omit coordinates for manifest-wide issues."""

    file: str
    line: int | None = Field(default=None, ge=1)
    column: int | None = Field(default=None, ge=1)


class Issue(BaseModel):
    """Projection of the locked deptry JSON format, excluding error prose."""

    error: DependencyError
    module: str
    location: Location


class Manifest(BaseModel):
    """The permit requires the reviewed pinned test dependency to remain declared."""

    groups: dict[str, list[str]] = Field(alias="dependency-groups")


class Assessment(BaseModel):
    """Raw exit and effective policy result are reported separately."""

    status: Literal["passed", "blocked", "invalid_evidence", "tool_error"]
    raw_exit: int | None
    permitted: list[Issue] = Field(default_factory=list)
    blocking: list[Issue] = Field(default_factory=list)


def permitted(issue: Issue, root: Path, manifest: Manifest) -> bool:
    """A module-level ignore must never allow coverage into application code."""
    path = Path(issue.location.file)
    if ".." in path.parts:
        return False
    path = path if path.is_absolute() else root / path
    return (
        issue.error.code is Rule.DEVELOPMENT
        and path == root / "scripts/ci_coverage.py"
        and issue.module == "coverage"
        and "coverage==7.16.0" in manifest.groups.get("test", [])
    )


def assess(report: Path, raw_exit: int, root: Path) -> Assessment:
    """Reject missing, contradictory or unknown evidence before applying exact permits."""
    if raw_exit not in {0, 1}:
        return Assessment(status="tool_error", raw_exit=raw_exit)
    try:
        issues = TypeAdapter(list[Issue]).validate_json(report.read_text())
        manifest = Manifest.model_validate(tomllib.loads((root / "pyproject.toml").read_text()))
    except (OSError, ValueError, ValidationError):
        return Assessment(status="invalid_evidence", raw_exit=raw_exit)
    if bool(raw_exit) != bool(issues):
        return Assessment(status="invalid_evidence", raw_exit=raw_exit)
    allowed = [issue for issue in issues if permitted(issue, root, manifest)]
    blocked = [issue for issue in issues if not permitted(issue, root, manifest)]
    return Assessment(
        status="blocked" if blocked else "passed",
        raw_exit=raw_exit,
        permitted=allowed,
        blocking=blocked,
    )


def run(root: Path, output: Path) -> Assessment:
    """Use the installed locked tool once; a new directory prevents stale-report reuse."""
    output.mkdir(parents=True, exist_ok=False)
    report = output / "deptry.json"
    try:
        result = subprocess.run(  # noqa: S603 -- fixed project executable and explicit argv.
            [
                str(root / ".venv/bin/deptry"),
                ".",
                "--json-output",
                str(report),
                "--enforce-posix-paths",
                "--no-ansi",
            ],
            cwd=root,
            capture_output=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        (output / "deptry.log").write_bytes((error.stdout or b"") + (error.stderr or b""))
        assessment = Assessment(status="tool_error", raw_exit=None)
    except OSError:
        assessment = Assessment(status="tool_error", raw_exit=None)
    else:
        (output / "deptry.log").write_bytes(result.stdout + result.stderr)
        assessment = assess(report, result.returncode, root)
    (output / "assessment.json").write_text(assessment.model_dump_json(indent=2) + "\n")
    print(assessment.model_dump_json(indent=2))
    return assessment


def main() -> None:
    """Keep CI's dependency step blocking while distinguishing scoped tool permissions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(Path.cwd(), args.output.resolve())
    raise SystemExit(0 if result.status == "passed" else 1)


if __name__ == "__main__":
    main()
