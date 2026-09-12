"""Check each engineering-standard directory independently in a coverage JSON report."""

import argparse
from pathlib import Path

from pydantic import BaseModel, Field

DIRECTORIES = ("app/core", "app/services", "app/agents")
MINIMUM_PERCENT = 75


class LineSummary(BaseModel):
    """Coverage.py statement counts, independent of presentation rounding."""

    covered_lines: int = Field(ge=0)
    num_statements: int = Field(ge=0)


class FileCoverage(BaseModel):
    """Only the fields used by this gate are parsed."""

    summary: LineSummary


class CoverageReport(BaseModel):
    """Typed input contract for coverage.py JSON."""

    files: dict[str, FileCoverage]


class DirectoryCoverage(LineSummary):
    """One directory's independently evaluated line coverage."""

    directory: str

    @property
    def percent(self) -> float:
        """Empty/missing directories cannot satisfy the gate."""
        return 100 * self.covered_lines / self.num_statements if self.num_statements else 0

    @property
    def passed(self) -> bool:
        """Compare counts without rounding a subthreshold value up to the minimum."""
        return (
            bool(self.num_statements)
            and self.covered_lines * 100 >= self.num_statements * MINIMUM_PERCENT
        )


def summarize(report: CoverageReport, root: Path) -> list[DirectoryCoverage]:
    """Normalize coverage's relative or absolute paths against the project root."""
    normalized: dict[str, FileCoverage] = {}
    for name, coverage in report.files.items():
        path = Path(name)
        if path.is_absolute():
            if not path.is_relative_to(root):
                continue
            path = path.relative_to(root)
        normalized[path.as_posix()] = coverage
    return [
        DirectoryCoverage(
            directory=directory,
            covered_lines=sum(
                value.summary.covered_lines
                for name, value in normalized.items()
                if name.startswith(directory + "/")
            ),
            num_statements=sum(
                value.summary.num_statements
                for name, value in normalized.items()
                if name.startswith(directory + "/")
            ),
        )
        for directory in DIRECTORIES
    ]


def main() -> None:
    """Print each directory's result and fail unless all three meet the standard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    arguments = parser.parse_args()
    report = CoverageReport.model_validate_json(arguments.report.read_text())
    results = summarize(report, Path(__file__).resolve().parents[1])
    for result in results:
        print(
            f"{result.directory}: {result.covered_lines}/{result.num_statements} ({result.percent:.2f}%) {'PASS' if result.passed else 'FAIL'}; required >= {MINIMUM_PERCENT}%"
        )
    raise SystemExit(0 if all(result.passed for result in results) else 1)


if __name__ == "__main__":
    main()
