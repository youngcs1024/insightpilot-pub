"""The coverage gate must not let a strong directory hide a weak or absent one."""

from pathlib import Path

import pytest

from scripts.check_test_coverage import CoverageReport, DirectoryCoverage, summarize


@pytest.mark.parametrize(
    ("covered", "total", "passed"), [(75, 100, True), (7499, 10000, False), (0, 0, False)]
)
def test_threshold_is_not_rounded(covered: int, total: int, passed: bool) -> None:
    assert (
        DirectoryCoverage(directory="app/core", covered_lines=covered, num_statements=total).passed
        is passed
    )


def test_each_directory_is_independent() -> None:
    root = Path("/project")
    report = CoverageReport.model_validate(
        {
            "files": {
                "app/core/a.py": {"summary": {"covered_lines": 100, "num_statements": 100}},
                "/project/app/services/b.py": {
                    "summary": {"covered_lines": 50, "num_statements": 100}
                },
                "/elsewhere/app/agents/c.py": {
                    "summary": {"covered_lines": 100, "num_statements": 100}
                },
            }
        }
    )
    assert [result.passed for result in summarize(report, root)] == [True, False, False]
