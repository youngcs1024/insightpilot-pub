"""Typed, shared JUnit inspection that never treats partial evidence as acceptance."""

import stat
import xml.etree.ElementTree as ET
from collections import Counter
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

MAX_JUNIT_DEPTH = 64


class FileState(StrEnum):
    """Distinguish absent, empty and unreadable artifacts without exception prose."""

    PRESENT = "present"
    MISSING = "missing"
    EMPTY = "empty"
    UNREADABLE = "unreadable"


class CaseOutcome(StrEnum):
    """Exactly one category per testcase, including contradictory result nodes."""

    PASSED = "passed"
    FAILURE = "failure"
    ERROR = "error"
    SKIPPED = "skipped"
    INVALID = "invalid"


class ReportProblem(StrEnum):
    """Finite report defects; raw XML and exception text never enter summaries."""

    MALFORMED = "malformed XML"
    STRUCTURE = "invalid JUnit structure"
    DUPLICATE = "duplicate testcase"
    CONTRADICTORY = "multiple testcase outcomes"
    UNNAMED = "missing testcase name"
    COUNTS = "inconsistent suite counts"
    NO_CASES = "no testcases"


class TestCase(BaseModel):
    """Only identities and categories survive XML projection, never failure bodies."""

    classname: str
    name: str
    outcome: CaseOutcome

    @property
    def identity(self) -> tuple[str, str]:
        """Parameter suffixes remain part of the identity."""
        return self.classname, self.name


class JunitReport(BaseModel):
    """Malformed reports may retain diagnostic cases but cannot pass either gate."""

    file_state: FileState
    cases: list[TestCase] = Field(default_factory=list)
    problems: list[ReportProblem] = Field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Structural validity is independent of individual test success."""
        return self.file_state == FileState.PRESENT and bool(self.cases) and not self.problems


def file_state(path: Path) -> FileState:
    """Check artifact availability; coverage content validity remains coverage.py's job."""
    try:
        info = path.stat()
    except FileNotFoundError:
        return FileState.MISSING
    except OSError:
        return FileState.UNREADABLE
    if not stat.S_ISREG(info.st_mode):
        return FileState.UNREADABLE
    return FileState.PRESENT if info.st_size else FileState.EMPTY


def parse_case(element: ET.Element, report: JunitReport) -> TestCase:
    """Collection errors and ordinary failures use the same single-case accounting."""
    outcomes = [child.tag for child in element if child.tag in {"failure", "error", "skipped"}]
    outcome = CaseOutcome(outcomes[0]) if outcomes else CaseOutcome.PASSED
    name = element.get("name", "")
    if len(outcomes) > 1:
        report.problems.append(ReportProblem.CONTRADICTORY)
        outcome = CaseOutcome.INVALID
    if not name:
        report.problems.append(ReportProblem.UNNAMED)
        outcome = CaseOutcome.INVALID
    allowed = {"failure", "error", "skipped", "properties", "system-out", "system-err"}
    if any(child.tag not in allowed or invalid_case_child(child) for child in element):
        report.problems.append(ReportProblem.STRUCTURE)
    return TestCase(classname=element.get("classname", ""), name=name, outcome=outcome)


def invalid_case_child(element: ET.Element) -> bool:
    """Result nodes cannot be hidden inside properties or another result node."""
    reserved = {"testcase", "testsuite", "testsuites", "failure", "error", "skipped"}
    return any(child.tag in reserved for child in element.iter() if child is not element)


def check_counts(element: ET.Element, cases: list[TestCase], report: JunitReport) -> None:
    """Do not trust stale aggregate attributes when individual cases say otherwise."""
    counts = Counter(case.outcome for case in cases)
    expected = {
        "tests": len(cases),
        "failures": counts[CaseOutcome.FAILURE],
        "errors": counts[CaseOutcome.ERROR],
        "skipped": counts[CaseOutcome.SKIPPED],
    }
    for key, value in expected.items():
        declared = element.get(key)
        if declared is None:
            continue
        try:
            matches = int(declared) == value and int(declared) >= 0
        except ValueError:
            matches = False
        if not matches:
            report.problems.append(ReportProblem.COUNTS)


def suite_cases(element: ET.Element, report: JunitReport, depth: int = 0) -> list[TestCase]:
    """Accept standard suite nesting, not arbitrary wrappers that hide result nodes."""
    if element.tag not in {"testsuites", "testsuite"} or depth >= MAX_JUNIT_DEPTH:
        report.problems.append(ReportProblem.STRUCTURE)
        return []
    cases: list[TestCase] = []
    for child in element:
        if child.tag in {"testsuites", "testsuite"}:
            cases.extend(suite_cases(child, report, depth + 1))
        elif child.tag == "testcase" and element.tag == "testsuite":
            cases.append(parse_case(child, report))
        elif child.tag not in {"properties", "system-out", "system-err"} or invalid_case_child(
            child
        ):
            report.problems.append(ReportProblem.STRUCTURE)
    check_counts(element, cases, report)
    return cases


def read_junit(path: Path) -> JunitReport:
    """Parse once for both test and migration reporting, with bounded safe diagnostics."""
    report = JunitReport(file_state=file_state(path))
    if report.file_state != FileState.PRESENT:
        return report
    try:
        root = ET.parse(path).getroot()  # noqa: S314 -- locally generated pytest XML only.
    except OSError:
        report.file_state = FileState.UNREADABLE
        return report
    except ET.ParseError:
        report.problems.append(ReportProblem.MALFORMED)
        return report
    report.cases = suite_cases(root, report)
    identities = Counter(case.identity for case in report.cases)
    if any(count > 1 for count in identities.values()):
        report.problems.append(ReportProblem.DUPLICATE)
    if not report.cases:
        report.problems.append(ReportProblem.NO_CASES)
    report.problems = list(dict.fromkeys(report.problems))
    return report
