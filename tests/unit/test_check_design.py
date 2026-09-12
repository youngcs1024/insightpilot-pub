"""Regression tests for the document gate, not for generated business data."""

from pathlib import Path

import pytest

from scripts.check_design import TABLES, TRAPS, check_design, main
from tests.design_support import synthetic_design


@pytest.fixture
def source() -> str:
    return synthetic_design()


def test_synthetic_design_passes(source: str) -> None:
    assert not check_design(source).issues


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("### Table: regions", "### Removed: regions", "missing section: Table: regions"),
        ("### Trap: T8", "### Removed: T8", "missing section: Trap: T8"),
        ("### Table: regions", "### Table: unknown", "unexpected section identifier"),
        ("### Trap: T8", "### Trap: T9", "unexpected section identifier"),
        ("### Table: inventory", "### Table: regions", "duplicate section identifier"),
        ("### Trap: T8", "### Trap: T7", "duplicate section identifier"),
        ("Rows: 5 exactly.", "Rows:", "missing or empty field: Rows"),
        ("Generation: stable IDs", "Missing: stable IDs", "missing or empty field: Generation"),
        ("CREATE TABLE biz.regions", "CREATE TABLE biz.wrong", "one complete CREATE TABLE"),
        ("CONSTRAINT pk_regions PRIMARY KEY", "PRIMARY KEY", "named primary key"),
        ("CREATE INDEX ix_regions_renamed_from", "-- missing index", "complete CREATE INDEX"),
        ("Naive query:", "Missing query:", "missing or empty field: Naive query"),
        ("Mechanism:", "Missing:", "missing or empty field: Mechanism"),
        ("Slice:", "Missing:", "missing or empty field: Slice"),
        ("Target:", "Missing:", "missing or empty field: Target"),
        ("Indexes:", "Missing:", "missing or empty field: Indexes"),
        ("Rows: 5 exactly.", "Rows: unknown", "numeric count"),
        (
            "AND NOT c.is_test_account;",
            "AND NOT c.is_test_account; SELECT 1;",
            "complete fenced SELECT",
        ),
        ("Correct query:", "Missing query:", "missing or empty field: Correct query"),
        ("SELECT SUM(o.gross_amount", "SELECT ... SUM(o.gross_amount", "complete fenced SELECT"),
        ("Target: naive 9,000,000", "Target: naive unknown", "quantify both naive and correct"),
        (
            "Design target, not measured.",
            "Design target, measured.",
            "percentage and 'not measured'",
        ),
        ("relative delta in [0.05, 0.20]", "no bounds", "ordered relative delta bounds"),
        ("relative delta in [0.05, 0.20]", "relative delta in [0.01, 0.20]", "lower bound"),
        ("relative delta in [0.05, 0.20]", "relative delta in [0.20, 0.05]", "ordered relative"),
        ("Delta: abs(naive - correct) / abs(correct)", "Delta: undefined", "relative difference"),
        ("`nl2sql-gmv-001`", "unassigned", "at least three distinct"),
        ("`nl2sql-gmv-007`", "`nl2sql-gmv-001`", "duplicate eval case ID"),
        ("`nl2sql-refund-001`", "`nl2sql-gmv-001`", "reused across traps"),
    ],
)
def test_contract_damage_is_rejected(source: str, old: str, new: str, message: str) -> None:
    assert old in source
    report = check_design(source.replace(old, new, 1))
    assert any(message in issue.message for issue in report.issues)
    assert all(issue.line >= 1 for issue in report.issues)


def test_empty_document_reports_all_missing_sections() -> None:
    assert len(check_design("").issues) == len(TABLES) + len(TRAPS)


def test_heading_inside_code_cannot_supply_missing_trap(source: str) -> None:
    source = source.replace("### Trap: T8", "```text\n### Trap: T8\n```")
    assert any(
        "missing section: Trap: T8" in issue.message for issue in check_design(source).issues
    )


def test_duplicate_field_is_rejected(source: str) -> None:
    source = source.replace("Rows: 5 exactly.", "Rows: 5 exactly.\n\nRows: 5 again.")
    assert any("duplicate field: Rows" in issue.message for issue in check_design(source).issues)


def test_identical_queries_are_rejected(source: str) -> None:
    source = source.replace(
        "AND NOT c.is_test_account AND o.status <> 'cancelled';", "AND NOT c.is_test_account;", 1
    )
    assert any("queries must differ" in issue.message for issue in check_design(source).issues)


def test_cli_success(tmp_path: Path, capsys: pytest.CaptureFixture[str], source: str) -> None:
    path = tmp_path / "synthetic.md"
    path.write_text(source, encoding="utf-8")
    assert main([str(path)]) == 0
    assert "structure only" in capsys.readouterr().out


def test_cli_invalid_document(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "invalid.md"
    path.write_text("# Empty design\n", encoding="utf-8")
    assert main([str(path)]) == 1
    assert f"{path}:1: Document: missing section" in capsys.readouterr().err


@pytest.mark.parametrize("kind", ["missing", "directory", "invalid_utf8"])
def test_cli_read_failure(tmp_path: Path, capsys: pytest.CaptureFixture[str], kind: str) -> None:
    path = tmp_path / "input.md"
    if kind == "directory":
        path.mkdir()
    elif kind == "invalid_utf8":
        path.write_bytes(b"\xff")
    assert main([str(path)]) == 1
    assert "cannot read UTF-8 design document" in capsys.readouterr().err


def test_crlf_document_passes(source: str) -> None:
    assert not check_design(source.replace("\n", "\r\n")).issues
