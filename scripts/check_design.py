"""Check the fixed Step 0.7 Markdown contract without executing SQL or loading settings."""

import argparse
import re
import sys
from collections import Counter
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, Field

TABLES = frozenset(
    {
        "regions",
        "promotions",
        "products",
        "customers",
        "orders",
        "order_items",
        "refunds",
        "inventory",
    }
)
TRAPS = frozenset(f"T{number}" for number in range(1, 9))
MIN_CASES = 3
MIN_DELTA = Decimal("0.05")
TABLE_FIELDS = ("Rows", "Generation", "DDL", "Indexes")
TRAP_FIELDS = (
    "Mechanism",
    "Slice",
    "Naive query",
    "Correct query",
    "Target",
    "Acceptance",
    "Delta",
    "Eval cases",
)
CASE_PATTERN = re.compile(r"`(nl2sql-[a-z]+-\d{3})`")
RANGE_PATTERN = re.compile(r"relative delta in \[(\d+(?:\.\d+)?), (\d+(?:\.\d+)?)\]")


class Section(BaseModel):
    """A heading and its source location, including fenced text for field validation."""

    title: str
    line: int = Field(ge=1)
    body: str


class DesignIssue(BaseModel):
    """One actionable document diagnostic."""

    line: int = Field(ge=1)
    section: str
    message: str


class DesignReport(BaseModel):
    """Structural findings; success does not imply SQL or dataset acceptance."""

    issues: list[DesignIssue] = Field(default_factory=list)

    def add(self, section: Section, message: str) -> None:
        """Attach a diagnostic to its enclosing section."""
        self.issues.append(DesignIssue(line=section.line, section=section.title, message=message))


def _outside_fences(source: str) -> str:
    """Blank code blocks while preserving line numbers for Markdown structure."""
    lines: list[str] = []
    fence: str | None = None
    for line in source.splitlines():
        stripped = line.strip()
        if fence is None and stripped.startswith(("```", "~~~")):
            fence = stripped[:3]
            lines.append("")
        elif fence is not None:
            if stripped == fence:
                fence = None
            lines.append("")
        else:
            lines.append(line)
    return "\n".join(lines)


def _sections(source: str) -> list[Section]:
    lines = source.splitlines()
    headings = [
        (number, line.removeprefix("### "))
        for number, line in enumerate(_outside_fences(source).splitlines())
        if line.startswith("### ") or line.startswith("## ")
    ]
    result: list[Section] = []
    for position, (number, title) in enumerate(headings):
        end = headings[position + 1][0] if position + 1 < len(headings) else len(lines)
        result.append(
            Section(title=title, line=number + 1, body="\n".join(lines[number + 1 : end]))
        )
    return result


def _fields(section: Section, names: tuple[str, ...], report: DesignReport) -> dict[str, str]:
    lines = section.body.splitlines()
    structure = _outside_fences(section.body)
    matches = list(re.finditer(r"^([A-Za-z ]+):[ \t]*(.*)$", structure, re.M))
    fields: dict[str, str] = {}
    for position, match in enumerate(matches):
        name = match[1]
        if name not in names:
            continue
        if name in fields:
            report.add(section, f"duplicate field: {name}")
        # Fence blanking changes character offsets, so use the structural text for line offsets.
        first = structure[: match.start()].count("\n")
        end = (
            structure[: matches[position + 1].start()].count("\n")
            if position + 1 < len(matches)
            else len(lines)
        )
        fields[name] = "\n".join(lines[first:end]).partition(":")[2].strip()
    for name in names:
        if not fields.get(name):
            report.add(section, f"missing or empty field: {name}")
    return fields


def _sql(field: str) -> str:
    blocks = re.findall(r"^```sql\s*\n(.*?)^```\s*$", field, re.M | re.S)
    return blocks[0].strip() if len(blocks) == 1 else ""


def _check_table(section: Section, report: DesignReport) -> None:
    name = section.title.removeprefix("Table: ")
    fields = _fields(section, TABLE_FIELDS, report)
    ddl = _sql(fields.get("DDL", ""))
    declarations = re.findall(r"CREATE TABLE biz\.(\w+)\s*\(", ddl)
    if declarations != [name] or not ddl.endswith(");"):
        report.add(section, "DDL must contain one complete CREATE TABLE for this biz table")
    if not re.search(r"CONSTRAINT pk_\w+ PRIMARY KEY", ddl):
        report.add(section, "DDL must include a named primary key")
    if not re.search(r"\d", fields.get("Rows", "")):
        report.add(section, "Rows must contain a numeric count")
    indexes = _sql(fields.get("Indexes", ""))
    index_tables = re.findall(r"CREATE INDEX \w+ ON biz\.(\w+)\s*\(", indexes)
    if not index_tables or set(index_tables) != {name} or not indexes.endswith(";"):
        report.add(section, "Indexes must contain complete CREATE INDEX SQL for this table")


def _check_queries(section: Section, fields: dict[str, str], report: DesignReport) -> None:
    queries = [_sql(fields.get(name, "")) for name in ("Naive query", "Correct query")]
    for name, query in zip(("Naive query", "Correct query"), queries, strict=True):
        if (
            not re.match(r"(?:SELECT|WITH)\b", query)
            or not query.endswith(";")
            or query.count(";") != 1
            or "..." in query
            or "…" in query
        ):
            report.add(section, f"{name} must contain one complete fenced SELECT/WITH query")
    if queries[0] and queries[0] == queries[1]:
        report.add(section, "naive and correct queries must differ")


def _check_numbers(section: Section, fields: dict[str, str], report: DesignReport) -> None:
    target = fields.get("Target", "")
    if not all(re.search(rf"\b{label}\s+\d", target) for label in ("naive", "correct")):
        report.add(section, "Target must quantify both naive and correct values")
    if "not measured" not in target or not re.search(r"\d+(?:\.\d+)?%", target):
        report.add(section, "Target must state a percentage and 'not measured'")
    bounds = RANGE_PATTERN.search(fields.get("Acceptance", ""))
    if bounds is None or not MIN_DELTA <= Decimal(bounds[1]) <= Decimal(bounds[2]):
        report.add(
            section, "Acceptance needs ordered relative delta bounds with lower bound >= 0.05"
        )
    delta = fields.get("Delta", "")
    if not all(token in delta for token in ("naive", "correct", "abs(", "/", "minimum 0.05")):
        report.add(section, "Delta must define the relative difference formula and minimum 0.05")


def _check_trap(section: Section, report: DesignReport) -> list[str]:
    fields = _fields(section, TRAP_FIELDS, report)
    _check_queries(section, fields, report)
    _check_numbers(section, fields, report)
    cases = CASE_PATTERN.findall(fields.get("Eval cases", ""))
    if len(set(cases)) < MIN_CASES:
        report.add(section, "Eval cases must contain at least three distinct nl2sql case IDs")
    if len(cases) != len(set(cases)):
        report.add(section, "duplicate eval case ID within trap")
    return cases


def _check_membership(
    sections: list[Section], prefix: str, expected: frozenset[str], report: DesignReport
) -> None:
    counts = Counter(section.title.removeprefix(prefix) for section in sections)
    root = Section(title="Document", line=1, body="")
    for name in sorted(expected - counts.keys()):
        report.add(root, f"missing section: {prefix}{name}")
    for section in sections:
        name = section.title.removeprefix(prefix)
        if name not in expected:
            report.add(section, "unexpected section identifier")
        if counts[name] > 1:
            report.add(section, "duplicate section identifier")


def check_design(source: str) -> DesignReport:
    """Validate required sections, SQL blocks, numeric contracts and eval identifiers."""
    report = DesignReport()
    sections = _sections(source)
    tables = [section for section in sections if section.title.startswith("Table: ")]
    traps = [section for section in sections if section.title.startswith("Trap: ")]
    _check_membership(tables, "Table: ", TABLES, report)
    _check_membership(traps, "Trap: ", TRAPS, report)
    for section in tables:
        _check_table(section, report)
    seen: set[str] = set()
    for section in traps:
        cases = _check_trap(section, report)
        if seen.intersection(cases):
            report.add(section, "eval case ID reused across traps")
        seen.update(cases)
    return report


def main(argv: list[str] | None = None) -> int:
    """Print source diagnostics; return nonzero for unreadable or invalid documents."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    path: Path = parser.parse_args(argv).path
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        print(f"{path}:1: cannot read UTF-8 design document", file=sys.stderr)
        return 1
    report = check_design(source)
    for issue in report.issues:
        print(f"{path}:{issue.line}: {issue.section}: {issue.message}", file=sys.stderr)
    if report.issues:
        return 1
    print("PASS: 8 tables, 8 traps, SQL blocks, numeric targets and eval mappings (structure only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
