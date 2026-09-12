"""Synthetic input for the optional Markdown contract parser's regression tests."""

from scripts.check_design import TABLES

REFUND_TRAP = 2


def synthetic_design() -> str:
    """Build minimal parser input without reading or reproducing private documents."""
    sections = ["# Synthetic parser fixture"]
    sections.extend(
        (
            f"### Table: {name}\n\n"
            "Rows: 5 exactly.\n\nGeneration: stable IDs\n\n"
            f"DDL:\n```sql\nCREATE TABLE biz.{name} (\n"
            f"id INTEGER, CONSTRAINT pk_{name} PRIMARY KEY (id)\n);\n```\n\n"
            f"Indexes:\n```sql\nCREATE INDEX ix_{name}_renamed_from "
            f"ON biz.{name} (id);\n```"
        )
        for name in sorted(TABLES)
    )
    for number in range(1, 9):
        family = "gmv" if number == 1 else "refund" if number == REFUND_TRAP else "synthetic"
        first = 1 if number <= REFUND_TRAP else number * 10
        cases = ", ".join(f"`nl2sql-{family}-{first + offset:03d}`" for offset in (0, 3, 6))
        query = "SELECT SUM(o.gross_amount) FROM biz.orders o JOIN biz.customers c ON c.id = o.id"
        sections.append(
            f"### Trap: T{number}\n\n"
            "Mechanism: synthetic filter comparison.\n\nSlice: synthetic rows.\n\n"
            f"Naive query:\n```sql\n{query} WHERE TRUE AND NOT c.is_test_account;\n```\n\n"
            f"Correct query:\n```sql\n{query} WHERE TRUE "
            "AND NOT c.is_test_account AND o.status <> 'cancelled';\n```\n\n"
            "Target: naive 9,000,000; correct 10,000,000; 10%. Design target, not measured.\n\n"
            "Acceptance: relative delta in [0.05, 0.20]\n\n"
            "Delta: abs(naive - correct) / abs(correct), minimum 0.05\n\n"
            f"Eval cases: {cases}"
        )
    return "\n\n".join(sections) + "\n"
