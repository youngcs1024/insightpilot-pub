"""Render only already-sanitized server metadata into deterministic prompt text."""

from app.schemas.schema_catalog import BUSINESS_TABLES, SemanticType
from app.schemas.schema_tools import ColumnSchema, TableSchema


def render_column(column: ColumnSchema) -> list[str]:
    """Include all authored semantics while excluding PII samples unconditionally."""
    markers = " [PK]" if column.is_primary_key else ""
    if column.fk_table:
        markers += f" [FK → {column.fk_table}.{column.fk_column}]"
    markers += " [NULL]" if column.nullable else " [NOT NULL]"
    lines = [f"  - {column.column_name} ({column.sql_type}){markers}: {column.description}"]
    if column.allowed_values:
        lines.append(
            "      [Allowed: "
            + " | ".join(f"{k}={v}" for k, v in sorted(column.allowed_values.items()))
            + "]"
        )
    if column.sample_values and not column.is_pii:
        lines.append(
            "      [Examples: " + " | ".join(v[:50] for v in column.sample_values[:3]) + "]"
        )
    lines.extend("      " + note for note in column.notes)
    return lines


def render_catalog(
    tables: list[TableSchema]
) -> str:
    """Use live column ordering; validation must have succeeded before this call."""
    semantic = {t.table_name: t for t in tables}
    lines: list[str] = []
    for name in BUSINESS_TABLES:
        if name not in semantic:
            continue
        table = semantic[name]
        columns = {c.column_name: c for c in table.columns}
        lines.extend([f"Table: {name} — {table.description}", "Columns:"])
        for column in sorted(table.columns, key=lambda c: c.ordinal_position):
            lines.extend(render_column(columns[column.column_name]))
        checks = {
            constraint.name: constraint
            for column in table.columns
            for constraint in column.constraints
            if constraint.kind in {"u", "c"}
            and not (column.semantic_type is SemanticType.ENUM and len(constraint.columns) == 1)
        }
        if checks:
            lines.extend(
                ["Constraints:", *("  - " + checks[key].definition for key in sorted(checks))]
            )
        if table.notes:
            lines.extend(["Notes:", *("  - " + note for note in table.notes)])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n" if lines else ""
