"""Pure comparison and rendering of typed metadata, with no database or HTTP I/O."""

from collections.abc import Sequence

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from app.core.errors import SchemaMetadataError
from app.schemas.schema_catalog import (
    BUSINESS_TABLES,
    BusinessSchemaResponse,
    DriftKind,
    PhysicalColumn,
    SchemaConstraint,
    SchemaDrift,
    SchemaDriftReport,
    SemanticColumn,
    SemanticTable,
    SemanticType,
)
from app.schemas.schema_tools import ColumnSchema, TableSchema


def enum_keys(column: SemanticColumn | ColumnSchema) -> set[str]:
    """Extract literal enum values from PostgreSQL's single-column CHECK expression."""
    if column.sql_type == "boolean":
        return {"true", "false"}
    values: set[str] = set()
    for constraint in column.constraints:
        if constraint.kind != "c" or constraint.columns != [column.column_name]:
            continue
        try:
            expression = sqlglot.parse_one(
                constraint.definition.removeprefix("CHECK "), read="postgres"
            )
        except ParseError as exc:
            raise SchemaMetadataError() from exc
        values.update(
            literal.this for literal in expression.find_all(exp.Literal) if literal.is_string
        )
    return values


def constraint_key(constraints: list[SchemaConstraint]) -> str:
    """Definitions come from PostgreSQL 16 with a fixed search path on both sides."""
    return "\n".join(c.model_dump_json() for c in sorted(constraints, key=lambda c: c.name))


def compare_catalog(
    tables: Sequence[SemanticTable | TableSchema],
    live: BusinessSchemaResponse,
    app_revision: str,
) -> SchemaDriftReport:
    """Report every missing/stale column and every differing physical property."""
    report = SchemaDriftReport(app_revision=app_revision, business_revision=live.revision)
    expected = {t.table_name: t for t in tables}
    actual = {t.table_name: t for t in live.tables}
    for name in sorted(set(expected) | set(actual) | set(BUSINESS_TABLES)):
        if name not in expected or name not in actual:
            report.differences.append(
                SchemaDrift(
                    kind=DriftKind.MISSING_METADATA
                    if name not in expected
                    else DriftKind.STALE_METADATA,
                    table_name=name,
                )
            )
        ec = {c.column_name: c for c in expected[name].columns} if name in expected else {}
        ac = {c.column_name: c for c in actual[name].columns} if name in actual else {}
        for column in sorted(set(ec) | set(ac)):
            if column not in ec or column not in ac:
                report.differences.append(
                    SchemaDrift(
                        kind=DriftKind.MISSING_METADATA
                        if column not in ec
                        else DriftKind.STALE_METADATA,
                        table_name=name,
                        column_name=column,
                    )
                )
                continue
            report.differences.extend(compare_column(name, ec[column], ac[column]))
    return report


def compare_column(
    table: str, expected: SemanticColumn | ColumnSchema, actual: PhysicalColumn
) -> list[SchemaDrift]:
    """Compare typed properties; error prose never controls behavior."""
    pairs = (
        (DriftKind.TYPE_MISMATCH, expected.sql_type, actual.sql_type),
        (DriftKind.COLUMN_ORDER, str(expected.ordinal_position), str(actual.ordinal_position)),
        (DriftKind.NULLABILITY, str(expected.nullable), str(actual.nullable)),
        (DriftKind.PRIMARY_KEY, str(expected.is_primary_key), str(actual.is_primary_key)),
        (
            DriftKind.FOREIGN_KEY,
            str((expected.fk_table, expected.fk_column)),
            str((actual.fk_table, actual.fk_column)),
        ),
        (
            DriftKind.CONSTRAINTS,
            constraint_key(expected.constraints),
            constraint_key(actual.constraints),
        ),
    )
    differences = [
        SchemaDrift(
            kind=kind,
            table_name=table,
            column_name=expected.column_name,
            expected=left,
            actual=right,
        )
        for kind, left, right in pairs
        if left != right
    ]
    if (
        not expected.is_pii
        and expected.semantic_type is SemanticType.ENUM
        and set(expected.allowed_values) != enum_keys(expected)
    ):
        differences.append(
            SchemaDrift(
                kind=DriftKind.ENUM_VALUES, table_name=table, column_name=expected.column_name
            )
        )
    return differences
