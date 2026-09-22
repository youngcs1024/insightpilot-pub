"""Pure deterministic schema projections shared by the exporter and MCP process."""

import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from app.core.errors import SchemaMetadataError
from app.schemas.schema_catalog import (
    BUSINESS_TABLES,
    BusinessSchemaResponse,
    PhysicalColumn,
    PhysicalTable,
    SemanticColumn,
    SemanticTable,
)
from app.schemas.schema_tools import (
    ColumnSchema,
    SchemaArtifact,
    SchemaResponse,
    TableSchema,
    metadata_digest,
)


def project_tables(
    tables: Sequence[SemanticTable | TableSchema], *, include_samples: bool = True
) -> list[TableSchema]:
    """Remove sensitive value fields before serialization, hashing or rendering."""
    projected: list[TableSchema] = []
    by_name = {table.table_name: table for table in tables}
    for name in BUSINESS_TABLES:
        if name not in by_name:
            continue
        table = by_name[name]
        columns = []
        source_columns: Sequence[SemanticColumn | ColumnSchema] = table.columns
        for column in sorted(source_columns, key=lambda c: c.ordinal_position):
            values = column.model_dump()
            values["allowed_values"] = {} if column.is_pii else column.allowed_values
            values["sample_values"] = (
                [value[:50] for value in column.sample_values[:3]]
                if include_samples and not column.is_pii
                else []
            )
            columns.append(ColumnSchema.model_validate(values))
        projected.append(
            TableSchema(
                table_name=name, description=table.description, notes=table.notes, columns=columns
            )
        )
    return projected


def build_artifact(tables: list[SemanticTable]) -> SchemaArtifact:
    """Export application-owned metadata as a sanitized, content-addressed resource."""
    if len(tables) != len(BUSINESS_TABLES) or {t.table_name for t in tables} != set(
        BUSINESS_TABLES
    ):
        raise SchemaMetadataError()
    safe = project_tables(tables)
    return SchemaArtifact(metadata_revision=metadata_digest(safe), tables=safe)


def artifact_json(artifact: SchemaArtifact) -> str:
    """The checked-in artifact and CI export use identical deterministic bytes."""
    return (
        json.dumps(artifact.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n"
    )


def load_artifact(path: Path) -> SchemaArtifact:
    """Load only a validated build resource; callers perform this outside async I/O."""
    try:
        return SchemaArtifact.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as exc:
        raise SchemaMetadataError() from exc


def physical_schema(response: SchemaResponse) -> BusinessSchemaResponse:
    """Recover physical structure for app-side authoring validation, without another tool."""
    return BusinessSchemaResponse(
        revision=response.business_revision,
        tables=[
            PhysicalTable(
                table_name=table.table_name,
                columns=[
                    PhysicalColumn.model_validate(
                        column.model_dump(include=set(PhysicalColumn.model_fields))
                    )
                    for column in table.columns
                ],
            )
            for table in response.tables
        ],
    )
