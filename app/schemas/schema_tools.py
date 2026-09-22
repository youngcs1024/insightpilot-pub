"""PII-safe schema tool and deployable metadata contracts, independent of storage."""

import hashlib
import json
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from app.schemas.mcp import Contract
from app.schemas.schema_catalog import (
    BUSINESS_TABLES,
    PhysicalColumn,
    SchemaDriftReport,
    SemanticType,
    TableName,
)

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
RequestedTable = Annotated[str, Field(min_length=1, max_length=127)]
Sample = Annotated[str, Field(max_length=50)]


class GetSchemaArgs(Contract):
    """Select metadata, optionally bypassing the disposable physical-schema cache."""

    tables: list[RequestedTable] | None = Field(default=None, min_length=1, max_length=64)
    include_samples: bool = False
    refresh: bool = False


class ColumnSchema(PhysicalColumn):
    """Safe output projection; authored enum completeness is validated before export."""

    description: str = Field(min_length=1, max_length=2000)
    semantic_type: SemanticType
    allowed_values: dict[str, str] = Field(default_factory=dict)
    is_pii: bool = False
    sample_values: list[Sample] = Field(default_factory=list, max_length=3)
    notes: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def pii_values_absent(self) -> Self:
        """An unsafe server or artifact is rejected, never silently trusted."""
        if self.is_pii and (self.sample_values or self.allowed_values):
            raise PydanticCustomError("pii_metadata", "PII values are forbidden")
        return self


class TableSchema(Contract):
    """A selected, sanitized table with all required physical and semantic properties."""

    table_name: TableName
    description: str = Field(min_length=1, max_length=2000)
    notes: list[str] = Field(default_factory=list, max_length=20)
    columns: list[ColumnSchema] = Field(min_length=1, max_length=1600)

    @model_validator(mode="after")
    def unique_columns(self) -> Self:
        """Do not allow duplicate identifiers or positions to disappear into mappings."""
        names = [column.column_name for column in self.columns]
        positions = [column.ordinal_position for column in self.columns]
        if len(names) != len(set(names)) or len(positions) != len(set(positions)):
            raise PydanticCustomError("schema_columns", "Duplicate columns or positions")
        return self


def metadata_digest(tables: list[TableSchema]) -> str:
    """Hash canonical metadata, independent of source row order or application migrations."""
    normalized = [
        table.model_copy(
            update={"columns": sorted(table.columns, key=lambda c: c.ordinal_position)}
        ).model_dump(mode="json")
        for table in sorted(tables, key=lambda t: t.table_name)
    ]
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


class SchemaArtifact(Contract):
    """Versioned build resource; schema_version describes this contract, not PostgreSQL."""

    schema_version: Literal[1] = 1
    metadata_revision: Digest
    tables: list[TableSchema] = Field(min_length=8, max_length=8)

    @model_validator(mode="after")
    def complete_and_verified(self) -> Self:
        """Validate the full allowlist, FK targets and content identity before serving."""
        if {table.table_name for table in self.tables} != set(BUSINESS_TABLES):
            raise PydanticCustomError("artifact_tables", "Artifact must cover the business schema")
        targets = {(t.table_name, c.column_name) for t in self.tables for c in t.columns}
        for table in self.tables:
            for column in table.columns:
                if column.fk_table and (column.fk_table, column.fk_column) not in targets:
                    raise PydanticCustomError("artifact_fk", "Artifact FK target is absent")
        if metadata_digest(self.tables) != self.metadata_revision:
            raise PydanticCustomError("artifact_digest", "Artifact content digest does not match")
        return self


class SchemaResponse(Contract):
    """Prompt-ready metadata, with explicit partial rejections and revision identities."""

    schema_version: Literal[1] = 1
    metadata_revision: Digest
    business_revision: str = Field(min_length=1, max_length=64)
    tables: list[TableSchema] = Field(max_length=8)
    notes: list[str]
    rendered: str
    rejected: list[RequestedTable] = Field(default_factory=list, max_length=64)

    @model_validator(mode="after")
    def allowed_and_unique(self) -> Self:
        """Reject out-of-boundary results before any node can use their text."""
        names = [table.table_name for table in self.tables]
        if len(names) != len(set(names)) or set(names) - set(BUSINESS_TABLES):
            raise PydanticCustomError("schema_tables", "Invalid response table set")
        return self


class SchemaToolError(Contract):
    """Schema-specific errors supplement the unchanged SQL tool error protocol."""

    schema_version: Literal[1] = 1
    code: Literal["SCHEMA_DRIFT", "SCHEMA_METADATA_INVALID"]
    message: str
    report: SchemaDriftReport | None = None
