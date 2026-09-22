"""Versioned physical schema and persisted semantic catalog contracts."""

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from app.schemas.mcp import Contract

BUSINESS_TABLES = (
    "biz.regions",
    "biz.promotions",
    "biz.products",
    "biz.customers",
    "biz.orders",
    "biz.order_items",
    "biz.refunds",
    "biz.inventory",
)
Identifier = Annotated[str, Field(min_length=1, max_length=63, pattern=r"^[a-z_][a-z0-9_]*$")]
TableName = Annotated[str, Field(min_length=1, max_length=127, pattern=r"^biz\.[a-z_][a-z0-9_]*$")]


class SemanticType(StrEnum):
    """Closed semantic vocabulary, independent of the physical PostgreSQL type."""

    IDENTIFIER = "identifier"
    MONEY = "money"
    TIMESTAMP = "timestamp"
    ENUM = "enum"
    QUANTITY = "quantity"
    TEXT = "text"
    FOREIGN_KEY = "foreign_key"


class SchemaConstraint(Contract):
    """PostgreSQL's stable rendered constraint, with ordered participating columns."""

    name: Identifier
    kind: Literal["p", "f", "u", "c"]
    definition: str = Field(min_length=1, max_length=8000)
    columns: list[Identifier] = Field(max_length=64)


class PhysicalColumn(Contract):
    """Live structure read without accessing any business row."""

    column_name: Identifier
    sql_type: str = Field(min_length=1, max_length=200)
    ordinal_position: int = Field(ge=1, le=1600)
    nullable: bool
    is_primary_key: bool
    fk_table: TableName | None = None
    fk_column: Identifier | None = None
    constraints: list[SchemaConstraint] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def paired_foreign_key(self) -> Self:
        """A partially described foreign key is never usable metadata."""
        if (self.fk_table is None) != (self.fk_column is None):
            raise PydanticCustomError("foreign_key_pair", "Both foreign key fields are required")
        return self


class PhysicalTable(Contract):
    """An existing allowlisted table, including a missing-column state for drift checks."""

    table_name: TableName
    columns: list[PhysicalColumn] = Field(max_length=1600)


class BusinessSchemaResponse(Contract):
    """An unchanged response contains no tables; a fresh response contains live structure."""

    schema_version: Literal[1] = 1
    revision: str = Field(min_length=1, max_length=64)
    unchanged: bool = False
    tables: list[PhysicalTable] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def valid_tables(self) -> Self:
        """Reject duplicate, out-of-boundary or inconsistent payloads."""
        names = [table.table_name for table in self.tables]
        if len(names) != len(set(names)) or set(names) - set(BUSINESS_TABLES):
            raise PydanticCustomError("schema_tables", "Invalid business schema table set")
        if self.unchanged and self.tables:
            raise PydanticCustomError("schema_unchanged", "Unchanged response must omit tables")
        for table in self.tables:
            columns = [column.column_name for column in table.columns]
            positions = [column.ordinal_position for column in table.columns]
            if len(columns) != len(set(columns)) or len(positions) != len(set(positions)):
                raise PydanticCustomError("schema_columns", "Duplicate columns or positions")
        return self


class SemanticColumn(PhysicalColumn):
    """Expected physical properties plus authored business semantics."""

    description: str = Field(min_length=1, max_length=2000)
    semantic_type: SemanticType
    allowed_values: dict[str, str] = Field(default_factory=dict)
    is_pii: bool = False
    sample_values: list[str] = Field(default_factory=list, max_length=20)
    notes: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def valid_semantics(self) -> Self:
        """Reject empty enum meanings and impossible semantic declarations."""
        if self.semantic_type is SemanticType.ENUM and not self.allowed_values:
            raise PydanticCustomError("enum_values", "Enum meanings are required")
        if any(not key or not value.strip() for key, value in self.allowed_values.items()):
            raise PydanticCustomError("enum_meaning", "Enum keys and meanings must be nonempty")
        if self.semantic_type is SemanticType.FOREIGN_KEY and self.fk_table is None:
            raise PydanticCustomError("foreign_key_target", "Foreign key target is required")
        return self


class SemanticTable(Contract):
    """Normalized table semantics; physical expectations live on its columns."""

    table_name: TableName
    description: str = Field(min_length=1, max_length=2000)
    notes: list[str] = Field(default_factory=list, max_length=20)
    columns: list[SemanticColumn] = Field(max_length=1600)


class SchemaCatalog(Contract):
    """A complete persisted catalog, shared by all users."""

    schema_version: Literal[1] = 1
    tables: list[SemanticTable] = Field(max_length=8)

    @model_validator(mode="after")
    def unique_metadata(self) -> Self:
        """Prevent silent config/model drift and unresolved foreign key targets."""
        names = [table.table_name for table in self.tables]
        if len(names) != len(set(names)):
            raise PydanticCustomError("duplicate_table", "Duplicate metadata table")
        targets = {(t.table_name, c.column_name) for t in self.tables for c in t.columns}
        for table in self.tables:
            columns = [column.column_name for column in table.columns]
            if len(columns) != len(set(columns)):
                raise PydanticCustomError("duplicate_column", "Duplicate metadata column")
            for column in table.columns:
                if column.fk_table and (column.fk_table, column.fk_column) not in targets:
                    raise PydanticCustomError(
                        "missing_target", "Foreign key target has no metadata"
                    )
        return self


class DriftKind(StrEnum):
    """Structured drift categories, never inferred from explanatory prose."""

    MISSING_METADATA = "missing_metadata"
    STALE_METADATA = "stale_metadata"
    TYPE_MISMATCH = "type_mismatch"
    COLUMN_ORDER = "column_order"
    METADATA_CONTENT = "metadata_content"
    NULLABILITY = "nullability"
    PRIMARY_KEY = "primary_key"
    FOREIGN_KEY = "foreign_key"
    CONSTRAINTS = "constraints"
    ENUM_VALUES = "enum_values"


class SchemaDrift(Contract):
    """Safe difference between an expected catalog and a live schema."""

    kind: DriftKind
    table_name: str
    column_name: str | None = None
    expected: str | None = None
    actual: str | None = None


class SchemaDriftReport(Contract):
    """CLI and service validation use the same typed report."""

    schema_version: Literal[1] = 1
    app_revision: str
    business_revision: str
    differences: list[SchemaDrift] = Field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Any difference prevents use of the catalog."""
        return not self.differences
