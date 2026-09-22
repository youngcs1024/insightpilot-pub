"""Typed object factories. Persistence is always the caller's explicit responsibility."""

from pathlib import Path
from uuid import UUID, uuid4

from mcp.types import CallToolResult

from app.core.schema_artifact import build_artifact
from app.schemas.schema_tools import SchemaResponse
from mcp_server.tools.schema_rendering import render_catalog
from app.db.models import Conversation, Turn, TurnRole, TurnStatus, User
from app.schemas.mcp import ColumnSpec, QueryResultPayload, SqlValue
from app.schemas.schema_catalog import BusinessSchemaResponse, PhysicalTable, SchemaCatalog


def user(*, email: str | None = None, hashed_password: str | None = None) -> User:
    """Build an independent unsaved user; authentication tests supply a real hash."""
    return User(
        id=uuid4(),
        email=email or f"{uuid4().hex}@example.com",
        hashed_password=hashed_password or "not-a-login-hash",
        display_name="Test user",
        is_active=True,
    )


def conversation(user_id: UUID, *, title: str = "Test conversation") -> Conversation:
    """Build an unsaved conversation belonging to the explicit user."""
    return Conversation(id=uuid4(), user_id=user_id, title=title)


def turn(
    conversation_id: UUID,
    *,
    seq: int = 1,
    role: TurnRole = TurnRole.USER,
    status: TurnStatus = TurnStatus.SUCCEEDED,
    content: str = "Test question",
) -> Turn:
    """Build an unsaved turn with explicit ownership through its conversation."""
    return Turn(
        id=uuid4(),
        conversation_id=conversation_id,
        seq=seq,
        role=role,
        status=status,
        content=content,
    )


def query_result(rows: list[list[int | str | None]] | None = None) -> QueryResultPayload:
    """Build the Phase 1 single-column result, preserving an explicitly empty result."""
    values = [[42]] if rows is None else rows
    return QueryResultPayload(
        executed_sql="SELECT 42 LIMIT 1001",
        limit_applied=True,
        execution_ms=1,
        mcp_call_id=str(uuid4()),
        columns=[ColumnSpec(name="n", type="int4")],
        rows=values,
        row_count=len(values),
        result_truncated=False,
    )


def physical(catalog: SchemaCatalog) -> BusinessSchemaResponse:
    """Project physical metadata from an explicit semantic catalog."""
    return BusinessSchemaResponse(
        revision="business-v1",
        tables=[
            PhysicalTable(
                table_name=table.table_name,
                columns=[
                    column.model_dump(
                        include={
                            "column_name",
                            "sql_type",
                            "ordinal_position",
                            "nullable",
                            "is_primary_key",
                            "fk_table",
                            "fk_column",
                            "constraints",
                        }
                    )
                    for column in table.columns
                ],
            )
            for table in catalog.tables
        ],
    )


def business_schema() -> SchemaResponse:
    """Full server response sourced from the frozen migration, without external I/O."""
    snapshot = Path(__file__).resolve().parents[1] / "alembic/app/data/0006_schema_metadata.json"
    artifact = build_artifact(SchemaCatalog.model_validate_json(snapshot.read_text()).tables)
    return SchemaResponse(
        metadata_revision=artifact.metadata_revision,
        business_revision="business-v1",
        tables=artifact.tables,
        notes=list(dict.fromkeys(note for table in artifact.tables for note in table.notes)),
        rendered=render_catalog(artifact.tables),
    )


def sanity_payload(
    rows: list[list[SqlValue]],
    *,
    column_type: str = "numeric",
    truncated: bool = False,
    columns: list[ColumnSpec] | None = None,
) -> QueryResultPayload:
    return QueryResultPayload(
        executed_sql="SELECT amount FROM example",
        limit_applied=True,
        execution_ms=1,
        mcp_call_id="sanity-test",
        columns=columns if columns is not None else [ColumnSpec(name="amount", type=column_type)],
        rows=rows,
        row_count=len(rows),
        result_truncated=truncated,
    )


def mcp_success() -> CallToolResult:
    return CallToolResult(
        content=[],
        structured_content={
            "schema_version": 2,
            "executed_sql": "SELECT 1",
            "limit_applied": False,
            "execution_ms": 1,
            "mcp_call_id": "test-call",
            "columns": [{"name": "n", "type": "int4"}],
            "rows": [[1]],
            "row_count": 1,
            "result_truncated": False,
        },
    )
