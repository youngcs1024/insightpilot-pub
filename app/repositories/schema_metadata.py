"""Read global reference metadata in the caller's application transaction."""

from pydantic import ValidationError
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import SchemaMetadataError
from app.db.models.schema_metadata import SchemaMetadataRecord, SchemaTableRecord
from app.schemas.schema_catalog import SemanticColumn, SemanticTable


class SchemaMetadataRepository:
    """No user identity is required: this catalog contains no user-owned rows."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def revision(self) -> str:
        """Read the real installed application revision on every lookup."""
        revisions = (
            (await self._session.execute(text("SELECT version_num FROM alembic_version_app")))
            .scalars()
            .all()
        )
        if len(revisions) != 1:
            raise SchemaMetadataError()
        return str(revisions[0])

    async def read(self) -> list[SemanticTable]:
        """Convert persistence shapes into typed models before crossing the boundary."""
        tables = (await self._session.scalars(select(SchemaTableRecord))).all()
        columns = (await self._session.scalars(select(SchemaMetadataRecord))).all()
        try:
            grouped: dict[str, list[SemanticColumn]] = {table.table_name: [] for table in tables}
            for row in columns:
                values = {name: getattr(row, name) for name in SemanticColumn.model_fields}
                grouped[row.table_name].append(SemanticColumn.model_validate(values))
            return [
                SemanticTable(
                    table_name=table.table_name,
                    description=table.description,
                    notes=table.notes,
                    columns=sorted(grouped[table.table_name], key=lambda c: c.ordinal_position),
                )
                for table in tables
            ]
        except (ValidationError, KeyError) as exc:
            raise SchemaMetadataError() from exc
