"""Server-owned enforcement applied again to every selected schema response."""

from app.core.schema_artifact import project_tables
from app.schemas.schema_tools import TableSchema


def suppress_values(tables: list[TableSchema], *, include_samples: bool) -> list[TableSchema]:
    """Defensively copy and sanitize before both structured and text output."""
    return project_tables(tables, include_samples=include_samples)
