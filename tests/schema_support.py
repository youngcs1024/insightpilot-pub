"""Offline schema authoring and rendering shared by catalog and agent tests."""

from pathlib import Path

from app.schemas.schema_catalog import BUSINESS_TABLES
from app.core.schema_artifact import project_tables
from app.schemas.schema_catalog import SemanticTable, BusinessSchemaResponse
from mcp_server.tools.schema_rendering import render_catalog as render_safe
from data.seed.schema_metadata_loader import load_catalog
from tests.factories import physical

AUTHORING = Path(__file__).resolve().parents[1] / "data/seed/schema_metadata.yaml"


def full_block() -> str:
    catalog = load_catalog(AUTHORING)
    return render_catalog(catalog.tables, physical(catalog), BUSINESS_TABLES)


def render_catalog(
    tables: list[SemanticTable], live: BusinessSchemaResponse, selected: tuple[str, ...]
) -> str:
    """Exercise the server renderer with explicit projected authoring input."""
    return render_safe(project_tables([t for t in tables if t.table_name in selected]))
