"""Offline schema authoring and rendering shared by catalog and agent tests."""

from pathlib import Path

from app.schemas.schema_catalog import BUSINESS_TABLES
from app.services.schema_validation import render_catalog
from data.seed.schema_metadata_loader import load_catalog
from tests.factories import physical

AUTHORING = Path(__file__).resolve().parents[1] / "data/seed/schema_metadata.yaml"


def full_block() -> str:
    catalog = load_catalog(AUTHORING)
    return render_catalog(catalog.tables, physical(catalog), BUSINESS_TABLES)
