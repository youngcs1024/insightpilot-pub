"""Artifact provenance, determinism and privacy without a database connection."""

# ruff: noqa: PLR2004 -- fixed schema and artifact bounds.

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.errors import SchemaMetadataError
from app.core.schema_artifact import artifact_json, build_artifact, load_artifact
from app.schemas.schema_catalog import SemanticType
from app.schemas.schema_tools import GetSchemaArgs, SchemaArtifact, metadata_digest
from tests.schema_tool_support import ARTIFACT, ROOT, authored_catalog


def test_artifact_matches_frozen_migration_and_is_order_independent() -> None:
    catalog = authored_catalog()
    artifact = build_artifact(catalog.tables)
    assert artifact_json(artifact) == ARTIFACT.read_text()
    catalog.tables.reverse()
    for table in catalog.tables:
        table.columns.reverse()
    assert artifact_json(build_artifact(catalog.tables)) == ARTIFACT.read_text()
    assert len(artifact.tables) == 8
    assert sum(len(t.columns) for t in artifact.tables) == 52


def test_export_strips_pii_and_bounds_non_pii_samples() -> None:
    catalog = authored_catalog()
    customers = next(t for t in catalog.tables if t.table_name == "biz.customers")
    phone = next(c for c in customers.columns if c.column_name == "phone")
    phone.sample_values = ["PII_SAMPLE"]
    phone.allowed_values = {"PII_ENUM": "private"}
    phone.semantic_type = SemanticType.ENUM
    customers.columns[0].sample_values = ["x" * 90, "two", "three", "four"]
    artifact = build_artifact(catalog.tables)
    source = artifact_json(artifact)
    assert "PII_SAMPLE" not in source
    assert "PII_ENUM" not in source
    safe = next(t for t in artifact.tables if t.table_name == "biz.customers")
    assert safe.columns[0].sample_values == ["x" * 50, "two", "three"]
    assert SchemaArtifact.model_validate_json(source) == artifact


@pytest.mark.parametrize("mutation", ["digest", "version", "table", "duplicate", "column", "pii"])
def test_corrupt_artifact_is_rejected(tmp_path: Path, mutation: str) -> None:
    payload = json.loads(ARTIFACT.read_text())
    if mutation == "digest":
        payload["metadata_revision"] = "0" * 64
    elif mutation == "version":
        payload["schema_version"] = 99
    elif mutation == "table":
        payload["tables"][0]["table_name"] = "biz.private"
    elif mutation == "duplicate":
        payload["tables"][1] = payload["tables"][0]
    elif mutation == "column":
        payload["tables"][0]["columns"].append(payload["tables"][0]["columns"][0])
    else:
        column = next(c for t in payload["tables"] for c in t["columns"] if c["is_pii"])
        column["sample_values"] = ["PII_SENTINEL"]
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(SchemaMetadataError):
        load_artifact(path)


def test_missing_or_non_json_artifact_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "schema.json"
    with pytest.raises(SchemaMetadataError):
        load_artifact(path)
    path.write_text("invalid")
    with pytest.raises(SchemaMetadataError):
        load_artifact(path)


def test_metadata_semantics_change_content_identity() -> None:
    artifact = load_artifact(ARTIFACT)
    tables = [table.model_copy(deep=True) for table in artifact.tables]
    tables[0].notes.append("A changed business convention")
    assert metadata_digest(tables) != artifact.metadata_revision


@pytest.mark.parametrize("tables", [[], [""], ["x" * 128], ["biz.orders"] * 65])
def test_invalid_selection_is_rejected(tables: list[str]) -> None:
    with pytest.raises(ValidationError):
        GetSchemaArgs(tables=tables)


def test_actual_retired_descriptor_retains_nullable_known_revision() -> None:
    fixture = json.loads((ROOT / "tests/fixtures/get_business_schema_descriptor.json").read_text())
    assert fixture["source_sha"] == "8dc7c472ee3285a4d09f4897c6ddd6c926062a2c"
    assert fixture["sdk_version"] == "2.1.1"
    assert fixture["server_name"] == "insightpilot-business"
    assert "server_version" in fixture
    assert fixture["tool"]["name"] == "get_business_schema"
    known = fixture["tool"]["inputSchema"]["properties"]["known_revision"]
    assert {branch["type"] for branch in known["anyOf"]} == {"string", "null"}
    assert known["default"] is None
    assert "outputSchema" in fixture["tool"]
