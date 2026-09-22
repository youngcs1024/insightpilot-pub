"""Authoring and pure comparison regressions independent of PostgreSQL and LLMs."""

# ruff: noqa: PLR2004 -- fixed 52-column acceptance contract.

import json
from pathlib import Path

import pytest
import yaml

from app.core.errors import SchemaMetadataError
from app.schemas.schema_catalog import DriftKind, SchemaCatalog
from app.services.schema_validation import compare_catalog
from data.seed.schema_metadata_loader import load_catalog
from scripts import schema_tokens
from scripts.render_schema_catalog import measure
from tests.factories import physical
from tests.schema_support import AUTHORING, render_catalog

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "alembic/app/data/0006_schema_metadata.json"


def test_yaml_snapshot_and_all_fields_are_bound() -> None:
    catalog = load_catalog(AUTHORING)
    assert catalog == SchemaCatalog.model_validate_json(SNAPSHOT.read_text())
    assert sum(len(t.columns) for t in catalog.tables) == 52
    assert compare_catalog(catalog.tables, physical(catalog), "app-v1").valid
    for table in catalog.tables:
        for column in table.columns:
            assert column.description
            if column.fk_table:
                block = render_catalog(catalog.tables, physical(catalog), (table.table_name,))
                assert f"[FK → {column.fk_table}.{column.fk_column}]" in block


@pytest.mark.parametrize(
    "mutation",
    ["unknown", "duplicate", "target", "enum", "boolean", "missing_table", "duplicate_column"],
)
def test_invalid_authoring_rejected(tmp_path: Path, mutation: str) -> None:
    data = yaml.safe_load(AUTHORING.read_text())
    orders = next(t for t in data["tables"] if t["table_name"] == "biz.orders")
    if mutation == "unknown":
        orders["columns"][0]["ignored_field"] = "must not disappear"
    elif mutation == "target":
        orders["columns"][1]["fk_column"] = "absent"
    elif mutation == "enum":
        orders["columns"][5]["allowed_values"].pop("cancelled")
    elif mutation == "boolean":
        orders["columns"][0]["nullable"] = "false"
    elif mutation == "missing_table":
        data["tables"] = data["tables"][:-1]
    elif mutation == "duplicate_column":
        orders["columns"].append(orders["columns"][0])
    source = yaml.safe_dump(data, allow_unicode=True)
    if mutation == "duplicate":
        source += "schema_version: 1\n"
    target = tmp_path / "invalid.yaml"
    target.write_text(source)
    with pytest.raises(SchemaMetadataError):
        load_catalog(target)


@pytest.mark.parametrize(
    ("field", "value", "kind"),
    [
        ("sql_type", "numeric(14,2)", DriftKind.TYPE_MISMATCH),
        ("nullable", True, DriftKind.NULLABILITY),
        ("is_primary_key", False, DriftKind.PRIMARY_KEY),
        ("fk_table", "biz.products", DriftKind.FOREIGN_KEY),
        ("constraints", [], DriftKind.CONSTRAINTS),
    ],
)
def test_physical_drift_is_typed(field: str, value: object, kind: DriftKind) -> None:
    catalog = load_catalog(AUTHORING)
    live = physical(catalog)
    live.tables[0].columns[0] = live.tables[0].columns[0].model_copy(update={field: value})
    report = compare_catalog(catalog.tables, live, "app-v1")
    assert kind in {d.kind for d in report.differences}


def test_pii_and_samples_are_bounded() -> None:
    catalog = load_catalog(AUTHORING)
    customers = next(t for t in catalog.tables if t.table_name == "biz.customers")
    phone = next(c for c in customers.columns if c.column_name == "phone")
    phone.sample_values = ["PII_SENTINEL"]
    customers.columns[0].sample_values = ["x" * 70, "two", "three", "four"]
    block = render_catalog(catalog.tables, physical(catalog), (customers.table_name,))
    assert "PII_SENTINEL" not in block
    assert "x" * 51 not in block
    assert "two | three" in block
    assert "four" not in block


def test_render_preserves_load_bearing_semantics_and_measures() -> None:
    catalog = load_catalog(AUTHORING)
    block = render_catalog(
        catalog.tables, physical(catalog), tuple(t.table_name for t in catalog.tables)
    )
    for marker in (
        "Asia/Shanghai",
        "cancelled=已取消",
        "COALESCE",
        "requested_at",
        "paid_at",
        "is_test_account",
        "region_id=3",
        "phone",
        "promo_id=17",
    ):
        assert marker in block
    tokens, upper = measure(block)
    assert 0 < tokens < upper
    assert upper == len(block.encode())
    assert "schema_version" not in block


def test_snapshot_is_json_not_runtime_yaml_dependency() -> None:
    assert json.loads(SNAPSHOT.read_text())["schema_version"] == 1


def test_offline_tokenizer_rejects_corrupt_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = tmp_path / "corrupt.gz"
    bad.write_bytes(b"invalid-compressed-vocabulary")
    schema_tokens.encoding.cache_clear()
    monkeypatch.setattr(schema_tokens, "VOCABULARY", bad)
    try:
        with pytest.raises(SchemaMetadataError):
            schema_tokens.encoding()
    finally:
        schema_tokens.encoding.cache_clear()
