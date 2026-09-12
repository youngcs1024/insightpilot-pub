"""Validate authoring YAML; the API never imports this loader."""

import json
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError

from app.core.errors import MetricCatalogError, SchemaMetadataError
from app.schemas.metrics import MetricCatalog
from app.services.metric_templates import validate_definitions
from data.seed.schema_metadata_loader import check_keys


def load_catalog(path: Path) -> MetricCatalog:
    """Reject duplicate keys, unknown fields and invalid templates before freezing."""
    try:
        source = path.read_text(encoding="utf-8")
        node = yaml.compose(source)
        if node is None:
            raise MetricCatalogError("Empty metric catalog.")
        check_keys(node)
        catalog = MetricCatalog.model_validate_json(json.dumps(yaml.safe_load(source)), strict=True)
        validate_definitions(catalog.definitions)
        return catalog
    except (OSError, yaml.YAMLError, ValidationError, TypeError, SchemaMetadataError) as exc:
        raise MetricCatalogError() from exc
