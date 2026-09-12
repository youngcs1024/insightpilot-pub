"""Strict authoring input; runtime reads PostgreSQL, not this YAML file."""

import json
from pathlib import Path

import yaml  # type: ignore[import-untyped]  # PyYAML ships no typing marker.
from pydantic import ValidationError
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode  # type: ignore[import-untyped]

from app.core.errors import SchemaMetadataError
from app.schemas.schema_catalog import BUSINESS_TABLES, SchemaCatalog, SemanticType
from app.services.schema_validation import enum_keys


def check_keys(node: Node, ancestors: frozenset[int] = frozenset()) -> None:
    """Reject duplicate YAML keys and recursive aliases before safe_load can lose data."""
    if id(node) in ancestors:
        raise SchemaMetadataError("Recursive YAML aliases are not supported.")
    ancestors = ancestors | {id(node)}
    if isinstance(node, MappingNode):
        keys: set[str] = set()
        for key, value in node.value:
            if not isinstance(key, ScalarNode) or key.value in keys:
                raise SchemaMetadataError("Duplicate or non-scalar YAML key.")
            keys.add(key.value)
            check_keys(value, ancestors)
    elif isinstance(node, SequenceNode):
        for item in node.value:
            check_keys(item, ancestors)


def load_catalog(path: Path) -> SchemaCatalog:
    """Load complete semantics with exact types, no coercion or ignored fields."""
    try:
        source = path.read_text(encoding="utf-8")
        node = yaml.compose(source)
        if node is None:
            raise SchemaMetadataError("Empty catalog.")
        check_keys(node)
        catalog = SchemaCatalog.model_validate_json(json.dumps(yaml.safe_load(source)), strict=True)
        validate_catalog(catalog)
        return catalog
    except (OSError, yaml.YAMLError, ValidationError, TypeError) as exc:
        raise SchemaMetadataError() from exc


def validate_catalog(catalog: SchemaCatalog) -> None:
    """Require complete table, column and enum declarations before authoring a migration."""
    if {t.table_name for t in catalog.tables} != set(BUSINESS_TABLES):
        raise SchemaMetadataError("Catalog must cover exactly eight business tables.")
    for table in catalog.tables:
        if not table.columns:
            raise SchemaMetadataError("Table metadata must have columns.")
        for column in table.columns:
            if column.semantic_type is SemanticType.ENUM and set(
                column.allowed_values
            ) != enum_keys(column):
                raise SchemaMetadataError("Enum meanings do not cover the declared values.")
