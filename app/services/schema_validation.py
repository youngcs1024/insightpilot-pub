"""Compatibility imports for credential-free schema comparison; rendering lives in MCP."""

from app.core.schema_validation import compare_catalog, compare_column, constraint_key, enum_keys

__all__ = ["compare_catalog", "compare_column", "constraint_key", "enum_keys"]
