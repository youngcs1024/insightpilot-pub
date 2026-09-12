"""Compatibility exports for the shared, credential-free SQL policy."""

from app.core.sql_policy.allowlist import (
    ALLOWED_TABLES,
    MAX_QUERY_DEPTH,
    query_depth,
    scoped_tables_reason,
    table_reason,
    validate_sources,
)

__all__ = [
    "ALLOWED_TABLES",
    "MAX_QUERY_DEPTH",
    "query_depth",
    "scoped_tables_reason",
    "table_reason",
    "validate_sources",
]
