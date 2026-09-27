"""Startup diagnosis complements, but never replaces, database concurrency enforcement."""

import asyncio

import sqlglot
import structlog
from sqlglot import exp
from sqlglot.errors import SqlglotError

from app.core.errors import MemorySchemaError
from app.db.session import Database
from app.repositories.memory_schema import memory_schema_status

_EXPECTED = (
    "CASE WHEN is_active AND memory_type = 'metric_override' "
    "THEN content ->> 'metric_key' ELSE NULL END"
)


def canonical_expression(value: str) -> exp.Expr:
    """Ignore PostgreSQL's redundant text casts and parenthesized catalog rendering."""
    expression = sqlglot.parse_one(value, read="postgres")
    # PostgreSQL may wrap a stored generated expression in its column's assignment cast.
    if isinstance(expression, exp.Cast) and expression.to == exp.DataType.build("VARCHAR(64)"):
        expression = expression.this.unnest()
    for node in reversed(list(expression.walk())):
        if isinstance(node, exp.Paren) or (
            isinstance(node, exp.Cast) and node.to.is_type(exp.DataType.Type.TEXT)
        ):
            node.replace(node.this)
    # Reparse after stripping casts so JSON string paths receive the same AST shape.
    return sqlglot.parse_one(expression.sql(dialect="postgres"), read="postgres")


async def validate_memory_schema(database: Database, *, timeout_s: float) -> None:
    """Refuse readiness for legacy schema drift; never repair or scan user data here."""
    async with database.session() as session, asyncio.timeout(timeout_s):
        status = await memory_schema_status(session)
    try:
        valid = status.valid and canonical_expression(status.expression) == canonical_expression(
            _EXPECTED
        )
    except SqlglotError:
        valid = False
    if not valid:
        raise MemorySchemaError()
    structlog.get_logger(__name__).info("memory_schema_validated")
