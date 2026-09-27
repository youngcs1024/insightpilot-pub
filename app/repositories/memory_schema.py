"""Read only PostgreSQL schema metadata, never unscoped user memory rows."""

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.mcp import Contract


class MemorySchemaStatus(Contract):
    """The generated key and deferrable uniqueness must agree with the model."""

    valid: bool
    expression: str


async def memory_schema_status(session: AsyncSession) -> MemorySchemaStatus:
    """Inspect the exact named constraint, key order and generated column metadata."""
    row = (
        (
            await session.execute(
                text("""
        SELECT c.contype = 'u' AND c.condeferrable AND c.condeferred
               AND c.convalidated AND i.indisvalid AND i.indisunique
               AND a.attgenerated = 's'
               AND a.atttypid = 'varchar'::regtype AND a.atttypmod = 68
               AND ARRAY(SELECT k.attname::text
                         FROM unnest(c.conkey) WITH ORDINALITY AS x(attnum, ord)
                         JOIN pg_attribute k ON k.attrelid = c.conrelid
                                             AND k.attnum = x.attnum
                         ORDER BY x.ord)
                   = ARRAY['user_id', 'memory_type', 'active_metric_key'] AS valid,
               pg_get_expr(d.adbin, d.adrelid) AS expression
        FROM pg_constraint c
        JOIN pg_index i ON i.indexrelid = c.conindid
        JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attname = 'active_metric_key'
        JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum
        WHERE c.conrelid = to_regclass('public.memories')
          AND c.conname = 'uq_memories_active_metric'
    """)
            )
        )
        .mappings()
        .one_or_none()
    )
    return (
        MemorySchemaStatus.model_validate(dict(row))
        if row
        else MemorySchemaStatus(valid=False, expression="")
    )
