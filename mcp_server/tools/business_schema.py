"""Read only allowlisted structure; never read business data or accept caller SQL."""

import asyncio

import psycopg
from psycopg.rows import dict_row
from pydantic import ValidationError

from app.core.errors import McpResultError, SqlExecutionError, SqlTimeoutError
from app.schemas.schema_catalog import BUSINESS_TABLES, BusinessSchemaResponse, PhysicalTable
from mcp_server.db import BusinessDatabase

# information_schema is the column authority. pg_catalog supplies type modifiers and
# constraints (information_schema key views hide constraints from SELECT-only roles).
SCHEMA_SQL = """
SELECT 'biz.' || c.table_name AS table_name,
       jsonb_agg(jsonb_build_object(
         'column_name', c.column_name,
         'sql_type', pg_catalog.format_type(a.atttypid, a.atttypmod),
         'ordinal_position', c.ordinal_position,
         'nullable', c.is_nullable = 'YES',
         'is_primary_key', EXISTS (
           SELECT 1 FROM pg_catalog.pg_constraint k
           WHERE k.conrelid = r.oid AND k.contype = 'p' AND a.attnum = ANY(k.conkey)),
         'fk_table', (SELECT nr.nspname || '.' || rr.relname
           FROM pg_catalog.pg_constraint k
           JOIN pg_catalog.pg_class rr ON rr.oid = k.confrelid
           JOIN pg_catalog.pg_namespace nr ON nr.oid = rr.relnamespace
           WHERE k.conrelid = r.oid AND k.contype = 'f' AND a.attnum = ANY(k.conkey)),
         'fk_column', (SELECT ar.attname FROM pg_catalog.pg_constraint k
           JOIN pg_catalog.pg_attribute ar ON ar.attrelid = k.confrelid
             AND ar.attnum = k.confkey[array_position(k.conkey, a.attnum)]
           WHERE k.conrelid = r.oid AND k.contype = 'f' AND a.attnum = ANY(k.conkey)),
         'constraints', COALESCE((SELECT jsonb_agg(jsonb_build_object(
           'name', k.conname, 'kind', k.contype,
           'definition', pg_catalog.pg_get_constraintdef(k.oid, false),
           'columns', (SELECT jsonb_agg(ak.attname ORDER BY ord)
              FROM unnest(k.conkey) WITH ORDINALITY AS keys(num, ord)
              JOIN pg_catalog.pg_attribute ak ON ak.attrelid = r.oid AND ak.attnum = num)
           ) ORDER BY k.conname)
           FROM pg_catalog.pg_constraint k WHERE k.conrelid = r.oid
             AND k.contype IN ('p', 'f', 'u', 'c') AND a.attnum = ANY(k.conkey)), '[]'::jsonb)
       ) ORDER BY c.ordinal_position) AS columns
FROM information_schema.columns c
JOIN pg_catalog.pg_namespace n ON n.nspname = c.table_schema
JOIN pg_catalog.pg_class r ON r.relnamespace = n.oid AND r.relname = c.table_name
JOIN pg_catalog.pg_attribute a ON a.attrelid = r.oid AND a.attname = c.column_name
WHERE c.table_schema = 'biz' AND c.table_name = ANY(%s) AND r.relkind = 'r'
GROUP BY c.table_name ORDER BY c.table_name
"""


class BusinessSchemaReader:
    """Own one bounded read-only structure transaction, with no retries here."""

    def __init__(self, database: BusinessDatabase) -> None:
        self.database = database

    async def read(self, known_revision: str | None = None) -> BusinessSchemaResponse:
        """Check the migration revision before optionally reading physical metadata."""
        try:
            async with asyncio.timeout(self.database.settings.operation_timeout_s):
                return await self._read(known_revision)
        except (psycopg.errors.QueryCanceled, TimeoutError) as exc:
            raise SqlTimeoutError() from exc
        except psycopg.Error as exc:
            raise SqlExecutionError() from exc
        except ValidationError as exc:
            raise McpResultError() from exc

    async def _read(self, known_revision: str | None) -> BusinessSchemaResponse:
        async with self.database.connection() as conn, conn.transaction():
            await conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            await conn.execute("SET LOCAL statement_timeout = '10s'")
            await conn.execute("SET LOCAL search_path = pg_catalog")
            cursor = await conn.execute("SELECT version_num FROM biz.alembic_version_biz")
            revisions = await cursor.fetchall()
            if len(revisions) != 1:
                raise McpResultError()
            revision = str(revisions[0][0])
            if revision == known_revision:
                return BusinessSchemaResponse(revision=revision, unchanged=True)
            async with conn.cursor(row_factory=dict_row) as rows:
                await rows.execute(SCHEMA_SQL, ([t.removeprefix("biz.") for t in BUSINESS_TABLES],))
                tables = [PhysicalTable.model_validate(row) for row in await rows.fetchall()]
            return BusinessSchemaResponse(revision=revision, tables=tables)
