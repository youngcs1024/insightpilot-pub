"""Serve a versioned semantic artifact after bounded live physical-schema validation."""

import asyncio
import time
from collections.abc import Callable
from pathlib import Path

import structlog

from app.core.errors import SchemaDriftError, SchemaMetadataError, SqlTimeoutError
from app.core.schema_artifact import load_artifact
from app.core.schema_validation import compare_catalog
from app.schemas.schema_catalog import BusinessSchemaResponse
from app.schemas.schema_tools import GetSchemaArgs, SchemaArtifact, SchemaResponse
from mcp_server.policy.allowlist import ALLOWED_TABLES
from mcp_server.policy.column_policy import suppress_values
from mcp_server.tools.business_schema import BusinessSchemaReader
from mcp_server.tools.schema_rendering import render_catalog

logger = structlog.get_logger(__name__)


class SchemaTool:
    """Cache verified physical structure only; the baked artifact remains authoritative."""

    def __init__(
        self,
        reader: BusinessSchemaReader,
        artifact_path: Path,
        *,
        ttl_s: float = 300,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._reader = reader
        self._clock = clock
        self._ttl_s = ttl_s
        self._lock = asyncio.Lock()
        self._live: BusinessSchemaResponse | None = None
        self._expires_at = 0.0
        self._artifact: SchemaArtifact | None = None
        # Construction is synchronous, before the async server lifespan starts.
        try:
            self._artifact = load_artifact(artifact_path)
        except SchemaMetadataError:
            logger.exception("schema_artifact_invalid")

    async def read(self, args: GetSchemaArgs) -> SchemaResponse:
        """Bound lock acquisition and database work within one typed timeout."""
        try:
            async with asyncio.timeout(self._reader.database.settings.operation_timeout_s):
                return await self._read(args)
        except TimeoutError as exc:
            self._live = None
            raise SqlTimeoutError() from exc

    async def _read(self, args: GetSchemaArgs) -> SchemaResponse:
        """Intersect selections server-side and return only validated, sanitized metadata."""
        async with asyncio.timeout(self._reader.database.settings.operation_timeout_s):
            async with self._lock:
                try:
                    live = await self._validated_live(refresh=args.refresh)
                except BaseException:
                    self._live = None
                    raise
        artifact = self._artifact
        if artifact is None:
            raise SchemaMetadataError()
        requested = None if args.tables is None else set(args.tables)
        selected = [
            table
            for table in artifact.tables
            if table.table_name in ALLOWED_TABLES
            and (requested is None or table.table_name in requested)
        ]
        rejected = list(dict.fromkeys(t for t in args.tables or [] if t not in ALLOWED_TABLES))
        if rejected:
            # Caller-controlled identifiers may themselves contain secrets. Log counts only.
            logger.warning("schema_tables_rejected", rejected_count=len(rejected))
        tables = suppress_values(selected, include_samples=args.include_samples)
        notes = list(dict.fromkeys(note for table in tables for note in table.notes))
        logger.info(
            "schema_read",
            metadata_revision=artifact.metadata_revision,
            business_revision=live.revision,
            table_count=len(tables),
            include_samples=args.include_samples,
        )
        return SchemaResponse(
            metadata_revision=artifact.metadata_revision,
            business_revision=live.revision,
            tables=tables,
            notes=notes,
            rendered=render_catalog(tables),
            rejected=rejected,
        )

    async def _validated_live(self, *, refresh: bool) -> BusinessSchemaResponse:
        artifact = self._artifact
        if artifact is None:
            raise SchemaMetadataError()
        cached = self._live
        force = refresh or cached is None or self._clock() >= self._expires_at
        live = await self._reader.read(None if force or cached is None else cached.revision)
        if live.unchanged:
            if cached is None or force or live.revision != cached.revision:
                raise SchemaMetadataError()
            return cached
        report = compare_catalog(artifact.tables, live, artifact.metadata_revision)
        if not report.valid:
            # Expected/actual constraint literals are not suitable error or log payloads.
            safe_report = report.model_copy(
                update={
                    "differences": [
                        item.model_copy(update={"expected": None, "actual": None})
                        for item in report.differences
                    ]
                }
            )
            raise SchemaDriftError(report=safe_report)
        self._live = live
        self._expires_at = self._clock() + self._ttl_s
        return live
