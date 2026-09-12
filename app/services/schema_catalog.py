"""Async application-side semantics merged with MCP-only physical metadata."""

import asyncio
import time
from collections.abc import Callable
from typing import Protocol

import structlog
from pydantic import Field
from sqlalchemy import text

from app.core.config_models import SchemaCatalogSettings
from app.core.deadline import Deadline
from app.core.errors import DeadlineExceededError, McpResultError, SchemaDriftError, ValidationError
from app.db.session import Database
from app.repositories.schema_metadata import SchemaMetadataRepository
from app.schemas.mcp import Contract
from app.schemas.schema_catalog import (
    BUSINESS_TABLES,
    BusinessSchemaArguments,
    BusinessSchemaResponse,
    SchemaCatalog,
    SchemaDriftReport,
    SemanticTable,
)
from app.services.schema_validation import compare_catalog, render_catalog

logger = structlog.get_logger(__name__)


class SchemaClient(Protocol):
    """Only the typed MCP method is available to the catalog service."""

    async def get_business_schema(
        self, args: BusinessSchemaArguments, *, deadline: Deadline
    ) -> BusinessSchemaResponse:
        """Fetch structure or validate a known revision."""
        ...


class CatalogCache(Contract):
    """Disposable copy of PostgreSQL-backed data, never its source of truth."""

    app_revision: str
    live: BusinessSchemaResponse
    tables: list[SemanticTable]
    expires_at: float
    rendered: dict[tuple[str, ...], str] = Field(default_factory=dict)


class SchemaCatalogService:
    """One application instance owns its cache; sessions are never retained."""

    def __init__(
        self,
        database: Database,
        client: SchemaClient,
        settings: SchemaCatalogSettings,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._database = database
        self._client = client
        self._settings = settings
        self._clock = clock
        self._lock = asyncio.Lock()
        self._cache: CatalogCache | None = None

    def clear(self) -> None:
        """Drop disposable cache state at lifecycle shutdown."""
        self._cache = None

    async def snapshot(self, *, deadline: Deadline | None = None) -> SchemaCatalog:
        """Expose detached typed metadata only after the existing live drift check."""
        cache, report = await self._load(force=False, deadline=deadline)
        if not report.valid:
            raise SchemaDriftError(differences=len(report.differences))
        return SchemaCatalog(tables=cache.tables).model_copy(deep=True)

    async def render(
        self, tables: list[str] | None = None, *, deadline: Deadline | None = None
    ) -> str:
        """Render a valid catalog; unknown or empty selections fail explicitly."""
        if tables is not None and (not tables or set(tables) - set(BUSINESS_TABLES)):
            raise ValidationError("Select one or more known business tables.")
        selected = tuple(t for t in BUSINESS_TABLES if tables is None or t in tables)
        cache, report = await self._load(force=False, deadline=deadline)
        if not report.valid:
            raise SchemaDriftError(differences=len(report.differences))
        if selected not in cache.rendered:
            cache.rendered[selected] = render_catalog(cache.tables, cache.live, selected)
        return cache.rendered[selected]

    async def validate_against_live_schema(
        self, *, deadline: Deadline | None = None
    ) -> SchemaDriftReport:
        """Always bypass both revision hints and TTL for an authoritative CI check."""
        _, report = await self._load(force=True, deadline=deadline)
        return report

    async def _load(
        self, *, force: bool, deadline: Deadline | None
    ) -> tuple[CatalogCache, SchemaDriftReport]:
        budget = deadline or Deadline(time.monotonic() + self._settings.operation_timeout_s)
        try:
            async with (
                asyncio.timeout(budget.budget(self._settings.operation_timeout_s)),
                self._lock,
            ):
                return await self._refresh(force=force, deadline=budget)
        except TimeoutError as exc:
            self.clear()
            logger.exception("schema_catalog_timeout")
            raise DeadlineExceededError() from exc
        except BaseException:
            self.clear()
            raise

    async def _refresh(
        self, *, force: bool, deadline: Deadline
    ) -> tuple[CatalogCache, SchemaDriftReport]:
        started = self._clock()
        cache = self._cache
        expired = cache is None or started >= cache.expires_at
        async with self._database.session() as session, session.begin():
            await session.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"))
            repository = SchemaMetadataRepository(session)
            revision = await repository.revision()
            reload = force or expired or cache is None or cache.app_revision != revision
            tables = await repository.read() if reload or cache is None else cache.tables
        live = await self._client.get_business_schema(
            BusinessSchemaArguments(
                known_revision=None if reload or cache is None else cache.live.revision
            ),
            deadline=deadline,
        )
        if live.unchanged:
            if reload or cache is None or live.revision != cache.live.revision:
                raise McpResultError()
            live = cache.live
        else:
            reload = True
        report = compare_catalog(tables, live, revision)
        if reload or cache is None:
            cache = CatalogCache(
                app_revision=revision,
                live=live,
                tables=tables,
                expires_at=self._clock() + self._settings.ttl_s,
            )
        self._cache = cache if report.valid else None
        logger.info(
            "schema_catalog_checked",
            app_revision=revision,
            business_revision=live.revision,
            cache_hit=not reload,
            differences=len(report.differences),
            duration_ms=int((self._clock() - started) * 1000),
        )
        return cache, report
