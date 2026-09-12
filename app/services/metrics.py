"""Application-owned reads of immutable metric versions with bounded retries."""

import time

import structlog

from app.core.config_models import DatabaseSettings
from app.core.deadline import Deadline
from app.core.retry import run_operation
from app.db.session import Database
from app.repositories.metric import MetricRepository
from app.schemas.metrics import MetricDefinition
from app.services.metric_templates import (
    render_catalog_block,
    render_expression,
    validate_definitions,
    validate_grain,
)

__all__ = ["MetricService", "render_catalog_block", "render_expression", "validate_grain"]

logger = structlog.get_logger(__name__)


class MetricService:
    """Own short read transactions, not a persistent session or an in-memory catalog."""

    render_catalog_block = staticmethod(render_catalog_block)
    render_expression = staticmethod(render_expression)
    validate_grain = staticmethod(validate_grain)

    def __init__(self, database: Database, settings: DatabaseSettings) -> None:
        self._database = database
        self._timeout_s = settings.command_timeout_s

    async def _read(
        self, key: str | None, version: int | None, deadline: Deadline | None
    ) -> list[MetricDefinition]:
        async def read() -> list[MetricDefinition]:
            async with self._database.session() as session, session.begin():
                repository = MetricRepository(session)
                if key is not None:
                    return [await repository.get(key, version)]
                return await repository.list_active()

        result = await run_operation(
            read,
            deadline=deadline or Deadline(time.monotonic() + self._timeout_s),
            timeout_s=self._timeout_s,
            name="metric_catalog_read",
        )
        logger.info("metric_catalog_read", definitions=len(result))
        return result

    async def get_active(self, key: str, *, deadline: Deadline | None = None) -> MetricDefinition:
        """Fetch the active version; never silently choose the highest version."""
        return (await self._read(key, None, deadline))[0]

    async def get_version(
        self, key: str, version: int, *, deadline: Deadline | None = None
    ) -> MetricDefinition:
        """Read an exact version even after it has been superseded."""
        return (await self._read(key, version, deadline))[0]

    async def list_active(self, *, deadline: Deadline | None = None) -> list[MetricDefinition]:
        """Load the current published catalog from PostgreSQL."""
        return await self._read(None, None, deadline)

    async def validate_startup(self) -> None:
        """Fail closed before serving requests when publication is incomplete or invalid."""
        definitions = await self.list_active()
        validate_definitions(definitions)
        logger.info("metric_catalog_validated", definitions=len(definitions))
