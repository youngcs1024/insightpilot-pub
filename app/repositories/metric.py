"""Read global reference data without retaining ORM rows across boundaries."""

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import MetricCatalogError, MetricNotFound
from app.db.models.metric_definition import MetricDefinitionRecord
from app.schemas.metrics import MetricDefinition


def definition(row: MetricDefinitionRecord) -> MetricDefinition:
    """Validate persisted JSON as rigorously as authored definitions."""
    try:
        return MetricDefinition.model_validate(
            {
                name: getattr(row, name)
                for name in MetricDefinition.model_fields
                if name != "schema_version"
            }
        )
    except ValidationError as exc:
        raise MetricCatalogError() from exc


class MetricRepository:
    """Catalog rows have no user owner; this repository deliberately has no writes."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, key: str, version: int | None = None) -> MetricDefinition:
        """Select an exact historical version or the unique active version."""
        query = select(MetricDefinitionRecord).where(MetricDefinitionRecord.key == key)
        query = (
            query.where(MetricDefinitionRecord.is_active.is_(True))
            if version is None
            else query.where(MetricDefinitionRecord.version == version)
        )
        rows = (await self._session.scalars(query)).all()
        if not rows:
            raise MetricNotFound(key=key, version=version)
        if len(rows) != 1:
            raise MetricCatalogError()
        return definition(rows[0])

    async def list_active(self) -> list[MetricDefinition]:
        """Return stable key order, with duplicates visible to the startup validator."""
        rows = (
            await self._session.scalars(
                select(MetricDefinitionRecord)
                .where(MetricDefinitionRecord.is_active.is_(True))
                .order_by(MetricDefinitionRecord.key)
            )
        ).all()
        return [definition(row) for row in rows]
