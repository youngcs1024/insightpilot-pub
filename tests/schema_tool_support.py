"""Scripted physical reads and explicit artifact fixtures for boundary tests."""

import asyncio
from pathlib import Path
from unittest.mock import Mock

from app.core.schema_artifact import artifact_json, build_artifact
from app.schemas.schema_catalog import BusinessSchemaResponse, SchemaCatalog
from mcp_server.db import BusinessDatabase
from mcp_server.tools.business_schema import BusinessSchemaReader
from tests.factories import physical

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "mcp_server/data/schema_metadata.json"


def authored_catalog() -> SchemaCatalog:
    return SchemaCatalog.model_validate_json(
        (ROOT / "alembic/app/data/0006_schema_metadata.json").read_bytes()
    )


def write_artifact(path: Path, catalog: SchemaCatalog) -> None:
    path.write_text(artifact_json(build_artifact(catalog.tables)), encoding="utf-8")


class ScriptedSchemaReader(BusinessSchemaReader):
    """Honor revision hints while retaining every physical-read request."""

    def __init__(self, catalog: SchemaCatalog | None = None) -> None:
        super().__init__(Mock(spec=BusinessDatabase, settings=Mock(operation_timeout_s=1.0)))
        self.source = physical(catalog or authored_catalog())
        self.calls: list[str | None] = []
        self.error: Exception | None = None
        self.delay = 0.0

    async def read(self, known_revision: str | None = None) -> BusinessSchemaResponse:
        self.calls.append(known_revision)
        await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        if known_revision == self.source.revision:
            return BusinessSchemaResponse(revision=self.source.revision, unchanged=True)
        return self.source.model_copy(deep=True)
