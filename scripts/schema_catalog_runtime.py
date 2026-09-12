"""Credential-scoped CLI composition shared by rendering and the CI drift gate."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from app.clients.mcp_client import McpClient
from app.core.config_models import Settings
from app.core.logging import setup_logging
from app.db.session import Database
from app.services.schema_catalog import SchemaCatalogService


@asynccontextmanager
async def catalog_service() -> AsyncIterator[SchemaCatalogService]:
    """Load only API-role credentials; all business structure travels through MCP."""
    settings = Settings.load()
    settings.observability.log_level = "WARNING"
    setup_logging(settings)
    database = Database(settings.database)
    client = McpClient(settings.mcp)
    database.start()
    try:
        yield SchemaCatalogService(database, client, settings.schema_catalog)
    finally:
        try:
            await client.aclose()
        finally:
            await database.aclose()
