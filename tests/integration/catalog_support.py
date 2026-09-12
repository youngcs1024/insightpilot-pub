"""Catalog migration, transaction adapter and MCP client shared by integration tests."""

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from alembic import command
from app.clients.mcp_client import McpClient
from app.core.config_models import DatabaseSettings, MCPSettings, Settings
from app.db.session import Database
from scripts.migrate_all import migration_config
from scripts.migration_settings import MigrationSettings, MigrationTarget
from tests.database_support import DatabaseStack


@pytest.fixture(scope="module")
def catalog_migrated(database_stack: DatabaseStack) -> None:
    settings = MigrationSettings(
        _env_file=None,
        migration={
            "host": "127.0.0.1",
            "port": database_stack.settings.db_host_port,
            "password": database_stack.settings.postgres_superuser_password,
        },
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(MigrationSettings, "load", classmethod(lambda cls: settings))
        for target in MigrationTarget:
            command.upgrade(migration_config(target), "head")


class TransactionDatabase(Database):
    """Every service session joins a rollback-only test transaction via savepoints."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self.sessions = sessions

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        async with self.sessions() as session:
            yield session


@pytest.fixture
async def client(catalog_migrated: None, server_endpoint: MCPSettings) -> AsyncIterator[McpClient]:
    client = McpClient(server_endpoint)
    try:
        yield client
    finally:
        await client.aclose()


def cli_environment(
    settings: Settings, stack: DatabaseStack, endpoint: MCPSettings
) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("IP_", "PYTHON"))}
    configured = settings.model_copy(
        update={
            "database": DatabaseSettings(
                port=stack.settings.db_host_port, app_password=stack.settings.bootstrap.app_password
            ),
            "mcp": endpoint,
        }
    )
    # Each process field is explicitly validated by the same API Settings schema.
    for name, value in configured.model_dump(mode="json").items():
        if name != "environment" and value is not None:
            env["IP_" + name.upper()] = (
                json.dumps(value) if isinstance(value, (dict, list)) else str(value).lower()
            )
    env["IP_ENVIRONMENT"] = "test"
    env["IP_DATABASE__APP_PASSWORD"] = stack.settings.bootstrap.app_password.get_secret_value()
    env["IP_MCP__AUTH_TOKEN"] = endpoint.auth_token.get_secret_value()
    env["IP_LLM__API_KEY"] = settings.llm.api_key.get_secret_value()
    env["IP_SECURITY__JWT_SECRET"] = settings.security.jwt_secret.get_secret_value()
    return env
