"""Shared real-database authentication fixtures, with synthetic signing credentials."""

from collections.abc import AsyncIterator
from time import monotonic

import httpx
import pytest

from alembic import command
from app.application import create_app
from app.core.config_models import DatabaseSettings, Settings
from app.core.deadline import Deadline
from app.db.session import Database
from scripts.migrate_all import migration_config
from scripts.migration_settings import MigrationSettings, MigrationTarget
from tests.database_support import DatabaseStack


@pytest.fixture
async def auth_database(
    settings: Settings, migrated: MigrationSettings, migration_stack: DatabaseStack
) -> AsyncIterator[Database]:
    resource = Database(
        DatabaseSettings(
            port=migration_stack.settings.db_host_port,
            app_password=migration_stack.settings.bootstrap.app_password,
        )
    )
    resource.start()
    try:
        yield resource
    finally:
        await resource.aclose()


@pytest.fixture
async def unauthenticated_client(
    settings: Settings, auth_database: Database
) -> AsyncIterator[httpx.AsyncClient]:
    application = create_app(settings, database=auth_database)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
def migrated(migration_stack: DatabaseStack, monkeypatch: pytest.MonkeyPatch) -> MigrationSettings:
    """Exercise the actual Alembic environments with operator-only configuration."""
    monkeypatch.setenv("IP_ENVIRONMENT", "test")
    monkeypatch.setenv("IP_MIGRATION__HOST", "127.0.0.1")
    monkeypatch.setenv("IP_MIGRATION__PORT", str(migration_stack.settings.db_host_port))
    monkeypatch.setenv("IP_MIGRATION__USER", "postgres")
    monkeypatch.setenv(
        "IP_MIGRATION__PASSWORD",
        migration_stack.settings.postgres_superuser_password.get_secret_value(),
    )
    settings = MigrationSettings(_env_file=None)
    for target in MigrationTarget:
        command.upgrade(migration_config(target), "head")
    return settings


def deadline() -> Deadline:
    return Deadline(monotonic() + 10)
