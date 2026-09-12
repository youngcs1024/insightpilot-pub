"""Reusable testcontainers PostgreSQL, independent of destructive migration acceptance."""

from __future__ import annotations

import json
import secrets
import shutil
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn
from uuid import uuid4

import pytest
from pydantic import BaseModel, Field, SecretStr
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, async_sessionmaker

from alembic import command
from app.core.config_models import DatabaseSettings
from app.db.session import Database
from scripts.migrate_all import migration_config
from scripts.migration_settings import MigrationSettings, MigrationTarget
from tests.ci_settings import InfrastructureSettings
from tests.database_support import foreign_snapshot

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from testcontainers.core.container import DockerContainer

ROOT = Path(__file__).resolve().parents[1]


def unavailable_docker(reason: str) -> NoReturn:
    """A missing CI prerequisite is a failure; local absence leaves acceptance pending."""
    if InfrastructureSettings().require_docker:
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason + "; real-container acceptance remains pending")


def require_docker() -> str:
    """Only unavailable Docker is skippable; subsequent provisioning failures are errors."""
    docker = shutil.which("docker")
    if docker is None:
        unavailable_docker("Docker CLI unavailable")
    try:
        result = subprocess.run(  # noqa: S603 -- fixed, read-only Docker readiness probe.
            [docker, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        unavailable_docker("Docker daemon unavailable")
    if result.returncode:
        unavailable_docker("Docker daemon unavailable")
    return docker


class TestPostgres(BaseModel):
    """Test-only credentials; only the app role is passed into an application."""

    __test__ = False
    port: int = Field(ge=1, le=65535)
    password: SecretStr
    app_password: SecretStr
    volume: str | None = None

    @property
    def app(self) -> DatabaseSettings:
        """Return only the application process's allowed credential."""
        return DatabaseSettings(host="127.0.0.1", port=self.port, app_password=self.app_password)


def stop_container(container: DockerContainer, evidence: Path) -> None:
    """Collect diagnostics, always stop our container, and never delete its volume."""
    try:
        if container._container is not None:
            (evidence / "postgresql.log").write_bytes(b"\n".join(container.get_logs()))
    finally:
        container.stop(delete_volume=False)


@pytest.fixture(scope="session")
def pg_container(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestPostgres]:
    """Provision bootstrap roles; preserve the unique named volume at teardown."""
    external = InfrastructureSettings().postgres
    if external is not None:
        yield TestPostgres(
            port=external.port, password=external.password, app_password=external.app_password
        )
        return
    docker = require_docker()
    # Lazy imports keep pure test collection independent of container initialization.
    from testcontainers.core.config import testcontainers_config  # noqa: PLC0415 -- lazy Docker setup.
    from testcontainers.core.container import DockerContainer  # noqa: PLC0415 -- lazy Docker setup.
    from testcontainers.core.wait_strategies import HealthcheckWaitStrategy  # noqa: PLC0415

    before = foreign_snapshot(docker)
    identity = f"insightpilot-test-{uuid4().hex[:12]}"
    password, app_password = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    container = DockerContainer(
        "pgvector/pgvector:0.8.5-pg16",
        docker_client_kw={"timeout": 30},
        labels={"com.docker.compose.project": identity},
        mem_limit="1536m",
        healthcheck={
            "test": ["CMD-SHELL", "pg_isready -h 127.0.0.1 -U postgres"],
            "interval": 1_000_000_000,
            "timeout": 3_000_000_000,
            "retries": 60,
        },
    )
    volume_name = f"{identity}_pgdata"
    sdk = container.get_docker_client().client
    sdk.volumes.create(name=volume_name, labels={"com.docker.compose.project": identity})
    container.with_envs(
        POSTGRES_PASSWORD=password,
        IP_BOOTSTRAP_APP_PASSWORD=app_password,
        IP_BOOTSTRAP_ETL_PASSWORD=secrets.token_urlsafe(32),
        IP_BOOTSTRAP_MCP_PASSWORD=secrets.token_urlsafe(32),
    )
    container.with_volume_mapping(
        str(ROOT / "docker/postgres/init"), "/docker-entrypoint-initdb.d", "ro"
    )
    container.with_volume_mapping(volume_name, "/var/lib/postgresql/data", "rw")
    # Docker assigns an available port atomically, restricted to loopback.
    container.ports["5432/tcp"] = ("127.0.0.1", None)
    evidence = tmp_path_factory.mktemp("shared-postgres")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(testcontainers_config, "ryuk_disabled", True)
        try:
            container.start()
            HealthcheckWaitStrategy().with_startup_timeout(90).wait_until_ready(container)
            yield TestPostgres(
                port=int(container.get_exposed_port(5432)),
                password=password,
                app_password=app_password,
                volume=volume_name,
            )
        finally:
            # No context-manager default stop(): its delete_volume default is True.
            stop_container(container, evidence)
            retained = subprocess.run(  # noqa: S603 -- inspect only the fixture-owned volume.
                [docker, "volume", "inspect", volume_name, "--format", "{{.Name}}"],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            ).stdout.strip()
            after = foreign_snapshot(docker)
            assert retained == volume_name
            assert after == before, "Pathfinder resources changed"
            (evidence / "isolation.json").write_text(
                json.dumps(
                    {
                        "volume_retained": retained,
                        "before": before,
                        "after": after,
                        "shutdown": "testcontainers stop(delete_volume=False); Ryuk disabled",
                    },
                    indent=2,
                )
                + "\n"
            )


@pytest.fixture(scope="session")
def migrated_db(pg_container: TestPostgres) -> TestPostgres:
    """Apply both actual Alembic histories once with isolated operator configuration."""
    with pytest.MonkeyPatch.context() as patch:
        settings = MigrationSettings(
            _env_file=None,
            migration={
                "host": "127.0.0.1",
                "port": pg_container.port,
                "user": "postgres",
                "password": pg_container.password,
            },
        )
        patch.setattr(MigrationSettings, "load", classmethod(lambda cls: settings))
        for target in MigrationTarget:
            command.upgrade(migration_config(target), "head")
    return pg_container


@asynccontextmanager
async def rollback_connection(settings: DatabaseSettings) -> AsyncIterator[AsyncConnection]:
    """Own an application connection and always roll back the outer transaction."""
    database = Database(settings)
    database.start()
    try:
        async with database.engine.connect() as connection:
            transaction = await connection.begin()
            try:
                yield connection
            finally:
                await transaction.rollback()
    finally:
        await database.aclose()


@pytest.fixture
async def db_connection(migrated_db: TestPostgres) -> AsyncIterator[AsyncConnection]:
    """Function-local engine avoids sharing asyncpg connections across event loops."""
    async with rollback_connection(migrated_db.app) as connection:
        yield connection


@pytest.fixture
async def db_session(db_connection: AsyncConnection) -> AsyncIterator[AsyncSession]:
    """Service commits release savepoints, never the test's outer transaction."""
    async with AsyncSession(
        bind=db_connection,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    ) as session:
        yield session


class TransactionDatabase(Database):
    """Sequential API requests get fresh sessions joined to the fixture transaction."""

    def __init__(self, settings: DatabaseSettings, connection: AsyncConnection) -> None:
        super().__init__(settings)
        # Reuse Database.session() error translation; only replace its session factory.
        self._sessions = async_sessionmaker(
            bind=connection,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        )
