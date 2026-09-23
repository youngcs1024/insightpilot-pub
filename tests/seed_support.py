"""Module-owned seed databases with disjoint small subnets and retained volumes."""

import asyncio
import secrets
import socket
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from alembic import command
from data.seed.contracts import Parameters
from data.seed.files import export
from data.seed.generation import generate
from scripts.deployment import (
    ROOT,
    BootstrapSecrets,
    Command,
    DeploymentSettings,
    Invocation,
    compose_prefix,
    process_environment,
    run_command,
)
from scripts.migrate_all import migration_config
from scripts.migration_settings import MigrationSettings, MigrationTarget
from scripts.seed import import_seed
from scripts.seed_settings import SeedDatabaseSettings, SeedSettings
from tests.database_support import DatabaseStack, foreign_snapshot, isolated_subnets
from tests.shared_database import require_docker


@pytest.fixture(scope="module")
def seed_database_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DatabaseStack]:
    """Keep only this module's PostgreSQL alive; never consume default /16 subnets."""
    docker = require_docker()
    before = foreign_snapshot(docker)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    settings = DeploymentSettings(
        _env_file=None,
        compose_project_name=f"insightpilot-test-{uuid4().hex[:12]}",
        db_host_port=port,
        postgres_superuser_password=secrets.token_urlsafe(32),
        bootstrap=BootstrapSecrets(
            **{
                f"{role}_password": secrets.token_urlsafe(32)
                for role in ("app", "etl", "mcp", "audit")
            }
        ),
    )
    call = Invocation(
        dev=True, command=Command.UP, deployment_file=ROOT / ".env.deployment.example"
    )
    evidence = tmp_path_factory.mktemp("seed-database")
    override = evidence / "networks.yml"
    override.write_text(
        yaml.safe_dump(
            {
                "networks": {
                    name: {"ipam": {"config": [{"subnet": subnet}]}}
                    for name, subnet in zip(
                        ("backend", "egress"), isolated_subnets(docker), strict=True
                    )
                }
            }
        )
    )
    prefix = [*compose_prefix(docker, settings, call), "-f", str(override)]
    env = process_environment(settings)
    started = False
    try:
        run_command([*prefix, "up", "-d", "--wait", "postgres"], env, timeout=120)
        started = True
        yield DatabaseStack(docker=docker, settings=settings, call=call)
    finally:
        run_command([*prefix, "down"], env, timeout=120)
        assert foreign_snapshot(docker) == before
        if started:
            retained = run_command(
                [
                    docker,
                    "volume",
                    "inspect",
                    f"{settings.compose_project_name}_pgdata",
                    "--format",
                    "{{.Name}}",
                ],
                env,
            ).stdout.strip()
            (evidence / "volume-retained.txt").write_text(retained + "\n")


@pytest.fixture(scope="module")
def seed_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("seed-baseline") / "export"
    export(generate(Parameters()), directory)
    return directory


@pytest.fixture(scope="module")
def seeded(seed_stack: DatabaseStack, seed_directory: Path) -> SeedSettings:
    stack = seed_stack.settings
    operator = MigrationSettings(
        _env_file=None,
        migration={
            "host": "127.0.0.1",
            "port": stack.db_host_port,
            "password": stack.postgres_superuser_password,
        },
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(MigrationSettings, "load", classmethod(lambda cls: operator))
        command.upgrade(migration_config(MigrationTarget.BUSINESS), "head")
    settings = SeedSettings(
        _env_file=None,
        seed=SeedDatabaseSettings(port=stack.db_host_port, password=stack.bootstrap.etl_password),
    )
    result = asyncio.run(import_seed(seed_directory, settings))
    assert result.outcome == "imported"
    return settings
