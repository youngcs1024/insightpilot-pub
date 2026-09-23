"""Real PostgreSQL in a unique Compose project; teardown always retains its volume."""

import ipaddress
import json
import os
import secrets
import socket
from collections.abc import Iterator
from uuid import uuid4

import pytest
from pydantic import BaseModel

from scripts.deployment import (
    ROOT,
    BootstrapSecrets,
    Command,
    DeploymentSettings,
    Invocation,
    execute,
    process_environment,
    run_command,
)


class DatabaseStack(BaseModel):
    """Only this fixture knows the administrative test credentials."""

    docker: str
    settings: DeploymentSettings
    call: Invocation


def foreign_snapshot(docker: str) -> tuple[str, str, str]:
    """Capture foreign identities/state without touching their lifecycle."""
    env = dict(os.environ)
    containers = run_command(
        [
            docker,
            "ps",
            "-a",
            "--filter",
            "name=pathfinder",
            "--format",
            "{{.ID}} {{.Names}} {{.State}}",
        ],
        env,
    ).stdout
    volumes = run_command([docker, "volume", "ls", "--format", "{{.Name}}"], env).stdout
    networks = run_command([docker, "network", "ls", "--format", "{{.ID}} {{.Name}}"], env).stdout
    return tuple(
        "\n".join(sorted(line for line in value.splitlines() if "pathfinder" in line))
        for value in (containers, volumes, networks)
    )


@pytest.fixture(scope="session")
def database_stack(tmp_path_factory: pytest.TempPathFactory) -> Iterator[DatabaseStack]:
    """Boot real SQL on an isolated volume; missing Docker fails acceptance."""
    from tests.shared_database import require_docker  # noqa: PLC0415 -- avoid circular fixture imports.

    docker = require_docker()
    before = foreign_snapshot(docker)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    settings = DeploymentSettings(
        _env_file=None,
        compose_project_name=f"insightpilot-test-{uuid4().hex[:12]}",
        db_host_port=port,
        postgres_superuser_password=secrets.token_urlsafe(32) + "@:/+'$ space",
        bootstrap=BootstrapSecrets(
            app_password=secrets.token_urlsafe(32) + "@:/+'$\\\" space",
            etl_password=secrets.token_urlsafe(32) + "@:/+'$\\\"",
            mcp_password=secrets.token_urlsafe(32) + "@:/+'$\\\"",
            audit_password=secrets.token_urlsafe(32) + "@:/+'$\\\"",
        ),
    )
    call = Invocation(
        dev=True,
        deployment_file=ROOT / ".env.deployment.example",
        command=Command.UP,
        arguments=["-d", "--wait", "postgres"],
    )
    stack = DatabaseStack(docker=docker, settings=settings, call=call)
    evidence = tmp_path_factory.mktemp("database-acceptance")
    try:
        execute(docker, settings, call)
        yield stack
    finally:
        try:
            logs = execute(
                docker,
                settings,
                call.model_copy(
                    update={"command": Command.LOGS, "arguments": ["--no-color", "postgres"]}
                ),
            )
            (evidence / "postgresql.log").write_text(logs)
        finally:
            execute(
                docker, settings, call.model_copy(update={"command": Command.DOWN, "arguments": []})
            )
        remaining = run_command(
            [docker, "volume", "ls", "--format", "{{.Name}}"], process_environment(settings)
        ).stdout.splitlines()
        assert f"{settings.compose_project_name}_pgdata" in remaining
        after = foreign_snapshot(docker)
        (evidence / "isolation.json").write_text(
            json.dumps(
                {
                    "project": settings.compose_project_name,
                    "port": port,
                    "before": before,
                    "after": after,
                    "volume_retained": f"{settings.compose_project_name}_pgdata" in remaining,
                    "shutdown": "compose down without volume deletion",
                },
                indent=2,
            )
            + "\n"
        )
        assert after == before, "Pathfinder resources changed during test lifecycle"


def isolated_subnets(docker: str) -> list[str]:
    """Allocate small test-only subnets without consuming two default /16 pools.

    Retained historical projects and session-scoped DB fixtures nearly exhaust
    Docker Desktop's default pools. Explicit disjoint /28s preserve those projects.
    Docker itself rejects a concurrent overlapping allocation; never retry it.
    """
    env = dict(os.environ)
    ids = run_command([docker, "network", "ls", "-q"], env).stdout.split()
    networks = json.loads(run_command([docker, "network", "inspect", *ids], env).stdout)
    occupied = [
        ipaddress.ip_network(config["Subnet"])
        for network in networks
        for config in (network["IPAM"]["Config"] or [])
        if config.get("Subnet") and ":" not in config["Subnet"]
    ]
    selected: list[str] = []
    for candidate in ipaddress.ip_network("172.16.0.0/12").subnets(new_prefix=28):
        if not any(candidate.overlaps(existing) for existing in occupied):
            selected.append(str(candidate))
        if len(selected) == 2:  # noqa: PLR2004 -- backend and egress.
            return selected
    pytest.fail("No disjoint private subnets available for isolated Docker acceptance")
