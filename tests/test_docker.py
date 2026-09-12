"""Docker contracts and isolated real-image acceptance; volumes are retained."""

import json
import os
import re
import secrets
import socket
import subprocess
from pathlib import Path
from uuid import uuid4

import pytest
import yaml

from scripts.deployment import (
    ROOT,
    BootstrapSecrets,
    Command,
    DeploymentError,
    DeploymentSettings,
    Invocation,
    compose_prefix,
    process_environment,
    run_command,
)
from tests.database_support import foreign_snapshot, isolated_subnets
from tests.shared_database import require_docker

pytestmark = pytest.mark.slow
IMAGE_LIMIT = 500_000_000


@pytest.mark.parametrize(
    ("service", "base"),
    [
        ("etcd", "quay.io/coreos/etcd:v3.5.18"),
        ("minio", "quay.io/minio/minio:RELEASE.2024-05-28T17-19-04Z"),
        ("milvus", "milvusdb/milvus:v2.6.4"),
    ],
)
def test_storage_images_keep_pinned_bases_and_nonroot_users(service: str, base: str) -> None:
    source = (ROOT / f"docker/Dockerfile.{service}").read_text()
    assert source.splitlines()[0] == "FROM " + base
    assert source.rstrip().endswith("USER 10001:10001")
    assert "ARG " not in source
    config = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    assert config["services"][service]["user"] == "10001:10001"
    assert config["services"][service]["build"]["dockerfile"] == f"docker/Dockerfile.{service}"


def test_compose_default_provider_resource_is_public() -> None:
    config = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    value = config["services"]["api"]["environment"]["IP_LLM__CAPABILITIES_PATHS"]
    prefix = "${IP_LLM_CAPABILITIES_PATHS:-"
    assert value.startswith(prefix)
    assert value.endswith("}")
    paths = json.loads(value[len(prefix) : -1])
    assert paths == ["app/resources/provider_capabilities.json"]
    assert (
        DeploymentSettings.model_fields["llm_capabilities_paths"].get_default(
            call_default_factory=True
        )
        == paths
    )
    assert all((ROOT / path).is_file() for path in paths)


def test_dockerignore_excludes_env() -> None:
    patterns = (ROOT / ".dockerignore").read_text().splitlines()
    assert {".env*", "**/.env*", ".git", "volumes", "tests", "models"} <= set(patterns)
    assert "docs" in patterns
    assert {"**/*.pem", "**/*.key", "**/.venv"} <= set(patterns)


@pytest.mark.parametrize("service", ["api", "mcp"])
def test_no_build_args_named_like_secrets(service: str) -> None:
    source = (ROOT / f"docker/Dockerfile.{service}").read_text()
    for name in re.findall(r"(?mi)^ARG\s+(\w+)", source):
        assert not re.search(r"key|secret|token|password|credential", name, re.I)
    builder, runtime = source.split(" AS runtime", 1)
    assert "build-essential" in builder
    assert "build-essential" not in runtime
    assert "USER appuser" in runtime
    assert "uv sync --frozen" in builder
    assert "target=/root/.cache/uv" in builder


@pytest.mark.parametrize(("service", "endpoint"), [("api", "8000/health"), ("mcp", "8001/ready")])
def test_healthcheck_command_uses_python_not_curl(service: str, endpoint: str) -> None:
    source = (ROOT / f"docker/Dockerfile.{service}").read_text()
    health = next(line for line in source.splitlines() if line.startswith("HEALTHCHECK"))
    assert "curl" not in health
    command = json.loads(health.split(" CMD ", 1)[1])
    assert command[:2] == ["python", "-c"]
    compile(command[2], "healthcheck", "exec")
    assert endpoint in command[2]
    assert "timeout=3" in command[2]
    for option in ("--interval=15s", "--timeout=5s", "--start-period=30s", "--retries=3"):
        assert option in health


@pytest.mark.parametrize("profile", ["core", "retrieval", "full", "dev", "ui"])
def test_profile_closure_and_resource_budgets(profile: str) -> None:
    config = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    services = config["services"]
    storage = {"etcd", "minio", "milvus"}
    assert set(services) == {
        "postgres",
        "migrate",
        "mcp",
        "api",
        "seed",
        "model-tunnel",
        "model-diagnostics",
        *storage,
    }
    for name, service in services.items():
        if name in {"seed", "model-diagnostics"}:
            continue
        if name in storage | {"model-tunnel"} and profile == "core":
            assert profile not in service["profiles"]
            continue
        assert profile in service["profiles"]
        assert service["platform"] == "linux/amd64"
        assert "container_name" not in service
        for dependency in service.get("depends_on", {}):
            assert profile in services[dependency]["profiles"]
    for name in ("api", "mcp"):
        assert (
            services[name]["depends_on"]["migrate"]["condition"] == "service_completed_successfully"
        )
        assert "volumes" not in services[name]
    assert services["api"]["depends_on"]["mcp"]["condition"] == "service_healthy"
    assert services["migrate"]["depends_on"]["postgres"]["condition"] == "service_healthy"
    for name, reserve, limit in (
        ("api", "400M", "1024M"),
        ("mcp", "128M", "384M"),
        ("etcd", "256M", "512M"),
        ("minio", "256M", "512M"),
        ("milvus", "2048M", "3072M"),
    ):
        assert services[name]["deploy"]["resources"] == {
            "limits": {"memory": limit},
            "reservations": {"memory": reserve},
        }
    assert "ports" not in services["postgres"]
    assert "ports" not in services["mcp"]
    for name in storage:
        assert "ports" not in services[name]
    for name in ("etcd", "minio"):
        assert services["milvus"]["depends_on"][name]["condition"] == "service_healthy"
    assert services["milvus"]["healthcheck"]["start_period"] == "90s"
    assert config["networks"]["backend"]["internal"] is True


@pytest.fixture(scope="session")
def docker_images(tmp_path_factory: pytest.TempPathFactory) -> str:
    docker = require_docker()
    evidence = tmp_path_factory.mktemp("docker-images")
    env = dict(os.environ, BUILDX_CONFIG=str(evidence / "buildx"))
    for service in ("api", "mcp"):
        result = run_command(
            [
                docker,
                "build",
                "--platform",
                "linux/amd64",
                "-f",
                f"docker/Dockerfile.{service}",
                "-t",
                f"insightpilot-{service}:step114",
                ".",
            ],
            env,
            timeout=900,
        )
        (evidence / f"{service}-build.txt").write_text(result.stdout + result.stderr)
    return docker


@pytest.mark.integration
@pytest.mark.parametrize("service", ["api", "mcp"])
def test_runtime_image(docker_images: str, service: str, tmp_path: Path) -> None:
    docker = docker_images
    env = dict(os.environ)
    image = f"insightpilot-{service}:step114"
    metadata = json.loads(run_command([docker, "image", "inspect", image], env).stdout)[0]
    assert metadata["Size"] < IMAGE_LIMIT
    assert metadata["Architecture"] == "amd64"
    assert metadata["Config"]["User"] == "appuser"
    assert run_command([docker, "run", "--rm", image, "whoami"], env).stdout.strip() == "appuser"
    probe = (
        "import pathlib,shutil; "
        "assert all(shutil.which(n) is None for n in ('cc','gcc','g++','make','uv')); "
        "root=pathlib.Path('/opt/insightpilot'); "
        "bundles={root/'.venv/lib/python3.12/site-packages/certifi/cacert.pem', "
        "root/'.venv/lib/python3.12/site-packages/grpc/_cython/_credentials/roots.pem'}; "
        "assert all(b'BEGIN CERTIFICATE' in p.read_bytes() and b'PRIVATE KEY' not in p.read_bytes() "
        "for p in bundles if p.exists()); "
        "assert not any(p.name.startswith('.env') or p.suffix in ('.pem','.key') "
        "for p in root.rglob('*') "
        "if p not in bundles); "
        "assert not (root/'mcp_server/.venv-step17-backup').exists()"
    )
    run_command([docker, "run", "--rm", image, "python", "-c", probe], env)
    history = run_command([docker, "history", "--no-trunc", image], env).stdout
    assert not re.search(r"(?i)(?:API_KEY|AUTH_TOKEN|PASSWORD|SECRET)\s*=", history)
    (tmp_path / "image.json").write_text(
        json.dumps(
            {
                "image": image,
                "id": metadata["Id"],
                "size": metadata["Size"],
                "architecture": metadata["Architecture"],
                "user": metadata["Config"]["User"],
                "healthcheck": metadata["Config"]["Healthcheck"],
                "compiler_and_secret_file_probe": "passed",
            },
            indent=2,
        )
    )
    (tmp_path / "history.txt").write_text(history)


def test_isolated_subnets_skip_occupied_and_builtin_networks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            "bridge\nhost\nnone\n",
            json.dumps(
                [
                    {"IPAM": {"Config": None}},
                    {"IPAM": {"Config": []}},
                    {"IPAM": {"Config": [{"Subnet": "172.16.0.0/28"}]}},
                ]
            ),
        ]
    )

    def fake_command(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, stdout=next(responses))

    monkeypatch.setattr("tests.database_support.run_command", fake_command)
    assert isolated_subnets("docker") == ["172.16.0.16/28", "172.16.0.32/28"]


@pytest.mark.integration
@pytest.mark.parametrize("migration_fails", [False, True])
def test_core_startup_and_migration_gate(
    docker_images: str,
    tmp_path: Path,
    migration_fails: bool,
) -> None:
    docker = docker_images
    before = foreign_snapshot(docker)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    settings = DeploymentSettings(
        _env_file=None,
        compose_project_name=f"insightpilot-test-{uuid4().hex[:12]}",
        api_host_port=port,
        postgres_superuser_password=secrets.token_urlsafe(32),
        bootstrap=BootstrapSecrets(
            **{f"{role}_password": secrets.token_urlsafe(32) for role in ("app", "etl", "mcp")}
        ),
        mcp_auth_token=secrets.token_urlsafe(32),
        jwt_secret=secrets.token_urlsafe(48),
        llm_api_key=secrets.token_urlsafe(32),
    )
    override = tmp_path / "compose.acceptance.yml"
    services = {
        name: {"image": f"insightpilot-{image}:step114"}
        for name, image in (("api", "api"), ("migrate", "api"), ("mcp", "mcp"))
    }
    if migration_fails:
        services["migrate"]["command"] = ["python", "-c", "raise SystemExit(23)"]
    subnets = isolated_subnets(docker)
    override.write_text(
        yaml.safe_dump(
            {
                "services": services,
                "networks": {
                    name: {"ipam": {"config": [{"subnet": subnet}]}}
                    for name, subnet in zip(("backend", "egress"), subnets, strict=True)
                },
            }
        )
    )
    call = Invocation(
        command=Command.UP, profiles=["core"], deployment_file=ROOT / ".env.deployment.example"
    )
    prefix = [*compose_prefix(docker, settings, call), "-f", str(override)]
    env = process_environment(settings)
    try:
        if migration_fails:
            with pytest.raises(DeploymentError):
                run_command([*prefix, "up", "-d", "--no-build", "--wait"], env, timeout=180)
        else:
            run_command([*prefix, "up", "-d", "--no-build", "--wait"], env, timeout=180)
        output = run_command([*prefix, "ps", "-a", "--format", "json"], env).stdout
        rows = [json.loads(line) for line in output.splitlines() if line.strip()]
        states = {row["Service"]: row for row in rows}
        assert states["migrate"]["ExitCode"] == (23 if migration_fails else 0)
        for name in ("api", "mcp"):
            if migration_fails:
                assert name not in states or states[name]["State"] != "running"
            else:
                assert states[name]["Health"] == "healthy"
        if not migration_fails:
            run_command(
                [
                    *prefix,
                    "exec",
                    "-T",
                    "api",
                    "python",
                    "-c",
                    "import urllib.request; assert urllib.request.urlopen('http://127.0.0.1:8000/ready',timeout=5).status == 200",
                ],
                env,
            )
        (tmp_path / "states.json").write_text(json.dumps(rows, indent=2))
    finally:
        run_command([*prefix, "down"], env, timeout=120)
        volumes = run_command([docker, "volume", "ls", "--format", "{{.Name}}"], env).stdout
        assert f"{settings.compose_project_name}_pgdata" in volumes.splitlines()
        after = foreign_snapshot(docker)
        (tmp_path / "isolation.json").write_text(
            json.dumps(
                {
                    "before": before,
                    "after": after,
                    "volume_retained": True,
                    "project": settings.compose_project_name,
                },
                indent=2,
            )
        )
        assert before == after


@pytest.mark.integration
def test_seed_container_create_and_replay(docker_images: str, tmp_path: Path) -> None:
    """The real 512 MiB one-shot imports once and replays without changing rows."""
    docker = docker_images
    before = foreign_snapshot(docker)
    settings = DeploymentSettings(
        _env_file=None,
        compose_project_name=f"insightpilot-test-{uuid4().hex[:12]}",
        postgres_superuser_password=secrets.token_urlsafe(32),
        bootstrap=BootstrapSecrets(
            **{f"{role}_password": secrets.token_urlsafe(32) for role in ("app", "etl", "mcp")}
        ),
    )
    override = tmp_path / "seed.acceptance.yml"
    subnets = isolated_subnets(docker)
    override.write_text(
        yaml.safe_dump(
            {
                "services": {
                    name: {"image": "insightpilot-api:step114"} for name in ("seed", "migrate")
                },
                "networks": {
                    name: {"ipam": {"config": [{"subnet": subnet}]}}
                    for name, subnet in zip(("backend", "egress"), subnets, strict=True)
                },
            }
        )
    )
    call = Invocation(
        command=Command.RUN,
        arguments=["--rm", "seed"],
        deployment_file=ROOT / ".env.deployment.example",
    )
    prefix = [*compose_prefix(docker, settings, call), "-f", str(override)]
    env = process_environment(settings)
    outputs = []
    try:
        for outcome in ("imported", "unchanged"):
            try:
                output = run_command([*prefix, "run", "--rm", "seed"], env, timeout=180)
            except DeploymentError as exc:
                (tmp_path / "seed-failure.txt").write_text(str(exc.context.get("diagnostics", "")))
                raise
            payload = json.loads(output.stdout.splitlines()[-1])
            assert payload["outcome"] == outcome
            outputs.append(output.stdout)
        counts = run_command(
            [
                *prefix,
                "exec",
                "-T",
                "postgres",
                "psql",
                "-U",
                "postgres",
                "-d",
                "insightpilot_business",
                "-c",
                "SELECT status,count(*) FROM biz.orders GROUP BY 1 ORDER BY 2 DESC",
            ],
            env,
        )
        (tmp_path / "seed-command.txt").write_text("\n".join(outputs) + counts.stdout)
        assert "38000" in counts.stdout
    finally:
        run_command([*prefix, "down"], env, timeout=120)
    assert foreign_snapshot(docker) == before
    assert run_command(
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


def test_milvus_nonroot_image_can_traverse_upstream_runtime() -> None:
    source = (ROOT / "docker/Dockerfile.milvus").read_text()
    assert "chmod -R a+rX /milvus" in source
    assert source.index("chmod -R a+rX /milvus") < source.index("USER 10001:10001")
