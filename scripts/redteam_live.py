"""Start a named isolated real-model stack, then run the dedicated injection gate."""

import argparse
import asyncio
import shutil
import socket
import sys
from pathlib import Path

from pydantic import ValidationError

from app.core.errors import InsightPilotError
from evals.harness import injection
from scripts.deployment import (
    ROOT,
    DeploymentError,
    DeploymentSettings,
    process_environment,
    run_command,
)
from scripts.tunnel_deployment import prepare


def free_port() -> int:
    """Allocate a loopback port without changing another process."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def command_prefix(project: str, deployment_file: Path) -> list[str]:
    """Never let Compose discover a foreign project or environment file."""
    docker = shutil.which("docker")
    if docker is None:
        raise DeploymentError("Docker is unavailable")
    return [
        docker,
        "compose",
        "-p",
        project,
        "--project-directory",
        str(ROOT),
        "--env-file",
        str(deployment_file),
        "-f",
        str(ROOT / "docker-compose.yml"),
        "-f",
        str(ROOT / "docker-compose.redteam.yml"),
        "--profile",
        "retrieval",
        "--profile",
        "seed",
    ]


def compose(command: list[str], environment: dict[str, str], *, stage: str, timeout: int) -> None:
    """Use the bounded operator process runner without printing environment secrets."""
    try:
        run_command(command, environment, timeout=timeout)
    except DeploymentError as exc:
        raise DeploymentError(f"Red-team Compose stage failed: {stage}") from exc


def require_new_project(project: str, environment: dict[str, str]) -> None:
    """Prevent an old test volume or container from contaminating a fresh score."""
    docker = shutil.which("docker")
    if docker is None:
        raise DeploymentError("Docker is unavailable")
    for kind in ("ps", "volume ls"):
        command = [
            docker,
            *kind.split(),
            "--filter",
            f"label=com.docker.compose.project={project}",
            "--format",
            "{{.ID}}" if kind == "ps" else "{{.Name}}",
        ]
        if kind == "ps":
            command.insert(2, "--all")
        if run_command(command, environment).stdout.strip():
            raise DeploymentError("Red-team Compose project already owns resources")


async def evaluate(options: injection.Options) -> int:
    """Persist all 24 full API attempts before applying the quality threshold."""
    report = await injection.run(options)
    target = injection.write_report(report, options.report)
    print(f"Report: {target}")
    print(f"Resistance: {report.resistance.passed}/{report.resistance.total}")
    return injection.exit_code(report, options.threshold_resistance)


def main(argv: list[str] | None = None) -> int:
    """Keep the isolated stack for inspection; no volumes or files are deleted."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-file", type=Path, default=ROOT / ".env.deployment")
    parser.add_argument("--project", required=True, help="Unique insightpilot-redteam-* project")
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--report", type=Path, default=ROOT / "evals/reports")
    args = parser.parse_args(argv)
    if not args.project.startswith("insightpilot-redteam-") or len(args.project) < len(
        "insightpilot-redteam-x"
    ):
        parser.error("--project must be a unique insightpilot-redteam-* name")
    try:
        deployment_file = args.deployment_file.resolve(strict=True)
        settings = DeploymentSettings(_env_file=deployment_file)
        if settings.llm_api_key is None:
            raise DeploymentError("Real-model API configuration is required")
        if args.model_label != settings.llm_model:
            raise DeploymentError("Model label must match the configured model")
        port = free_port()
        retrieval = settings.retrieval.model_copy(deep=True)
        retrieval.milvus.collection = "kb_chunks_redteam"
        retrieval.enabled = True
        environment = process_environment(settings)
        environment.update(prepare(ROOT, write_config=True))
        environment.update(
            {
                "IP_RETRIEVAL": retrieval.model_dump_json(),
                "IP_API_HOST_PORT": str(port),
                "IP_REDTEAM_SEED_IMAGE": args.project + ":seed",
            }
        )
        require_new_project(args.project, environment)
        prefix = command_prefix(args.project, deployment_file)
        compose(
            [*prefix, "build", "api", "mcp", "model-tunnel", "etcd", "minio", "milvus"],
            environment,
            stage="build services",
            timeout=1200,
        )
        compose([*prefix, "build", "seed"], environment, stage="build seed", timeout=600)
        compose(
            [*prefix, "up", "-d", "--wait", "--wait-timeout", "300", "model-tunnel", "milvus"],
            environment,
            stage="start model storage",
            timeout=360,
        )
        compose(
            [
                *prefix,
                "run",
                "--rm",
                "model-diagnostics",
                "python",
                "-m",
                "scripts.probe_model_runtime",
                "--ready",
            ],
            environment,
            stage="probe model runtime",
            timeout=90,
        )
        compose(
            [*prefix, "up", "-d", "--wait", "--wait-timeout", "300", "api"],
            environment,
            stage="start API and ingest",
            timeout=600,
        )
        return asyncio.run(
            evaluate(
                injection.Options(
                    api_url=f"http://127.0.0.1:{port}",
                    model_label=args.model_label,
                    model_config_path=deployment_file,
                    collection="kb_chunks_redteam",
                    report=args.report,
                )
            )
        )
    except (OSError, ValidationError, InsightPilotError, TimeoutError) as exc:
        description = str(exc) if isinstance(exc, DeploymentError) else type(exc).__name__
        print(f"Red-team run failed: {description}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
