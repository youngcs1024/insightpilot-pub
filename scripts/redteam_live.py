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
from scripts.deployment import DeploymentError, DeploymentSettings, ROOT, process_environment
from scripts.tunnel_deployment import prepare


def free_port() -> int:
    """Allocate a loopback port without changing another process."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def command_prefix(project: str, deployment_file: Path) -> list[str]:
    """Never let Compose discover a foreign project or environment file."""
    docker = shutil.which("docker")
    if docker is None:
        raise DeploymentError("Docker is unavailable")
    return [
        docker, "compose", "-p", project, "--project-directory", str(ROOT),
        "--env-file", str(deployment_file), "-f", str(ROOT / "docker-compose.yml"),
        "-f", str(ROOT / "docker-compose.redteam.yml"), "--profile", "retrieval",
        "--profile", "seed",
    ]


def compose(command: list[str], environment: dict[str, str], *, timeout: int) -> None:
    """Use the bounded operator process runner without printing environment secrets."""
    from scripts.deployment import run_command  # noqa: PLC0415 -- startup-only operator.

    run_command(command, environment, timeout=timeout)


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
    if not args.project.startswith("insightpilot-redteam-") or len(args.project) < 21:
        parser.error("--project must be a unique insightpilot-redteam-* name")
    try:
        deployment_file = args.deployment_file.resolve(strict=True)
        settings = DeploymentSettings(_env_file=deployment_file)
        if settings.llm_api_key is None:
            raise DeploymentError("Real-model API configuration is required")
        port = free_port()
        retrieval = settings.retrieval.model_copy(deep=True)
        retrieval.milvus.collection = "kb_chunks_redteam"
        retrieval.enabled = True
        environment = process_environment(settings)
        environment.update(prepare(ROOT, write_config=True))
        environment.update({
            "IP_RETRIEVAL": retrieval.model_dump_json(),
            "IP_API_HOST_PORT": str(port),
            "IP_REDTEAM_SEED_IMAGE": args.project + ":seed",
        })
        prefix = command_prefix(args.project, deployment_file)
        compose(
            [*prefix, "build", "api", "mcp", "model-tunnel", "etcd", "minio", "milvus"],
            environment,
            timeout=1200,
        )
        compose([*prefix, "build", "seed"], environment, timeout=600)
        compose(
            [*prefix, "up", "-d", "--wait", "--wait-timeout", "300", "model-tunnel", "milvus"],
            environment,
            timeout=360,
        )
        compose(
            [*prefix, "run", "--rm", "model-diagnostics", "python", "-m",
             "scripts.probe_model_runtime", "--ready"],
            environment,
            timeout=90,
        )
        compose(
            [*prefix, "up", "-d", "--wait", "--wait-timeout", "300", "api"],
            environment,
            timeout=600,
        )
        return asyncio.run(evaluate(injection.Options(
            api_url=f"http://127.0.0.1:{port}", model_label=args.model_label,
            model_config_path=deployment_file, collection="kb_chunks_redteam",
            report=args.report,
        )))
    except (OSError, ValidationError, InsightPilotError, TimeoutError) as exc:
        print(f"Red-team run failed: {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
