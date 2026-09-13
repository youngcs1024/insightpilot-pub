"""Explicit live evaluation entry point; ordinary pytest never invokes a live provider."""

import argparse
import asyncio
from pathlib import Path

from pydantic import ValidationError

from app.core.config_models import Settings
from app.core.errors import InsightPilotError
from app.core.logging import setup_logging
from app.schemas.mcp import QueryResultPayload
from evals.harness.contracts import Observation, Options, Report
from evals.harness.dataset import load_cases
from evals.harness.nl2sql import observe
from evals.harness.report import exit_code, write_report
from evals.harness.runner import run_suite
from evals.harness.runtime import execute, fresh, live_context, snapshot
from evals.harness import retrieval_cli


async def run(options: Options) -> Report:
    """Only questions reach the generator; canonical SQL belongs to the evaluator."""
    cases = load_cases()
    settings = Settings.load()
    setup_logging(settings)
    async with live_context(settings) as ctx:
        config = await snapshot(ctx, options)

        async def generate(question: str) -> Observation:
            return await observe(question, fresh(ctx))

        async def canonical(sql: str) -> QueryResultPayload:
            return await execute(ctx, sql)

        return await run_suite(cases, options, config, generate, canonical)


def main(argv: list[str] | None = None) -> int:
    """Write reports before applying gates; failures never print secret inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    retrieval_cli.add_parser(commands)
    command = commands.add_parser("run")
    command.add_argument("--suite", choices=["nl2sql"], default="nl2sql")
    command.add_argument("--repeats", type=int, default=1)
    command.add_argument("--report", type=Path, default=Options().report)
    command.add_argument("--seed-manifest", type=Path, default=Options().seed_manifest)
    command.add_argument("--threshold-result-accuracy", type=float)
    args = vars(parser.parse_args(argv))
    command_name = args.pop("command")
    try:
        if command_name == "ablation":
            return retrieval_cli.run(retrieval_cli.Options.model_validate(args))
        options = Options.model_validate(args)
        report = asyncio.run(run(options))
        write_report(report, options.report)
        print(f"Report: {options.report / 'nl2sql_latest.md'}")
        return exit_code(report, options.threshold_result_accuracy)
    except (InsightPilotError, ValidationError, OSError):
        print("Evaluation could not produce valid evidence; check configuration and services.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
