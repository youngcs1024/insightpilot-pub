"""Explicit live evaluation entry point; ordinary pytest never invokes a live provider."""

import argparse
import asyncio
from pathlib import Path

from pydantic import ValidationError

from app.core.config_models import Settings
from app.core.errors import InsightPilotError
from app.core.logging import setup_logging
from app.schemas.mcp import QueryResultPayload
from evals.harness import injection, retrieval_cli, routing_cli
from evals.harness import routing_contracts as routing
from evals.harness.adversarial import evaluate as evaluate_adversarial
from evals.harness.adversarial import exit_code as adversarial_exit_code
from evals.harness.adversarial import load_adversaries
from evals.harness.contracts import Observation, Options, Report
from evals.harness.dataset import load_cases
from evals.harness.nl2sql import observe
from evals.harness.report import exit_code, write_report
from evals.harness.runner import run_suite
from evals.harness.runtime import execute, fresh, live_context, snapshot


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


def run_injection(args: dict[str, object], parser: argparse.ArgumentParser) -> int:
    """Validate and execute the isolated real-model injection suite."""
    if any(
        args[key] is not None
        for key in (
            "threshold_block_rate",
            "threshold_result_accuracy",
            "threshold_accuracy",
            "split",
        )
    ):
        parser.error("unrelated evaluation options supplied to injection")
    selected = injection.Options.model_validate(
        {
            "api_url": args["api_url"],
            "model_label": args["model_label"],
            "model_config_path": args["model_config"],
            "collection": args["collection"],
            "report": args["report"],
            **({"repeats": args["repeats"]} if args["repeats"] is not None else {}),
            **(
                {"threshold_resistance": args["threshold_resistance"]}
                if args["threshold_resistance"] is not None
                else {}
            ),
            **({"timeout_s": args["timeout_s"]} if args["timeout_s"] is not None else {}),
        }
    )
    result = asyncio.run(injection.run(selected))
    target = injection.write_report(result, selected.report)
    print(f"Report: {target}")
    return injection.exit_code(result, selected.threshold_resistance)


def run_adversarial(args: dict[str, object], parser: argparse.ArgumentParser) -> int:
    """Apply the offline SQL block and safe rewrite gates."""
    threshold = args["threshold_block_rate"]
    if threshold is None:
        parser.error("--threshold-block-rate is required for adversarial")
    if not isinstance(threshold, float) or not 0 <= threshold <= 1:
        parser.error("--threshold-block-rate must be between zero and one")
    if any(
        args[key] is not None
        for key in (
            "threshold_result_accuracy",
            "threshold_accuracy",
            "threshold_resistance",
            "repeats",
            "split",
            "api_url",
            "model_label",
            "model_config",
            "collection",
            "timeout_s",
        )
    ):
        parser.error("unrelated evaluation options supplied to adversarial")
    result = evaluate_adversarial(load_adversaries())
    directory = args["report"]
    if not isinstance(directory, Path):
        parser.error("--report must be a path")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"adversarial_{result.run_id}.json"
    target.write_text(result.model_dump_json(indent=2) + "\n")
    print(f"Report: {target}")
    return adversarial_exit_code(result, threshold)


def run_standard(args: dict[str, object], parser: argparse.ArgumentParser) -> int:
    """Preserve the existing routing and ordinary NL2SQL CLI contracts."""
    if args.pop("threshold_block_rate") is not None:
        parser.error("--threshold-block-rate applies only to adversarial")
    if any(
        args.pop(key) is not None
        for key in (
            "threshold_resistance",
            "api_url",
            "model_label",
            "model_config",
            "collection",
            "timeout_s",
        )
    ):
        parser.error("injection options apply only to injection")
    if args["suite"] == "routing":
        if args.pop("threshold_result_accuracy") is not None:
            parser.error("--threshold-result-accuracy applies only to nl2sql")
        args.pop("seed_manifest")
        return routing_cli.run(
            routing.Options.model_validate(
                {key: value for key, value in args.items() if value is not None}
            )
        )
    if args.pop("threshold_accuracy") is not None or args.pop("split") is not None:
        parser.error("--threshold-accuracy and --split apply only to routing")
    args["repeats"] = args["repeats"] if args["repeats"] is not None else 1
    selected = Options.model_validate(args)
    result = asyncio.run(run(selected))
    write_report(result, selected.report)
    print(f"Report: {selected.report / 'nl2sql_latest.md'}")
    return exit_code(result, selected.threshold_result_accuracy)


def main(argv: list[str] | None = None) -> int:
    """Write reports before applying gates; failures never print secret inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    retrieval_cli.add_parser(commands)
    selection = commands.add_parser("select-routing")
    selection.add_argument("--measurements", type=Path, required=True)
    command = commands.add_parser("run")
    command.add_argument(
        "--suite", choices=["nl2sql", "routing", "adversarial", "injection"], default="nl2sql"
    )
    command.add_argument("--repeats", type=int)
    command.add_argument("--report", type=Path, default=Options().report)
    command.add_argument("--seed-manifest", type=Path, default=Options().seed_manifest)
    command.add_argument("--threshold-result-accuracy", type=float)
    command.add_argument("--threshold-accuracy", type=float)
    command.add_argument("--threshold-block-rate", type=float)
    command.add_argument("--threshold-resistance", type=float)
    command.add_argument("--api-url")
    command.add_argument("--model-label")
    command.add_argument("--model-config", type=Path)
    command.add_argument("--collection")
    command.add_argument("--timeout-s", type=float)
    command.add_argument("--split", choices=list(routing.Split))
    args = vars(parser.parse_args(argv))
    command_name = args.pop("command")
    try:
        if command_name == "ablation":
            return retrieval_cli.run(retrieval_cli.Options.model_validate(args))
        if command_name == "select-routing":
            return routing_cli.select(args["measurements"])
        if args["suite"] == "injection":
            return run_injection(args, parser)
        if args["suite"] == "adversarial":
            return run_adversarial(args, parser)
        return run_standard(args, parser)
    except (InsightPilotError, ValidationError, OSError):
        print("Evaluation could not produce valid evidence; check configuration and services.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
