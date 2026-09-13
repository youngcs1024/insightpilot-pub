"""Suite B phases keep operator-owned precision switching outside application code."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import Field

from app.schemas.mcp import Contract
from evals.harness.ablation import choose, evaluate
from evals.harness.contracts import EvaluationError
from evals.harness.retrieval_contracts import ROOT, Measurements, Split
from evals.harness.retrieval_dataset import load_dataset
from evals.harness.retrieval_report import exit_code, write_report
from evals.harness.retrieval_runtime import collect, committed_selection, control
from scripts.model_evidence import Provenance

if TYPE_CHECKING:
    import argparse


class Options(Contract):
    """Public options contain paths and numeric gates, never credentials."""

    suite: Literal["retrieval"] = "retrieval"
    arms: Literal["all"] = "all"
    phase: Literal["collect", "control", "select", "report"] = "collect"
    split: Split = Split.DEVELOPMENT
    dataset: Path = ROOT
    report: Path = ROOT.parents[1] / "reports/retrieval_ablation.md"
    measurements: Path | None = None
    provenance: Path | None = None
    selection: Path = ROOT / "selected.yaml"
    threshold: float | None = Field(default=None, ge=0, le=1)


def add_parser(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add one subcommand without changing the existing NL2SQL run interface."""
    command = commands.add_parser("ablation")
    command.add_argument("--suite", choices=["retrieval"], default="retrieval")
    command.add_argument("--arms", choices=["all"], default="all")
    command.add_argument(
        "--phase", choices=["collect", "control", "select", "report"], default="collect"
    )
    command.add_argument("--split", choices=list(Split), default=Split.DEVELOPMENT)
    command.add_argument("--dataset", type=Path, default=ROOT)
    command.add_argument("--report", type=Path, default=Options().report)
    command.add_argument("--measurements", type=Path)
    command.add_argument("--provenance", type=Path)
    command.add_argument("--selection", type=Path, default=Options().selection)
    command.add_argument("--threshold", type=float)


def run(options: Options) -> int:
    """Collect eight arms, replay FP32, commit development selection, then report frozen."""
    selection, commit = None, None
    if options.split is Split.FROZEN:
        selection, commit = committed_selection(options.selection)
    dataset = load_dataset(options.split, options.dataset)
    path = options.measurements or options.report.with_suffix(".raw.json")
    if options.phase == "collect":
        raw = asyncio.run(collect(dataset, provenance(options), selection, commit))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(raw.model_dump_json(indent=2))
    else:
        raw = Measurements.model_validate_json(path.read_text())
    if options.phase == "control":
        raw = asyncio.run(control(raw, provenance(options)))
        path.write_text(raw.model_dump_json(indent=2))
    report = evaluate(
        raw,
        dataset,
        selection,
        selection_commit=commit,
        require_control=options.phase != "select",
    )
    write_report(report, options.report)
    if options.phase == "select":
        selected = choose(report, dataset)
        options.selection.parent.mkdir(parents=True, exist_ok=True)
        options.selection.write_text(
            yaml.safe_dump(selected.model_dump(mode="json"), allow_unicode=True, sort_keys=False)
        )
        print("Development selection written; commit before evaluating frozen labels.")
    print("Report: " + str(options.report))
    return exit_code(report, options.threshold)


def provenance(options: Options) -> Provenance:
    """Deployment receipts must be supplied explicitly by the authorized operator."""
    if options.provenance is None:
        raise EvaluationError("A server provenance artifact is required for live evaluation")
    return Provenance.model_validate_json(options.provenance.read_text())
