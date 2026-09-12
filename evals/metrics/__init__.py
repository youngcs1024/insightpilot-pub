"""Discover only project-owned scorer modules in deterministic filename order."""

from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from typing import cast

from evals.harness.contracts import CaseResult, EvaluationError, Score, Summary

type Scorer = Callable[[list[CaseResult]], Score]


def summarize(results: list[CaseResult]) -> Summary:
    """Require the complete metric registry; unknown or absent scorers never disappear."""
    scores: dict[str, Score] = {}
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if path.name.startswith("_"):
            continue
        module = import_module(f"evals.metrics.{path.stem}")
        scorer = getattr(module, "score", None)
        if not callable(scorer):
            raise EvaluationError("Scorer must export score(results)")
        scores[path.stem] = cast("Scorer", scorer)(results)
    return Summary.model_validate(scores)
