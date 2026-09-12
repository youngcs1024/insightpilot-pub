"""Canonical execution result agreement on ordinary attempts."""

from evals.harness.contracts import CaseResult, Score


def score(results: list[CaseResult]) -> Score:
    """Do not discard failed attempts before calculating accuracy."""
    selected = [r for r in results if not r.adversarial]
    return Score(passed=sum(r.result_accuracy for r in selected), total=len(selected))
