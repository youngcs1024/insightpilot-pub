"""Complete semantic binding accuracy, independent of SQL execution."""

from evals.harness.contracts import CaseResult, Score


def score(results: list[CaseResult]) -> Score:
    """Include every ordinary attempt even if SQL generation failed."""
    selected = [r for r in results if not r.adversarial]
    return Score(passed=sum(r.metric_resolution_accuracy for r in selected), total=len(selected))
