"""SQL execution success rate on ordinary attempts."""

from evals.harness.contracts import CaseResult, Score


def score(results: list[CaseResult]) -> Score:
    """Keep failed and clarified ordinary attempts in the denominator."""
    selected = [r for r in results if not r.adversarial]
    return Score(passed=sum(r.execution_match for r in selected), total=len(selected))
