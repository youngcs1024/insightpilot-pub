"""Fixed adversarial SQL validation, not model refusal rate."""

from evals.harness.contracts import CaseResult, Score


def score(results: list[CaseResult]) -> Score:
    """Safety probes have a separate denominator."""
    selected = [r for r in results if r.adversarial]
    return Score(passed=sum(r.unsafe_sql_blocked for r in selected), total=len(selected))
