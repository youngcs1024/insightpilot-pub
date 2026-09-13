# ruff: noqa: PLR2004 -- explicit metric oracles and acceptance quotas.
"""Hand-computed metric oracles, independent of the production sorting/filtering code."""

import math
from uuid import UUID

import pytest

from evals.harness.contracts import EvaluationError
from evals.harness.ir_metrics import IRMetrics, measure


def identifiers(count: int) -> list[UUID]:
    return [UUID(int=index + 1) for index in range(count)]


def test_ndcg_matches_hand_computed_example() -> None:
    a, b, c, d, e = identifiers(5)
    # Ideal grades 3,3,2,1,0; observed grades 2,0,3,1,3.
    numerator = 3 + 7 / math.log2(4) + 1 / math.log2(5) + 7 / math.log2(6)
    denominator = 7 + 7 / math.log2(3) + 3 / math.log2(4) + 1 / math.log2(5)
    result = measure([a, b, c, d, e], {a: 2, b: 0, c: 3, d: 1, e: 3})
    assert result.ndcg_10 == pytest.approx(numerator / denominator)


def test_recall_counts_grade_two_and_above() -> None:
    a, b, c, d = identifiers(4)
    assert measure([a, b], {a: 1, b: 2, c: 3, d: 0}).recall_5 == 0.5


def test_mrr_uses_first_relevant() -> None:
    a, b, c = identifiers(3)
    assert measure([a, b, c], {a: 1, b: 0, c: 2}).mrr_10 == pytest.approx(1 / 3)


def test_perfect_ranking_scores_one() -> None:
    keys = identifiers(5)
    result = measure(keys, dict.fromkeys(keys, 3))
    assert all(value == 1 for value in result.model_dump().values())


def test_empty_results_score_zero() -> None:
    assert measure([], {UUID(int=1): 3}) == IRMetrics()
    assert measure([], {}) == IRMetrics()


def test_precision_uses_fixed_five_denominator() -> None:
    key = UUID(int=1)
    assert measure([key], {key: 3}).precision_5 == 0.2


def test_ideal_uses_all_judgments_not_retrieved_only() -> None:
    a, b = identifiers(2)
    assert measure([a], {a: 2, b: 3}).ndcg_10 < 1


def test_cutoffs_and_no_answer_queries() -> None:
    keys = identifiers(11)
    grades = dict.fromkeys(keys, 0)
    grades[keys[-1]] = 3
    assert measure(keys, grades) == IRMetrics()
    assert measure(keys, dict.fromkeys(keys, 0)) == IRMetrics()


def test_duplicate_unjudged_and_invalid_grades_fail() -> None:
    a, b = identifiers(2)
    for ranking, grades in [([a, a], {a: 3}), ([b], {a: 3}), ([a], {a: 4}), ([a], {a: True})]:
        with pytest.raises(EvaluationError):
            measure(ranking, grades)
