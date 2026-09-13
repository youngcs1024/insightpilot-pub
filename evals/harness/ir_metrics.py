"""Deterministic graded IR metrics; missing judgments are invalid evidence."""

import math
from collections.abc import Mapping, Sequence
from uuid import UUID

from pydantic import Field

from app.schemas.mcp import Contract
from evals.harness.contracts import EvaluationError

RELEVANT = 2
MAX_GRADE = 3


class IRMetrics(Contract):
    """Cutoffs are fixed across arms, including short result lists."""

    recall_5: float = Field(default=0, ge=0, le=1)
    recall_10: float = Field(default=0, ge=0, le=1)
    ndcg_10: float = Field(default=0, ge=0, le=1)
    mrr_10: float = Field(default=0, ge=0, le=1)
    precision_5: float = Field(default=0, ge=0, le=1)


def measure(ranking: Sequence[UUID], grades: Mapping[UUID, int]) -> IRMetrics:
    """Use the entire judged set for recall/IDCG, never the retrieved subset."""
    if len(set(ranking)) != len(ranking) or any(item not in grades for item in ranking):
        raise EvaluationError("Duplicate results or unjudged retrieval")
    if any(type(grade) is not int or not 0 <= grade <= MAX_GRADE for grade in grades.values()):
        raise EvaluationError("Invalid relevance grade")
    values = [grades[item] for item in ranking]
    relevant = sum(grade >= RELEVANT for grade in grades.values())
    ideal = dcg(sorted(grades.values(), reverse=True)[:10])
    return IRMetrics(
        recall_5=sum(value >= RELEVANT for value in values[:5]) / relevant if relevant else 0,
        recall_10=sum(value >= RELEVANT for value in values[:10]) / relevant if relevant else 0,
        ndcg_10=min(1.0, dcg(values[:10]) / ideal) if ideal else 0,
        mrr_10=next((1 / rank for rank, grade in enumerate(values[:10], 1) if grade >= RELEVANT), 0),
        precision_5=sum(value >= RELEVANT for value in values[:5]) / 5,
    )


def dcg(grades: Sequence[int]) -> float:
    """Exponential gain with base-two logarithmic rank discount."""
    return sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(grades))
