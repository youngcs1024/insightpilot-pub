"""Numerically stable sigmoid and validated input-order rerank output."""

import math

from pydantic import BaseModel

from app.schemas.model_runtime import Score


class Scores(BaseModel):
    """No sorting occurs at the model service boundary."""

    scores: list[Score]


def sigmoid(value: float) -> float:
    """Normalize extreme logits without an exponential overflow."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)
