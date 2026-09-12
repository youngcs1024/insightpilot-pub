"""Typed sparse conversion, shared by direct BGE forward and contract tests."""

from pydantic import BaseModel

from app.schemas.model_runtime import Dense, Sparse


class Embeddings(BaseModel):
    """Validated model-adapter output before it crosses into HTTP orchestration."""

    dense: list[Dense]
    sparse: list[Sparse]


def lexical_weights(
    weights: list[float], token_ids: list[int], excluded: set[int]
) -> dict[int, float]:
    """BGE lexical weights use maximum positive weight per non-special token ID."""
    result: dict[int, float] = {}
    for weight, token_id in zip(weights, token_ids, strict=True):
        if token_id not in excluded and weight > 0:
            result[token_id] = max(result.get(token_id, 0.0), weight)
    return result
