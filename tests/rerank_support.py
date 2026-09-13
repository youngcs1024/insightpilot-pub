"""Detached candidates and typed responses for filtering and model transport tests."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.schemas.model_runtime import RerankResult
from app.schemas.retrieval import Candidate
from tests.ingestion_support import model_metadata
from tests.retrieval_support import candidate


def scored(values: list[float], sources: list[str] | None = None) -> list[Candidate]:
    result = []
    for index, value in enumerate(values):
        item = candidate()
        item.source_path = sources[index] if sources else f"source-{index}.md"
        item.scores.rerank = value
        result.append(item)
    return result


def model(scores: list[float]) -> SimpleNamespace:
    response = RerankResult(
        request_id="rerank-test",
        ms=1,
        queue_ms=0,
        inference_ms=1,
        metadata=model_metadata(),
        scores=scores,
    )
    return SimpleNamespace(rerank=AsyncMock(return_value=response))
