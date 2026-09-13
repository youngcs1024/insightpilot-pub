"""One bounded cross-encoder request shared by retrieval and dedicated live acceptance."""

import time

import structlog

from app.clients.model_runtime import ModelRuntimeClient
from app.core.deadline import Deadline
from app.core.errors import RetrievalConfigurationError
from app.core.observability import TraceMetadata, observe
from app.retrieval.config import RetrievalConfig
from app.retrieval.filtering import diagnostic, fallback, filter_ranked
from app.schemas.model_runtime import ModelFailureKind
from app.schemas.retrieval import (
    Candidate, RankingResult, RetrievalStage, RetrievalTrace, StageStatus,
)
from model_runtime.errors import ModelError

logger = structlog.get_logger(__name__)
DEGRADABLE = frozenset(
    {
        ModelFailureKind.UNAVAILABLE,
        ModelFailureKind.QUEUE_FULL,
        ModelFailureKind.OOM,
        ModelFailureKind.DEADLINE,
    }
)


async def rerank_candidates(
    query: str,
    candidates: list[Candidate],
    model: ModelRuntimeClient | None,
    config: RetrievalConfig,
    *,
    deadline: Deadline,
    trace: RetrievalTrace | None = None,
) -> RankingResult:
    """Retain child/parent text and native scores; reject contract and authorization failures."""
    deadline.check("rerank")
    detached = [item.model_copy(deep=True) for item in candidates]
    if trace is not None:
        trace.admitted = [item.model_copy(deep=True) for item in detached]
        trace.ranked = [item.model_copy(deep=True) for item in detached]
    if not config.use_rerank or not detached:
        status = StageStatus.DISABLED if not config.use_rerank else StageStatus.EMPTY
        return RankingResult(
            candidates=detached,
            stages=[
                diagnostic(stage, len(detached), detached, status=status)
                for stage in (
                    RetrievalStage.RERANK,
                    RetrievalStage.THRESHOLD,
                    RetrievalStage.DIVERSITY,
                    RetrievalStage.TRUNCATION,
                )
            ],
        )
    if model is None:
        raise RetrievalConfigurationError(reason="model_client_required")
    for item in detached:
        item.scores.rerank = None
    started = time.monotonic()
    with observe("retrieval_rerank", TraceMetadata(row_count=len(detached))):
        try:
            response = await model.rerank(
                query, [item.content for item in detached], deadline=deadline, max_length=320
            )
        except ModelError as exc:
            if exc.kind not in DEGRADABLE:
                raise
            deadline.check("rerank_fallback")
            rerank_ms = int((time.monotonic() - started) * 1000)
            logger.exception("retrieval_rerank_degraded", code=exc.code)
            with observe("retrieval_filter", TraceMetadata(status="degraded")):
                result = fallback(detached, config.filtering)
            result.degradation = exc.kind
            result.stages.insert(
                0,
                diagnostic(
                    RetrievalStage.RERANK,
                    len(detached),
                    detached,
                    elapsed_ms=rerank_ms,
                    status=StageStatus.DEGRADED,
                ),
            )
            return result
    deadline.check("retrieval_filter")
    rerank_ms = int((time.monotonic() - started) * 1000)
    for item, value in zip(detached, response.scores, strict=True):
        item.scores.rerank = value
    if trace is not None:
        trace.rerank_response = response.model_copy(deep=True)
        trace.ranked = [
            item.model_copy(deep=True)
            for item in sorted(detached, key=lambda item: item.scores.rerank or 0, reverse=True)
        ]
    with observe("retrieval_filter", TraceMetadata(row_count=len(detached))):
        result = filter_ranked(detached, config.filtering)
    result.response = response
    result.stages.insert(
        0, diagnostic(RetrievalStage.RERANK, len(detached), detached, elapsed_ms=rerank_ms)
    )
    return result
