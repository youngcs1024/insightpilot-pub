"""Production retrieval adapter and immutable FP32 candidate replay."""

import time

from app.clients.model_runtime import ModelRuntimeClient
from app.core.deadline import Deadline
from app.core.errors import InsightPilotError
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.reranking import rerank_candidates
from app.schemas.retrieval import ObservedRetrieval, RetrievalQuery, RetrievalTrace
from evals.harness.contracts import EvaluationError
from evals.harness.retrieval_contracts import Arm, Attempt, RetrievalCase, arm_config


async def observe(case: RetrievalCase, pipeline: RetrievalPipeline, arm: Arm) -> Attempt:
    """Only query and time reach the pipeline; all failures remain explicit attempts."""
    try:
        result = await pipeline.retrieve_observed(
            RetrievalQuery(
                standalone=case.text, original_question=case.text, time_scope=case.time_scope
            ),
            deadline=Deadline(time.monotonic() + 90),
        )
        return Attempt(query_id=case.id, arm=arm, observed=result)
    except InsightPilotError as exc:
        return Attempt(query_id=case.id, arm=arm, failure_code=exc.code)


async def replay_fp32(original: Attempt, model: ModelRuntimeClient) -> Attempt:
    """Never re-encode or re-search after the service changes precision."""
    observed = original.observed
    if original.arm != Arm.D_RERANK or observed is None or original.failure_code:
        raise EvaluationError("FP32 replay requires a complete D FP16 observation")
    started = time.monotonic()
    trace = RetrievalTrace(encoded=observed.trace.encoded)
    try:
        ranked = await rerank_candidates(
            observed.result.query.standalone,
            observed.trace.admitted,
            model,
            arm_config(Arm.D_FP32, observed.result.retrieval_config.filtering),
            deadline=Deadline(started + 90),
            trace=trace,
        )
    except InsightPilotError as exc:
        return Attempt(query_id=original.query_id, arm=Arm.D_FP32, failure_code=exc.code)
    result = observed.result.model_copy(deep=True)
    result.candidates = ranked.candidates
    result.reranked = ranked.reranked
    result.degradation = ranked.degradation
    result.top_rerank_score = ranked.top_rerank_score
    result.meets_floor = ranked.meets_floor
    result.rerank_metadata = ranked.response.metadata if ranked.response else None
    result.stages = result.stages[:2] + ranked.stages
    result.timings.rerank_ms = ranked.stages[0].elapsed_ms
    result.timings.filter_ms = sum(stage.elapsed_ms for stage in ranked.stages[1:])
    # The FP32 control is replay-only: do not invent a new end-to-end measurement.
    result.timings.total_ms = int((time.monotonic() - started) * 1000)
    result.timings.encode_ms = 0
    result.timings.search_ms = 0
    result.timings.admission_ms = 0
    return Attempt(
        query_id=original.query_id,
        arm=Arm.D_FP32,
        observed=ObservedRetrieval(result=result, trace=trace),
    )


def verify_precision(left: Attempt, right: Attempt) -> None:
    """All paired candidates, model revisions and thresholds must be identical."""
    if left.observed is None or right.observed is None:
        raise EvaluationError("Missing precision observation")
    a, b = left.observed, right.observed
    first, second = a.trace.rerank_response, b.trace.rerank_response
    if not a.trace.admitted:
        if b.trace.admitted or first is not None or second is not None:
            raise EvaluationError("Inconsistent empty precision control")
        return
    if (
        a.trace.admitted != b.trace.admitted
        or a.trace.encoded != b.trace.encoded
        or a.result.query != b.result.query
        or a.result.retrieval_config != b.result.retrieval_config
        or first is None
        or second is None
        or first.metadata.precision != "fp16"
        or second.metadata.precision != "fp32"
        or first.metadata.model_dump(exclude={"precision"})
        != second.metadata.model_dump(exclude={"precision"})
    ):
        raise EvaluationError("Precision control identity mismatch")
