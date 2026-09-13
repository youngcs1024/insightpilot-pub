"""Grade complete production observations and select on development data only."""

import math
import statistics
from collections import Counter
from itertools import product
from uuid import UUID

from app.retrieval.config import FilterConfig
from app.retrieval.filtering import filter_ranked
from app.schemas.ingestion import canonical, digest
from app.schemas.model_runtime import EMBED_REVISION, RERANK_REVISION
from app.schemas.retrieval import Candidate
from evals.harness.contracts import EvaluationError
from evals.harness.ir_metrics import IRMetrics, measure
from evals.harness.retrieval import verify_precision
from evals.harness.retrieval_contracts import (
    AblationReport,
    Arm,
    ArmSummary,
    Attempt,
    Dataset,
    Judgments,
    Measurements,
    Percentiles,
    RetrievalCase,
    ScoredAttempt,
    Selection,
    Split,
    arm_config,
)
from evals.harness.retrieval_dataset import applicable
from scripts.model_evidence import DEFAULT_BATCH, EMBED_LENGTH, RERANK_LENGTH

BASE_ARMS = tuple(arm for arm in Arm if arm is not Arm.D_FP32)


def ids(candidates: list[Candidate]) -> list[UUID]:
    """Preserve rank order when joining to deterministic registered identities."""
    return [item.chunk_uuid for item in candidates]


def validate_attempt(attempt: Attempt, case: RetrievalCase, dataset: Dataset) -> None:
    """Refuse drift, silent fallback, changed effective workloads and fabricated ranks."""
    observed = attempt.observed
    if attempt.failure_code or observed is None:
        raise EvaluationError("Missing successful observation")
    result, trace = observed.result, observed.trace
    if not (set(type(result.timings).model_fields) - {"schema_version"}).issubset(
        result.timings.model_fields_set
    ):
        raise EvaluationError("Missing client stage measurement")
    expected = arm_config(attempt.arm, result.retrieval_config.filtering)
    if (
        result.query.standalone != case.text
        or result.query.time_scope != case.time_scope
        or result.corpus_version != dataset.manifest.corpus_version
        or result.retrieval_config != expected
        or result.degradation is not None
        or len(ids(trace.admitted)) != len(set(ids(trace.admitted)))
        or set(ids(trace.ranked)) != set(ids(trace.admitted))
        or len(trace.ranked) != len(trace.admitted)
    ):
        raise EvaluationError("Observation identity or candidate mismatch")
    validate_ranking(attempt)
    validate_provenance(attempt, dataset)


def validate_provenance(attempt: Attempt, dataset: Dataset) -> None:
    """Returned source text and encoding retain the dataset's frozen identities."""
    observed = attempt.observed
    if observed is None or observed.trace.encoded is None:
        raise EvaluationError("Missing encoded query")
    encoded = observed.trace.encoded
    metadata = encoded.metadata
    if (
        metadata is None
        or metadata.precision != "fp16"
        or metadata.embed_revision != EMBED_REVISION
        or metadata.rerank_revision != RERANK_REVISION
        or metadata.embed_max_length != EMBED_LENGTH
        or metadata.embed_batch != DEFAULT_BATCH
        or encoded.dense is None
        or encoded.sparse is None
        or encoded.text != observed.result.query.standalone
        or observed.result.model_metadata != metadata
    ):
        raise EvaluationError("Invalid encoded query identity")
    chunks = {item.chunk_id: item for item in dataset.manifest.chunks}
    for item in observed.trace.admitted:
        source = chunks.get(item.chunk_uuid)
        if (
            source is None
            or source.content_sha256 != digest(item.content)
            or source.content_sha256 != item.content_sha256
            or source.source_path != item.source_path
            or source.effective_from != item.effective_from
            or source.effective_to != item.effective_to
        ):
            raise EvaluationError("Candidate source differs from judged original")


def validate_ranking(attempt: Attempt) -> None:
    """Recompute filtering and verify returned score order, including empty pools."""
    observed = attempt.observed
    if observed is None:
        raise EvaluationError("Missing observation")
    result, trace = observed.result, observed.trace
    if not result.retrieval_config.use_rerank:
        if trace.ranked != trace.admitted or result.candidates != trace.ranked:
            raise EvaluationError("Unreranked arm changed candidate order")
        return
    if not trace.admitted:
        if result.candidates or trace.ranked or trace.rerank_response:
            raise EvaluationError("Invalid empty ranking")
        return
    response = trace.rerank_response
    precision = "fp32" if attempt.arm is Arm.D_FP32 else "fp16"
    if (
        response is None
        or not result.reranked
        or response.metadata.precision != precision
        or response.metadata.rerank_batch != DEFAULT_BATCH
        or response.metadata.rerank_max_length != RERANK_LENGTH
        or len(response.scores) != len(trace.admitted)
        or result.rerank_metadata != response.metadata
    ):
        raise EvaluationError("Invalid rerank workload or precision")
    expected = [item.model_copy(deep=True) for item in trace.admitted]
    for item, score in zip(expected, response.scores, strict=True):
        item.scores.rerank = score
    expected.sort(key=lambda item: item.scores.rerank or 0, reverse=True)
    filtered = filter_ranked(expected, result.retrieval_config.filtering)
    if (
        trace.ranked != expected
        or filtered.candidates != result.candidates
        or (
            filtered.meets_floor != result.meets_floor
            or filtered.top_rerank_score != result.top_rerank_score
        )
    ):
        raise EvaluationError("Recorded ranking does not match raw model scores")


def score_attempt(
    attempt: Attempt, case: RetrievalCase, labels: Judgments, dataset: Dataset
) -> ScoredAttempt:
    """Invalid evidence remains in the denominator rather than disappearing."""
    scored = ScoredAttempt(query_id=case.id, arm=attempt.arm, answerable=case.answerable)
    try:
        validate_attempt(attempt, case, dataset)
        observed = attempt.observed
        if observed is None:
            raise EvaluationError("Missing observation")
        grades = {item.chunk_id: item.grade for item in labels.judgments}
        chunks = {item.chunk_id: item for item in dataset.manifest.chunks}
        scored.ranking = measure(ids(observed.trace.ranked), grades)
        scored.filtered = measure(ids(observed.result.candidates), grades)
        scored.abstained = not observed.result.candidates
        scored.wrong_version_count = sum(
            not applicable(chunks[item.chunk_uuid], case.time_scope)
            for item in observed.trace.ranked
        )
        if scored.wrong_version_count:
            raise EvaluationError("Retrieved inapplicable policy version")
    except EvaluationError:
        scored.ranking = None
        scored.filtered = None
        scored.abstained = None
        scored.failure_code = attempt.failure_code or "EVALUATION_INVALID"
    return scored


def percentiles(values: list[float]) -> Percentiles:
    """Median and nearest-rank p95, with the actual sample count."""
    return Percentiles(
        count=len(values),
        p50=statistics.median(values) if values else None,
        p95=sorted(values)[math.ceil(0.95 * len(values)) - 1] if values else None,
    )


def average(values: list[IRMetrics]) -> IRMetrics:
    """No relevant-query success can be invented from an empty population."""
    return IRMetrics.model_validate(
        {
            name: statistics.mean(getattr(value, name) for value in values) if values else 0
            for name in IRMetrics.model_fields
        }
    )


def summarize(arm: Arm, results: list[ScoredAttempt], attempts: list[Attempt]) -> ArmSummary:
    """Latencies exclude failed attempts and identify FP32 total as replay-only."""
    rows = [item for item in results if item.arm is arm]
    valid = [item for item in rows if item.failure_code is None]
    query_ids = {item.query_id for item in valid}
    timings = [
        item.observed.result.timings
        for item in attempts
        if item.arm is arm and item.query_id in query_ids and item.observed is not None
    ]
    latency = {
        name: percentiles([float(getattr(value, name)) for value in timings])
        for name in ("encode_ms", "search_ms", "admission_ms", "rerank_ms", "filter_ms", "total_ms")
    }
    return ArmSummary(
        arm=arm,
        attempted=len(rows),
        valid=len(valid),
        answerable=sum(item.answerable for item in rows),
        ranking=average(
            [item.ranking for item in valid if item.answerable and item.ranking is not None]
        ),
        filtered=average(
            [item.filtered for item in valid if item.answerable and item.filtered is not None]
        ),
        negative_count=sum(not item.answerable for item in rows),
        correct_abstentions=sum(not item.answerable and item.abstained is True for item in valid),
        false_evidence=sum(not item.answerable and item.abstained is False for item in valid),
        latency_ms=latency,
    )


def evaluate(
    measurements: Measurements,
    dataset: Dataset,
    selection: Selection | None = None,
    *,
    selection_commit: str | None = None,
    require_control: bool = True,
) -> AblationReport:
    """Recompute all metrics from attempts and detect missing/duplicate grid cells."""
    if (
        measurements.dataset_identity != dataset.identity
        or measurements.corpus_version != dataset.manifest.corpus_version
    ):
        raise EvaluationError("Measurements belong to another dataset")
    issues: list[str] = []
    required = tuple(Arm) if require_control else BASE_ARMS
    expected = {(case.id, arm) for case in dataset.cases for arm in required}
    counts = Counter((item.query_id, item.arm) for item in measurements.attempts)
    if set(counts) != expected or any(count != 1 for count in counts.values()):
        issues.append("incomplete_or_duplicate_grid")
    if measurements.source_dirty:
        issues.append("dirty_client_source")
    cases = {case.id: case for case in dataset.cases}
    labels = {item.query_id: item for item in dataset.labels}
    results = [
        score_attempt(item, cases[item.query_id], labels[item.query_id], dataset)
        for item in measurements.attempts
        if item.query_id in cases
    ]
    if any(item.failure_code for item in results):
        issues.append("invalid_attempts")
    if require_control:
        issues.extend(precision_issues(measurements.attempts, dataset))
    if measurements.split is not dataset.cases[0].split:
        issues.append("wrong_partition")
    if measurements.split is Split.FROZEN and (
        selection is None
        or selection.dataset_identity != dataset.identity
        or selection_commit is None
        or measurements.selection_commit != selection_commit
        or measurements.selection_identity != digest(canonical(selection.model_dump(mode="json")))
    ):
        issues.append("uncommitted_development_selection")
    if selection is not None and any(
        item.arm is selection.arm
        and item.observed is not None
        and item.observed.result.retrieval_config != selection.config
        for item in measurements.attempts
    ):
        issues.append("selected_configuration_not_measured")
    summaries = [summarize(arm, results, measurements.attempts) for arm in required]
    if (
        not issues
        and len(
            {tuple(row.ranking.model_dump().values()) for row in summaries if row.arm in BASE_ARMS}
        )
        == 1
    ):
        issues.append("all_arm_metrics_identical_review_query_difficulty")
    return AblationReport(
        measurements=measurements,
        selection=selection,
        selection_commit=selection_commit,
        results=results,
        summaries=summaries,
        evidence_valid=not issues,
        issues=issues,
    )


def precision_issues(attempts: list[Attempt], dataset: Dataset) -> list[str]:
    """One paired comparison per question, never score correlation as quality evidence."""
    indexed = {(item.query_id, item.arm): item for item in attempts}
    for case in dataset.cases:
        left, right = indexed.get((case.id, Arm.D_RERANK)), indexed.get((case.id, Arm.D_FP32))
        if left is None or right is None:
            return ["missing_precision_control"]
        try:
            verify_precision(left, right)
        except EvaluationError:
            return ["invalid_precision_control"]
    return []


def choose(report: AblationReport, dataset: Dataset) -> Selection:
    """Tuning accepts development-only data and a complete eight-arm measurement."""
    if report.measurements.split is not Split.DEVELOPMENT or any(
        case.split is not Split.DEVELOPMENT for case in dataset.cases
    ):
        raise EvaluationError("Frozen labels are forbidden in tuning")
    if not report.evidence_valid:
        raise EvaluationError("Cannot tune on incomplete evidence")
    summaries = [item for item in report.summaries if item.arm in BASE_ARMS]
    winner = min(
        summaries,
        key=lambda item: (
            -item.ranking.ndcg_10,
            -item.ranking.recall_10,
            item.latency_ms["total_ms"].p50 or 0,
            BASE_ARMS.index(item.arm),
        ),
    )
    config = arm_config(winner.arm)
    if config.use_rerank:
        config.filtering = tune_filter(winner.arm, report.measurements.attempts, dataset)
    return Selection(
        dataset_identity=dataset.identity,
        development_sha=report.measurements.client_sha,
        arm=winner.arm,
        config=config,
        rationale="Development pre-filter nDCG@10, Recall@10, client p50, stable arm order; filter sweep uses development labels only.",
    )


def tune_filter(arm: Arm, attempts: list[Attempt], dataset: Dataset) -> FilterConfig:
    """Keep the candidate pool and all non-threshold controls fixed during the sweep."""
    if any(case.split is not Split.DEVELOPMENT for case in dataset.cases):
        raise EvaluationError("Frozen labels are forbidden in tuning")
    labels = {
        item.query_id: {label.chunk_id: label.grade for label in item.judgments}
        for item in dataset.labels
    }
    cases = {case.id: case for case in dataset.cases}
    options: list[tuple[tuple[float, int, float, float, float], FilterConfig]] = []
    for floor, ratio in product((0.2, 0.3, 0.4), (0.4, 0.5, 0.6)):
        config = FilterConfig(absolute_floor=floor, dynamic_ratio=ratio)
        scores: list[float] = []
        false_evidence = 0
        for attempt in attempts:
            if attempt.arm is not arm or attempt.observed is None:
                continue
            kept = filter_ranked(attempt.observed.trace.ranked, config).candidates
            if cases[attempt.query_id].answerable:
                scores.append(measure(ids(kept), labels[attempt.query_id]).ndcg_10)
            else:
                false_evidence += bool(kept)
        quality = statistics.mean(scores) if scores else 0
        distance = abs(floor - 0.3) + abs(ratio - 0.5)
        options.append(((-quality, false_evidence, distance, floor, ratio), config))
    return min(options, key=lambda item: item[0])[1]
