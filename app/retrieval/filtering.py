"""Pure source diversity and normalized relevance filtering, with stage diagnostics."""

import re
import time
from collections import Counter
from pathlib import PurePosixPath
from typing import Literal

from app.core.errors import IngestionRegistryError, RetrievalConfigurationError
from app.retrieval.config import FilterConfig
from app.schemas.retrieval import (
    Candidate,
    RankingResult,
    RetrievalStage,
    ScoreRange,
    StageDiagnostic,
    StageStatus,
)

ScoreName = Literal["dense", "sparse_learned", "sparse_bm25", "rrf", "rerank"]
COPY_SUFFIX = re.compile(r"(?:_副本| \(副本\)|_copy| \(copy\)| \([0-9]+\))$", re.IGNORECASE)


def canonical_source(source: str) -> str:
    """Collapse copy suffixes on filenames only; preserve directories and formal versions."""
    path = PurePosixPath(source)
    stem = path.stem
    while (normalized := COPY_SUFFIX.sub("", stem)) != stem:
        stem = normalized
    return str(path.with_name((stem + path.suffix).casefold()))


def score(candidate: Candidate) -> float:
    """Never silently reinterpret fusion scores as cross-encoder relevance."""
    if candidate.scores.rerank is None:
        raise RetrievalConfigurationError(reason="missing_rerank_score")
    return candidate.scores.rerank


def source_key(candidate: Candidate) -> str:
    """Missing provenance is a registry failure, not one shared empty source group."""
    if candidate.source_path is None:
        raise IngestionRegistryError()
    return canonical_source(candidate.source_path)


def diagnostic(
    stage: RetrievalStage,
    input_count: int,
    candidates: list[Candidate],
    *,
    elapsed_ms: int = 0,
    status: StageStatus = StageStatus.COMPLETED,
) -> StageDiagnostic:
    """Summarize each native score scale independently; omit unmeasured scores."""
    ranges: dict[ScoreName, ScoreRange] = {}
    names: tuple[ScoreName, ...] = ("dense", "sparse_learned", "sparse_bm25", "rrf", "rerank")
    for name in names:
        values = [value for item in candidates if (value := getattr(item.scores, name)) is not None]
        if values:
            ranges[name] = ScoreRange(minimum=min(values), maximum=max(values))
    return StageDiagnostic(
        stage=stage, status=status, input_count=input_count, output_count=len(candidates),
        elapsed_ms=elapsed_ms, score_ranges=ranges,
    )


def diversify(candidates: list[Candidate], config: FilterConfig, top: float | None) -> list[Candidate]:
    """Preserve global stable ordering while enforcing per-canonical-source ceilings."""
    keys = [source_key(item) for item in candidates]
    high = Counter(
        key for key, item in zip(keys, candidates, strict=True)
        if top is not None and score(item) >= top * config.high_ratio
    )
    seen: Counter[str] = Counter()
    output = []
    for key, item in zip(keys, candidates, strict=True):
        cap = config.max_per_doc if top is None else min(
            max(high[key], config.min_per_doc), config.max_per_doc
        )
        # RagMate/backend/core/retriever.py:277-282: dominance "boost" is a no-op
        # for limit <= k and bypasses MAX_PER_SOURCE. Always enforce the hard cap.
        if seen[key] < cap:
            output.append(item)
            seen[key] += 1
    return output


def truncate(candidates: list[Candidate], config: FilterConfig) -> list[Candidate]:
    """A strict score cliff may stop before final_k; no quota is filled afterward."""
    output: list[Candidate] = []
    for item in candidates:
        if len(output) >= config.final_k:
            break
        if output and score(output[-1]) - score(item) > config.gap_threshold:
            break
        output.append(item)
    return output


def filter_ranked(candidates: list[Candidate], config: FilterConfig) -> RankingResult:
    """Run threshold, diversity and cliff stages without mutating caller-owned candidates."""
    ordered = sorted(candidates, key=score, reverse=True)
    top = score(ordered[0]) if ordered else None
    started = time.monotonic()
    threshold = max(top * config.dynamic_ratio, config.absolute_floor) if top is not None else None
    kept = [item for item in ordered if threshold is not None and score(item) >= threshold]
    stages = [diagnostic(RetrievalStage.THRESHOLD, len(ordered), kept,
                         elapsed_ms=int((time.monotonic() - started) * 1000))]
    started = time.monotonic()
    diversified = diversify(kept, config, top)
    stages.append(diagnostic(RetrievalStage.DIVERSITY, len(kept), diversified,
                             elapsed_ms=int((time.monotonic() - started) * 1000)))
    started = time.monotonic()
    output = truncate(diversified, config)
    stages.append(diagnostic(RetrievalStage.TRUNCATION, len(diversified), output,
                             elapsed_ms=int((time.monotonic() - started) * 1000)))
    return RankingResult(
        candidates=output, reranked=bool(ordered), top_rerank_score=top,
        meets_floor=top is not None and top >= config.absolute_floor, stages=stages,
    )


def filter_candidates(candidates: list[Candidate], config: FilterConfig) -> list[Candidate]:
    """Return the filtered set for callers that do not need stage diagnostics."""
    return filter_ranked(candidates, config).candidates


def fallback(candidates: list[Candidate], config: FilterConfig) -> RankingResult:
    """Bound unscored results by source and total count, without score-based filtering."""
    started = time.monotonic()
    diversified = diversify(candidates, config, None)
    diversity_ms = int((time.monotonic() - started) * 1000)
    output = diversified[:config.final_k]
    return RankingResult(candidates=output, stages=[
        diagnostic(RetrievalStage.THRESHOLD, len(candidates), candidates, status=StageStatus.DEGRADED),
        diagnostic(RetrievalStage.DIVERSITY, len(candidates), diversified, elapsed_ms=diversity_ms),
        diagnostic(RetrievalStage.TRUNCATION, len(diversified), output),
    ])
