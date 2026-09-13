"""Pure arm construction, safe temporal predicates and native score bookkeeping."""

from datetime import date, timedelta
from enum import StrEnum

from pydantic import BaseModel, Field, FiniteFloat, TypeAdapter, ValidationError
from pymilvus import AnnSearchRequest  # type: ignore[import-untyped]

from app.core.errors import RetrievalConfigurationError, RetrievalUnavailableError
from app.retrieval.config import RetrievalConfig
from app.schemas.retrieval import (
    Candidate,
    EncodedQuery,
    KnowledgeTimeScope,
    PointTimeScope,
    RetrievalScores,
)

EPOCH = date(1970, 1, 1)
OUTPUT_FIELDS = [
    "chunk_uuid", "document_id", "doc_type", "heading_path", "content", "parent_content",
    "effective_from", "effective_to", "document_version", "chunking_version", "content_sha256",
]


class SearchArm(StrEnum):
    """The field names are fixed protocol identities, never user expressions."""

    DENSE = "dense"
    LEARNED = "sparse_learned"
    BM25 = "sparse_bm25"


def enabled_arms(config: RetrievalConfig) -> list[SearchArm]:
    """Keep the same deterministic arm order for requests and diagnostics."""
    return [
        arm for arm, enabled in (
            (SearchArm.DENSE, config.use_dense),
            (SearchArm.LEARNED, config.use_sparse_learned),
            (SearchArm.BM25, config.use_bm25),
        ) if enabled
    ]


def time_filter(scope: KnowledgeTimeScope) -> str:
    """Build expressions exclusively from validated dates, retaining half-open bounds."""
    if isinstance(scope, PointTimeScope):
        day = (scope.as_of - EPOCH).days
        return (
            f"(effective_from == -1 or effective_from <= {day}) and "
            f"(effective_to == -1 or effective_to > {day})"
        )
    merged: list[tuple[date, date]] = []
    for period in sorted(scope.periods, key=lambda item: (item.start, item.end)):
        if merged and period.start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], period.end))
        else:
            merged.append((period.start, period.end))
    return " or ".join(
        f"((effective_from == -1 or effective_from < {(end - EPOCH).days}) and "
        f"(effective_to == -1 or effective_to > {(start - EPOCH).days}))"
        for start, end in merged
    )


def arm_request(
    arm: SearchArm, query: EncodedQuery, config: RetrievalConfig, expression: str
) -> AnnSearchRequest:
    """Keep SDK dictionaries inside the SDK adapter boundary."""
    if arm is SearchArm.DENSE:
        if query.dense is None:
            raise RetrievalConfigurationError(reason="missing_dense_query")
        return AnnSearchRequest(
            data=[query.dense], anns_field=arm.value,
            param={"metric_type": "IP", "params": {"ef": config.hnsw_ef}},
            limit=config.pool, expr=expression,
        )
    if arm is SearchArm.LEARNED:
        if query.sparse is None:
            raise RetrievalConfigurationError(reason="missing_sparse_query")
        return AnnSearchRequest(
            data=[query.sparse], anns_field=arm.value,
            param={"metric_type": "IP", "params": {"drop_ratio_search": config.drop_ratio_search}},
            limit=config.pool, expr=expression,
        )
    return AnnSearchRequest(
        data=[query.text], anns_field=arm.value, param={"metric_type": "BM25"},
        limit=config.pool, expr=expression,
    )


class SearchEntity(BaseModel):
    """Normalize only the two Milvus epoch-day fields before candidate validation."""

    chunk_uuid: str
    document_id: str
    document_version: str
    chunking_version: str
    content_sha256: str
    doc_type: str
    heading_path: str
    content: str
    parent_content: str
    effective_from: int = Field(strict=True)
    effective_to: int = Field(strict=True)


class SearchHit(BaseModel):
    """The pinned SDK returns pk rather than id for this collection."""

    pk: int = Field(strict=True, ge=0)
    distance: FiniteFloat = Field(strict=True)
    entity: SearchEntity


def candidates(raw: object, arm: SearchArm | None) -> list[Candidate]:
    """Reject malformed batches without leaking opaque SDK payloads to callers."""
    try:
        batches = TypeAdapter(list[list[SearchHit]]).validate_python(raw)
        if len(batches) != 1:
            raise RetrievalUnavailableError(reason="search_result_count")
        result = []
        for hit in batches[0]:
            entity = hit.entity.model_dump()
            for field in ("effective_from", "effective_to"):
                day = getattr(hit.entity, field)
                entity[field] = None if day == -1 else EPOCH + timedelta(days=day)
            result.append(Candidate(
                **entity, milvus_pk=hit.pk,
                scores=RetrievalScores.model_validate({arm.value if arm else "rrf": hit.distance}),
            ))
        return result
    except (ValidationError, OverflowError) as exc:
        raise RetrievalUnavailableError(reason="invalid_search_response") from exc


def attach_scores(fused: list[Candidate], diagnostic: list[Candidate], arm: SearchArm) -> None:
    """A physical hit cannot borrow the score of a different version or duplicate row."""
    indexed = {
        (item.milvus_pk, item.chunk_uuid, item.document_id, item.document_version,
         item.chunking_version, item.content_sha256): getattr(item.scores, arm.value)
        for item in diagnostic
    }
    for item in fused:
        score = indexed.get((
            item.milvus_pk, item.chunk_uuid, item.document_id, item.document_version,
            item.chunking_version, item.content_sha256,
        ))
        setattr(item.scores, arm.value, score)
