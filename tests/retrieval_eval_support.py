"""Scripted Suite B artifacts; no model or storage work is performed."""

from datetime import date
from uuid import NAMESPACE_URL, uuid5

from app.retrieval.filtering import filter_ranked
from app.schemas.ingestion import canonical, digest
from app.schemas.model_runtime import RerankResult
from app.schemas.retrieval import (
    Candidate, EncodedQuery, ObservedRetrieval, PointTimeScope, RetrievalQuery,
    RetrievalResult, RetrievalScores, RetrievalTimings, RetrievalTrace,
)
from evals.harness.retrieval_contracts import (
    Arm, Attempt, ChunkReview, Dataset, DatasetManifest, Judgment, Judgments,
    Measurements, RetrievalCase, Split, arm_config,
)
from tests.ingestion_support import model_metadata
from scripts.model_evidence import Provenance

SHA = "a" * 40


def candidate_pool() -> list[Candidate]:
    return [
        Candidate(
            chunk_uuid=uuid5(NAMESPACE_URL, f"eval/{index}"),
            document_id=uuid5(NAMESPACE_URL, f"document/{index}"),
            document_version="a" * 64, chunking_version="b" * 64,
            content_sha256=digest(f"Evidence {index}"), content=f"Evidence {index}",
            parent_content=f"Parent evidence {index}", heading_path="Rule",
            source_path=f"policy-{index}.md", doc_type="policy",
            milvus_pk=index, effective_from=None, effective_to=None,
            scores=RetrievalScores(dense=0.8, rrf=0.1),
        )
        for index in range(8)
    ]


def dataset(split: Split = Split.DEVELOPMENT) -> Dataset:
    pool = candidate_pool()
    cases = [RetrievalCase(
        id="r-001", text="Which rule applies?", time_scope=PointTimeScope(as_of=date(2026, 9, 1)),
        semantic_group_id="one", split=split, category="policy", notes="Synthetic fixture",
    )]
    labels = [Judgments(query_id="r-001", judgments=[
        Judgment(chunk_id=item.chunk_uuid, grade=3 if index < 2 else 0, why="Scripted test only")
        for index, item in enumerate(pool)
    ])]
    manifest = DatasetManifest(
        corpus_version="c" * 64, chunking_version="b" * 64,
        queries_sha256="d" * 64, judgments_sha256={item: "e" * 64 for item in Split},
        chunk_ids=[item.chunk_uuid for item in pool],
        chunks=[ChunkReview(
            chunk_id=item.chunk_uuid, source_path=item.source_path,
            content_sha256=item.content_sha256, effective_from=None, effective_to=None,
        ) for item in pool],
    )
    return Dataset(
        manifest=manifest, cases=cases, labels=labels,
        identity=digest(canonical(manifest.model_dump(mode="json"))),
    )


def attempt(arm: Arm) -> Attempt:
    case = dataset().cases[0]
    config = arm_config(arm)
    admitted = candidate_pool()
    if arm is Arm.A:
        admitted.reverse()
    ranked = [item.model_copy(deep=True) for item in admitted]
    metadata = model_metadata()
    response = None
    filtered = ranked
    reranked, top, meets = False, None, None
    if config.use_rerank:
        response = RerankResult(
            request_id="scripted-eval", ms=10, queue_ms=2, inference_ms=8,
            metadata=metadata.model_copy(update={"precision": "fp32"}) if arm is Arm.D_FP32 else metadata,
            scores=[0.9, 0.8, *([0.1] * 6)],
        )
        for item, score in zip(ranked, response.scores, strict=True):
            item.scores.rerank = score
        result = filter_ranked(ranked, config.filtering)
        filtered = result.candidates
        reranked, top, meets = result.reranked, result.top_rerank_score, result.meets_floor
    return Attempt(query_id=case.id, arm=arm, observed=ObservedRetrieval(
        trace=RetrievalTrace(
            admitted=admitted, ranked=ranked, rerank_response=response,
            encoded=EncodedQuery(text=case.text, dense=[1.0, *([0.0] * 1023)], sparse={1: 1.0}, metadata=metadata),
        ),
        result=RetrievalResult(
            query=RetrievalQuery(standalone=case.text, time_scope=case.time_scope),
            corpus_version="c" * 64, candidates=filtered, retrieval_config=config,
            model_metadata=metadata, timings=RetrievalTimings(encode_ms=10, search_ms=20, admission_ms=1, rerank_ms=10, filter_ms=1, total_ms=42),
            reranked=reranked, top_rerank_score=top, meets_floor=meets,
            rerank_metadata=response.metadata if response else None,
        ),
    ))


def measurements(*, control: bool = True, split: Split = Split.DEVELOPMENT) -> Measurements:
    return Measurements(
        client_sha=SHA, source_dirty=False, dataset_identity=dataset(split).identity,
        corpus_version="c" * 64, split=split,
        server=Provenance(source_sha=SHA, image_id="sha256:" + "a" * 64, gpu_uuid="GPU-00000000-0000-0000-0000-000000000000"),
        attempts=[attempt(arm) for arm in Arm if control or arm is not Arm.D_FP32],
    )
