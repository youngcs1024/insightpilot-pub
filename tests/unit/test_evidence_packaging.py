"""Whole-context selection, immutable snapshots and exact rendered-token bounds."""
# ruff: noqa: PLR2004 -- explicit budgets and scores are the acceptance expectations.

from datetime import date
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.agents.knowledge.nodes.package_evidence import package_evidence as package_node
from app.core.errors import DeadlineExceededError, KnowledgeEvidenceError
from app.retrieval.config import EvidenceConfig
from app.retrieval.evidence import package_evidence, render_documents
from app.schemas.ingestion import digest
from app.schemas.knowledge import KnowledgeEvidence, TextSelection
from app.schemas.model_runtime import ModelFailureKind
from app.schemas.retrieval import PolicyPeriod, RangeTimeScope, RetrievalScores
from app.services.schema_tokens import SchemaTokenCounter
from tests.ingestion_support import model_metadata
from tests.knowledge_support import dated_candidate, retrieval
from tests.retrieval_support import candidate, deadline


@pytest.fixture
def counter() -> SchemaTokenCounter:
    return SchemaTokenCounter()


def test_all_stage_scores_present(counter: SchemaTokenCounter) -> None:
    result = retrieval()
    result.candidates[0].scores = RetrievalScores(
        dense=0.4,
        sparse_learned=0.5,
        sparse_bm25=3.2,
        rrf=0.03,
        rerank=0.8,
    )
    result.reranked, result.top_rerank_score, result.meets_floor = True, 0.8, True
    result.model_metadata = model_metadata()
    result.rerank_metadata = model_metadata()
    evidence = package_evidence(result, EvidenceConfig(), counter)
    assert evidence.chunks[0].scores.model_dump() == result.candidates[0].scores.model_dump()
    assert evidence.reranked
    assert evidence.meets_floor
    assert evidence.top_rerank_score == 0.8
    assert evidence.model_metadata.model_dump() == result.model_metadata.model_dump()
    assert evidence.rerank_metadata.model_dump() == result.rerank_metadata.model_dump()


def test_config_snapshot_embedded(counter: SchemaTokenCounter) -> None:
    result = retrieval()
    config = EvidenceConfig(max_tokens=5900)
    evidence = package_evidence(result, config, counter)
    result.retrieval_config.filtering.final_k = 2
    config.max_tokens = 1
    assert evidence.retrieval_config.filtering.final_k == 8
    assert evidence.packaging_config.max_tokens == 5900


def test_chunk_ids_resolvable(counter: SchemaTokenCounter) -> None:
    result = retrieval([dated_candidate()])
    result.provenance[0] = result.provenance[0].model_copy(update={"page": 7})
    evidence = package_evidence(result, EvidenceConfig(), counter)
    chunk = evidence.chunks[0]
    source = result.provenance[0]
    assert chunk.chunk_id == source.chunk_uuid
    assert chunk.document_id == source.document_id
    assert chunk.document_title == source.document_title
    assert chunk.document_version == source.document_version
    assert chunk.chunking_version == source.chunking_version
    assert chunk.heading_path == source.heading_path
    assert chunk.source_path == source.source_path
    assert chunk.page == 7
    assert chunk.effective_from == date(2026, 7, 1)
    assert chunk.effective_to == date(2026, 8, 1)


def test_parent_context_selected_without_compression(counter: SchemaTokenCounter) -> None:
    result = retrieval()
    evidence = package_evidence(result, EvidenceConfig(), counter)
    chunk = evidence.chunks[0]
    assert chunk.original_text == chunk.generation_text == result.candidates[0].parent_content
    assert chunk.original_text != result.candidates[0].content
    assert not chunk.compressed
    assert not evidence.compressed
    assert chunk.text_selection is TextSelection.PARENT
    assert evidence.generation_tokens == counter.count(evidence.generation_block)
    assert evidence.generation_block == render_documents(evidence.chunks)


def test_parent_too_large_falls_back_to_whole_child(counter: SchemaTokenCounter) -> None:
    result = retrieval()
    result.candidates[0].parent_content = "中文父文档。" * 5000
    evidence = package_evidence(result, EvidenceConfig(), counter)
    assert evidence.chunks[0].text_selection is TextSelection.CHILD
    assert evidence.chunks[0].generation_text == result.candidates[0].content
    assert evidence.decisions[0].selection is TextSelection.CHILD


def test_oversized_candidate_skipped_but_later_candidate_considered(
    counter: SchemaTokenCounter,
) -> None:
    large, small = candidate(), candidate()
    large.content = "中文子片段。" * 4000
    large.content_sha256 = digest(large.content)
    large.parent_content = large.content
    result = retrieval([large, small])
    evidence = package_evidence(result, EvidenceConfig(), counter)
    assert [chunk.chunk_id for chunk in evidence.chunks] == [small.chunk_uuid]
    assert [item.selection for item in evidence.decisions] == [
        TextSelection.OMITTED_BUDGET,
        TextSelection.PARENT,
    ]
    assert evidence.generation_tokens <= 6000


def test_exact_budget_includes_delimiters_and_metadata(counter: SchemaTokenCounter) -> None:
    result = retrieval()
    result.candidates[0].parent_content = result.candidates[0].content
    full = package_evidence(result, EvidenceConfig(), counter)
    exact = EvidenceConfig(max_tokens=full.generation_tokens)
    assert package_evidence(result, exact, counter).chunks
    tight = EvidenceConfig(max_tokens=full.generation_tokens - 1)
    empty = package_evidence(result, tight, counter)
    assert not empty.chunks
    assert empty.generation_block == ""
    assert empty.generation_tokens == 0
    assert full.generation_tokens > counter.count(result.candidates[0].content)


def test_empty_and_below_floor_results(counter: SchemaTokenCounter) -> None:
    result = retrieval([])
    result.corpus_version = None
    empty = package_evidence(result, EvidenceConfig(), counter)
    assert empty.chunks == empty.decisions == ()
    assert empty.corpus_version is None
    result.reranked, result.top_rerank_score, result.meets_floor = True, 0.1, False
    below = package_evidence(result, EvidenceConfig(), counter)
    assert not below.chunks
    assert below.meets_floor is False
    assert below.top_rerank_score == 0.1


@pytest.mark.parametrize("degradation", [None, ModelFailureKind.UNAVAILABLE])
def test_unknown_scores_and_degradation_remain_unknown(
    counter: SchemaTokenCounter, degradation: ModelFailureKind | None
) -> None:
    result = retrieval()
    result.degradation = degradation
    evidence = package_evidence(result, EvidenceConfig(), counter)
    assert evidence.degradation == degradation
    assert evidence.meets_floor is None
    assert evidence.top_rerank_score is None
    assert evidence.chunks[0].scores.rerank is None
    assert not evidence.reranked
    assert evidence.chunks[0].scores.sparse_bm25 is None


@pytest.mark.parametrize(
    "damage", ["missing", "duplicate", "version", "title", "hash", "corpus", "floor"]
)
def test_inconsistent_provenance_is_typed_failure(counter: SchemaTokenCounter, damage: str) -> None:
    result = retrieval()
    if damage == "missing":
        result.provenance = []
    elif damage == "duplicate":
        result.provenance *= 2
    elif damage == "version":
        result.candidates[0].document_version = "f" * 64
    elif damage == "title":
        result.provenance[0] = result.provenance[0].model_copy(update={"document_title": ""})
    elif damage == "hash":
        result.candidates[0].content = "tampered"
    elif damage == "corpus":
        result.corpus_version = None
    else:
        result.meets_floor = False
    with pytest.raises(KnowledgeEvidenceError):
        package_evidence(result, EvidenceConfig(), counter)


def test_snapshot_survives_input_mutation_and_json_roundtrip(counter: SchemaTokenCounter) -> None:
    result = retrieval()
    result.query.time_scope = RangeTimeScope(
        periods=[PolicyPeriod(start=date(2026, 7, 1), end=date(2026, 8, 1), label="七月")]
    )
    result.query.assumptions.append("比较七月政策")
    evidence = package_evidence(result, EvidenceConfig(), counter)
    serialized = evidence.model_dump_json()
    result.candidates[0].parent_content = "replacement"
    result.candidates[0].scores.dense = 999
    result.query.time_scope.periods[0].label = "new label"
    result.query.assumptions.clear()
    result.provenance.clear()
    result.retrieval_config.filtering.final_k = 2
    restored = KnowledgeEvidence.model_validate_json(serialized)
    assert evidence.model_dump_json() == serialized == restored.model_dump_json()
    assert restored.generation_block == evidence.generation_block
    with pytest.raises(ValidationError):
        restored.time_scope.periods[0].label = "changed"
    with pytest.raises(ValidationError):
        restored.time_scope.periods = ()


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("self", "generation_block", "changed"),
        ("chunk", "original_text", "changed"),
        ("score", "dense", 1.0),
        ("filter", "final_k", 1),
        ("config", "use_dense", False),
        ("scope", "as_of", date(2020, 1, 1)),
        ("budget", "max_tokens", 1),
        ("timing", "total_ms", 1),
        ("model", "embed_batch", 1),
    ],
)
def test_every_snapshot_layer_is_frozen(
    counter: SchemaTokenCounter, target: str, field: str, value: object
) -> None:
    result = retrieval()
    result.model_metadata = model_metadata()
    evidence = package_evidence(result, EvidenceConfig(), counter)
    targets = {
        "self": evidence,
        "chunk": evidence.chunks[0],
        "score": evidence.chunks[0].scores,
        "filter": evidence.retrieval_config.filtering,
        "config": evidence.retrieval_config,
        "scope": evidence.time_scope,
        "budget": evidence.packaging_config,
        "timing": evidence.timings,
        "model": evidence.model_metadata,
    }
    with pytest.raises(ValidationError):
        setattr(targets[target], field, value)
    assert isinstance(evidence.chunks, tuple)
    assert isinstance(evidence.decisions, tuple)


@pytest.mark.parametrize("budget", [0, 6001])
def test_budget_settings_are_bounded(budget: int) -> None:
    with pytest.raises(ValidationError):
        EvidenceConfig(max_tokens=budget)


def test_node_uses_injected_resources_and_deadline(counter: SchemaTokenCounter) -> None:
    context = SimpleNamespace(
        deadline=deadline(),
        schema_token_counter=counter,
        settings=SimpleNamespace(retrieval=SimpleNamespace(evidence=EvidenceConfig())),
    )
    assert package_node(retrieval(), context).chunks
    context.deadline = deadline(-1)
    with pytest.raises(DeadlineExceededError):
        package_node(retrieval(), context)
