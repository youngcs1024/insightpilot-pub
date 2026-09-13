"""Authored dataset and split isolation checks use source files, never GPU inference."""

from pathlib import Path

import pytest

from app.schemas.ingestion import digest
from app.services.ingestion_config import IngestionSettings
from app.services.ingestion_plan import stage
from evals.harness.contracts import EvaluationError
from evals.harness.retrieval_contracts import ROOT, Dataset, Queries, Split
from evals.harness.retrieval_dataset import load_dataset, validate_labels, validate_queries


@pytest.fixture(scope="module")
def development() -> Dataset:
    return load_dataset(Split.DEVELOPMENT)


@pytest.fixture(scope="module")
def frozen() -> Dataset:
    return load_dataset(Split.FROZEN)


def test_no_semantic_group_overlap(development: Dataset, frozen: Dataset) -> None:
    assert len(development.cases) == len(frozen.cases) == 15
    assert not {case.semantic_group_id for case in development.cases} & {case.semantic_group_id for case in frozen.cases}
    assert sum(not case.answerable for case in development.cases) == 2
    assert sum(not case.answerable for case in frozen.cases) == 2


def test_all_judgments_reference_actual_original_chunks(development: Dataset, frozen: Dataset) -> None:
    prepared = stage(ROOT.parents[2] / "data/corpus", [], IngestionSettings())
    assert not prepared.failed
    chunks = {chunk.chunk_uuid: chunk for document in prepared.changed for chunk in document.chunks}
    assert set(chunks) == set(development.manifest.chunk_ids)
    for dataset in (development, frozen):
        for record in dataset.labels:
            assert len(record.judgments) == len(chunks) == 191
            assert all(item.why for item in record.judgments)
        assert all(item.content_sha256 == digest(chunks[item.chunk_id].content) for item in dataset.manifest.chunks)


def test_expired_policy_grade_zero(development: Dataset) -> None:
    metadata = {item.chunk_id: item for item in development.manifest.chunks}
    july = next(item for item in development.labels if item.query_id == "r-001")
    current = next(item for item in development.labels if item.query_id == "r-002")
    assert all(item.grade == 0 for item in july.judgments if metadata[item.chunk_id].source_path == "refund_policy_v3.md")
    assert all(item.grade == 0 for item in current.judgments if metadata[item.chunk_id].source_path == "refund_policy_v2.md")


def test_same_policy_valid_for_historical_question(development: Dataset) -> None:
    paths = {item.chunk_id: item.source_path for item in development.manifest.chunks}
    july = next(item for item in development.labels if item.query_id == "r-001")
    assert any(item.grade >= 2 and paths[item.chunk_id] == "refund_policy_v2.md" for item in july.judgments)
    damaged = development.model_copy(deep=True)
    current = next(item for item in damaged.labels if item.query_id == "r-002")
    next(item for item in current.judgments if paths[item.chunk_id] == "refund_policy_v2.md").grade = 3
    with pytest.raises(EvaluationError, match="Expired"):
        validate_labels(damaged)


def test_tuning_does_not_read_frozen_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    read = Path.read_text
    def guarded(path: Path, *args: object, **kwargs: object) -> str:
        assert path.name != "frozen.yaml", "Tuner opened frozen labels"
        return read(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", guarded)
    assert load_dataset(Split.DEVELOPMENT).cases


def test_query_group_and_judgment_corruption_fails(development: Dataset, frozen: Dataset) -> None:
    cases = [case.model_copy(deep=True) for case in development.cases + frozen.cases]
    cases[-1].semantic_group_id = cases[0].semantic_group_id
    with pytest.raises(EvaluationError, match="leakage"):
        validate_queries(Queries(queries=cases))
    damaged = development.model_copy(deep=True)
    damaged.labels[0].judgments[1].chunk_id = damaged.labels[0].judgments[0].chunk_id
    with pytest.raises(EvaluationError):
        validate_labels(damaged)


def test_manifest_hash_mismatch_fails_before_eval(tmp_path: Path) -> None:
    (tmp_path / "judgments.yaml").write_text((ROOT / "judgments.yaml").read_text())
    (tmp_path / "queries.yaml").write_text((ROOT / "queries.yaml").read_text() + "\n")
    with pytest.raises(EvaluationError, match="hash mismatch"):
        load_dataset(Split.DEVELOPMENT, tmp_path)
