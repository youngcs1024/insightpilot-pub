"""Hash-bound Suite B loading; frozen labels are never opened by development loading."""

from pathlib import Path
from uuid import UUID

from app.schemas.ingestion import ActiveManifest, canonical, digest
from app.schemas.retrieval import KnowledgeTimeScope, PointTimeScope
from app.services.corpus_sources import parse_yaml
from evals.harness.contracts import EvaluationError
from evals.harness.ir_metrics import RELEVANT
from evals.harness.retrieval_contracts import (
    ROOT,
    ChunkReview,
    Dataset,
    DatasetManifest,
    JudgmentPartition,
    Queries,
    Split,
)


def applicable(chunk: ChunkReview, scope: KnowledgeTimeScope) -> bool:
    """Use the same half-open business-date predicates as the retrieval contract."""
    if isinstance(scope, PointTimeScope):
        return (chunk.effective_from is None or chunk.effective_from <= scope.as_of) and (
            chunk.effective_to is None or scope.as_of < chunk.effective_to
        )
    return any(
        (chunk.effective_from is None or chunk.effective_from < period.end)
        and (chunk.effective_to is None or chunk.effective_to > period.start)
        for period in scope.periods
    )


def dataset_identity(manifest: DatasetManifest) -> str:
    """Include both partition hashes, without reading their contents."""
    return digest(canonical(manifest.model_dump(mode="json")))


def checked_text(path: Path, expected: str) -> str:
    """Never silently refresh the expected hash from an edited resource."""
    text = path.read_text(encoding="utf-8")
    if digest(text) != expected:
        raise EvaluationError("Dataset resource hash mismatch")
    return text


def load_dataset(split: Split, root: Path = ROOT) -> Dataset:
    """Validate all group assignments, then open only the requested judgment partition."""
    manifest = parse_yaml((root / "judgments.yaml").read_text(), DatasetManifest)
    queries = parse_yaml(checked_text(root / "queries.yaml", manifest.queries_sha256), Queries)
    validate_queries(queries)
    if set(manifest.judgments_sha256) != set(Split):
        raise EvaluationError("Missing partition identity")
    partition = parse_yaml(
        checked_text(root / f"{split.value}.yaml", manifest.judgments_sha256[split]),
        JudgmentPartition,
    )
    if partition.split != split:
        raise EvaluationError("Wrong judgment partition")
    dataset = Dataset(
        manifest=manifest,
        cases=[case for case in queries.queries if case.split == split],
        labels=partition.records,
        identity=dataset_identity(manifest),
    )
    validate_labels(dataset)
    return dataset


def validate_queries(queries: Queries) -> None:
    """All paraphrases and historical variants stay in the same semantic partition."""
    ids = [case.id for case in queries.queries]
    groups: dict[str, Split] = {}
    if len(ids) != len(set(ids)) or not ids:
        raise EvaluationError("Duplicate or missing queries")
    for case in queries.queries:
        if groups.setdefault(case.semantic_group_id, case.split) != case.split:
            raise EvaluationError("Semantic group leakage")
    if {case.split for case in queries.queries} != set(Split):
        raise EvaluationError("Both query partitions are required")


def validate_labels(dataset: Dataset) -> None:
    """Identity, answerability and time applicability are ground-truth invariants."""
    chunks = {chunk.chunk_id: chunk for chunk in dataset.manifest.chunks}
    ids = dataset.manifest.chunk_ids
    if (
        len(ids) != len(set(ids))
        or set(ids) != set(chunks)
        or len(chunks) != len(dataset.manifest.chunks)
    ):
        raise EvaluationError("Invalid judged chunk inventory")
    records = {item.query_id: item for item in dataset.labels}
    if len(records) != len(dataset.labels) or set(records) != {case.id for case in dataset.cases}:
        raise EvaluationError("Missing or duplicate query judgments")
    for case in dataset.cases:
        labels = records[case.id].judgments
        keys = [item.chunk_id for item in labels]
        if len(keys) != len(set(keys)) or set(keys) - set(chunks):
            raise EvaluationError("Duplicate or unknown judged chunks")
        if case.answerable != any(item.grade >= RELEVANT for item in labels):
            raise EvaluationError("Judgments contradict answerability")
        if any(
            item.grade > 0 and not applicable(chunks[item.chunk_id], case.time_scope)
            for item in labels
        ):
            raise EvaluationError("Expired policy has positive grade")


def check_manifest(dataset: Dataset, active: ActiveManifest | None, chunk_ids: set[UUID]) -> None:
    """Fail before any model or search call when the deployed corpus differs."""
    if (
        active is None
        or active.corpus_version != dataset.manifest.corpus_version
        or set(active.splitter_configs) != {dataset.manifest.chunking_version}
        or set(dataset.manifest.chunk_ids) != chunk_ids
    ):
        raise EvaluationError("Active corpus manifest does not match dataset")
