"""Synchronous offline corpus inspection; no ingestion, stores or model calls."""

from collections import Counter
from pathlib import Path
from statistics import mean

from app.core.errors import CorpusValidationError
from app.schemas.corpus import CorpusDocument, CorpusManifest, CorpusStatistics, DocumentStatistics
from app.services.corpus_sources import (
    excel_text,
    markdown_parts,
    parse_yaml,
    pdf_text,
    read_document,
    read_utf8,
    source_path,
    table_row,
)

# Retain the authoring API while sharing its parser with ingestion.
__all__ = [
    "corpus_statistics",
    "excel_text",
    "load_corpus",
    "markdown_parts",
    "parse_yaml",
    "pdf_text",
    "read_document",
    "read_utf8",
    "source_path",
    "table_row",
]
CORPUS_ROOT = Path(__file__).resolve().parent / "corpus"
ADVERSARIAL_ROOT = CORPUS_ROOT / "adversarial"


def validate_predecessors(documents: list[CorpusDocument]) -> None:
    """Require same-type predecessors and continuous, strictly advancing validity."""
    by_path = {document.entry.path: document for document in documents}
    replaced: set[str] = set()
    for document in documents:
        name = document.metadata.supersedes
        if name is None:
            continue
        predecessor = by_path.get(name)
        if predecessor is None or name in replaced:
            raise CorpusValidationError("Missing or multiply replaced predecessor.", path=name)
        old, new = predecessor.metadata, document.metadata
        if (
            old.doc_type is not new.doc_type
            or old.effective_from >= new.effective_from
            or old.effective_to != new.effective_from
        ):
            raise CorpusValidationError("Invalid replacement validity chain.", path=name)
        replaced.add(name)


def load_corpus(root: Path = CORPUS_ROOT) -> list[CorpusDocument]:
    """Validate the complete inventory before returning detached authoring resources."""
    root = root.resolve()
    manifest = parse_yaml(read_utf8(source_path(root, "MANIFEST.yaml")), CorpusManifest)
    expected = {"MANIFEST.yaml"}
    for entry in manifest.documents:
        expected.add(entry.path)
        if entry.metadata_path is not None:
            expected.add(entry.metadata_path)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
        if root != CORPUS_ROOT.resolve() or not path.is_relative_to(ADVERSARIAL_ROOT)
    }
    if actual != expected:
        raise CorpusValidationError(
            "Corpus inventory mismatch.",
            missing=sorted(expected - actual),
            extra=sorted(actual - expected),
        )
    documents = [read_document(root, entry) for entry in manifest.documents]
    validate_predecessors(documents)
    return documents


def corpus_statistics(documents: list[CorpusDocument]) -> CorpusStatistics:
    """Summarize extracted bodies, never sidecar bytes or binary file sizes."""
    if not documents:
        raise CorpusValidationError("Cannot summarize an empty corpus.")
    records = [
        DocumentStatistics(
            path=document.entry.path,
            doc_type=document.metadata.doc_type,
            format=document.entry.format,
            char_count=len(document.text),
        )
        for document in documents
    ]
    lengths = [record.char_count for record in records]
    return CorpusStatistics(
        total=len(records),
        by_type=dict(Counter(record.doc_type for record in records)),
        by_format=dict(Counter(record.format for record in records)),
        min_chars=min(lengths),
        max_chars=max(lengths),
        mean_chars=mean(lengths),
        documents=records,
    )
