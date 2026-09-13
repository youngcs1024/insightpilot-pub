"""Typed corpus fixtures; no source reading or infrastructure work at import time."""

from datetime import date
from pathlib import Path

import yaml

from app.schemas.corpus import (
    CorpusDocument,
    CorpusEntry,
    CorpusFormat,
    CorpusMetadata,
    DocumentType,
)
from app.schemas.metrics import MetricDefinition
from data.corpus_metrics import metric_contract_text


def metadata_source() -> str:
    """Return minimal business metadata without depending on the shipped inventory."""
    return (
        "title: 测试规则\ndoc_type: policy\neffective_from: 2026-01-01\n"
        "effective_to: null\nsupersedes: null\n"
    )


def markdown_source(body: str | None = None) -> str:
    """Build a source with visible metadata and a nonempty body."""
    content = body if body is not None else "# 测试规则\n\n退款申请需核验商品行。\n"
    return "---\n" + metadata_source() + "---\n\n" + content


def write_inventory(root: Path, entries: list[CorpusEntry]) -> None:
    """Write only the explicit temporary manifest; never delete fixture files."""
    value = {"schema_version": 1, "documents": [entry.model_dump(mode="json") for entry in entries]}
    (root / "MANIFEST.yaml").write_text(yaml.safe_dump(value), encoding="utf-8")


def entry(name: str = "rule.md") -> CorpusEntry:
    """Make a canonical entry for one of the three resource formats."""
    format_value = CorpusFormat(Path(name).suffix[1:])
    return CorpusEntry(
        path=name,
        format=format_value,
        metadata_path=None if format_value is CorpusFormat.MARKDOWN else name + ".meta.yaml",
    )


def memo(definition: MetricDefinition) -> CorpusDocument:
    """Build a detached memo for fault injection into its actual normative body."""
    return CorpusDocument(
        entry=entry(definition.key + ".md"),
        metadata=CorpusMetadata(
            title="指标说明",
            doc_type=DocumentType.METRIC_MEMO,
            effective_from=date(2025, 3, 1),
            effective_to=None,
            supersedes=None,
            metric_key=definition.key,
        ),
        text="## 规范定义\n\n" + metric_contract_text(definition) + "\n\n## 使用说明\n\n业务说明。",
    )
