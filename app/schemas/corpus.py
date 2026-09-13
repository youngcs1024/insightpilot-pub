"""Authoring contracts only; persistent ingestion identities belong to Step 3.4."""

from datetime import date
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

from app.schemas.mcp import Contract


class DocumentType(StrEnum):
    """The six business document families in the retrieval corpus."""

    POLICY = "policy"
    PROMO_RULE = "promo_rule"
    METRIC_MEMO = "metric_memo"
    SOP = "sop"
    REGION_RULE = "region_rule"
    ANALYSIS_NOTE = "analysis_note"


class CorpusFormat(StrEnum):
    """Supported source formats; sidecars are not source documents."""

    MARKDOWN = "md"
    EXCEL = "xlsx"
    PDF = "pdf"


def relative_source(value: str) -> str:
    """Require one canonical POSIX path beneath the corpus root."""
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or ":" in value
        or str(path) != value
        or value == "."
    ):
        raise PydanticCustomError("corpus_path", "Expected a canonical relative source path")
    return value


class CorpusMetadata(Contract):
    """Business dates are Shanghai calendar dates with an exclusive upper bound."""

    title: str = Field(min_length=1, max_length=200)
    doc_type: DocumentType
    effective_from: date
    effective_to: date | None
    supersedes: str | None
    metric_key: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_]*$")

    @field_validator("supersedes")
    @classmethod
    def predecessor_path(cls, value: str | None) -> str | None:
        """Use the same root-relative identity for predecessor references."""
        return relative_source(value) if value is not None else None

    @model_validator(mode="after")
    def business_interval(self) -> Self:
        """Reject empty validity and metric annotations on non-metric documents."""
        if self.effective_to is not None and self.effective_from >= self.effective_to:
            raise PydanticCustomError("corpus_interval", "Validity must be nonempty")
        if self.metric_key is not None and self.doc_type is not DocumentType.METRIC_MEMO:
            raise PydanticCustomError("corpus_metric", "Metric keys require a metric memo")
        return self


class CorpusEntry(Contract):
    """File inventory; business metadata has exactly one source per document."""

    path: str
    format: CorpusFormat
    metadata_path: str | None
    negative_control: bool = Field(default=False, strict=True)

    @field_validator("path", "metadata_path")
    @classmethod
    def safe_path(cls, value: str | None) -> str | None:
        """Disallow ambiguous and escaping inventory paths."""
        return relative_source(value) if value is not None else None

    @model_validator(mode="after")
    def format_matches_path(self) -> Self:
        """Markdown owns front matter; binary sources own same-name YAML sidecars."""
        if PurePosixPath(self.path).suffix != "." + self.format.value:
            raise PydanticCustomError("corpus_format", "Format does not match source extension")
        expected = None if self.format is CorpusFormat.MARKDOWN else self.path + ".meta.yaml"
        if self.metadata_path != expected:
            raise PydanticCustomError("corpus_sidecar", "Unexpected metadata location")
        return self


class CorpusManifest(Contract):
    """Static source inventory, distinct from the future committed ingestion manifest."""

    schema_version: Literal[1] = 1
    documents: list[CorpusEntry] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def unique_paths(self) -> Self:
        """Never silently collapse duplicate source entries."""
        paths = [entry.path for entry in self.documents]
        if len(paths) != len(set(paths)):
            raise PydanticCustomError("corpus_duplicate", "Duplicate corpus source")
        return self


class CorpusDocument(Contract):
    """Detached extracted text used only for authoring checks and statistics."""

    entry: CorpusEntry
    metadata: CorpusMetadata
    text: str = Field(min_length=1)


class DocumentStatistics(Contract):
    """Length counts extracted Unicode characters, excluding metadata and outer space."""

    path: str
    doc_type: DocumentType
    format: CorpusFormat
    char_count: int = Field(gt=0)


class CorpusStatistics(Contract):
    """Machine-readable summary from one complete, validated source inventory."""

    schema_version: Literal[1] = 1
    total: int = Field(gt=0)
    by_type: dict[DocumentType, int]
    by_format: dict[CorpusFormat, int]
    min_chars: int = Field(gt=0)
    max_chars: int = Field(gt=0)
    mean_chars: float = Field(gt=0)
    documents: list[DocumentStatistics]
