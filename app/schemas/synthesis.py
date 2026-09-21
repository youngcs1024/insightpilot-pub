"""Bounded model claims and typed locations in the committed model view."""

from enum import StrEnum
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from app.schemas.mcp import Contract, SqlValue


class ClaimKind(StrEnum):
    """Epistemic labels, independent of the wording of a claim."""

    FACT_DATA = "fact_data"
    FACT_DOCUMENT = "fact_document"
    INFERENCE = "inference"
    UNSUPPORTED = "unsupported"


class CellReference(Contract):
    """Zero-based position in generation_block.sample_rows, not the audit sample."""

    kind: Literal["cell"] = "cell"
    row: int = Field(ge=0, strict=True)
    column: int = Field(ge=0, strict=True)
    value: SqlValue


class StatisticReference(Contract):
    """A visible column statistic, computed over all returned rows."""

    kind: Literal["statistic"] = "statistic"
    column: int = Field(ge=0, strict=True)
    field: Literal["null_count", "distinct_count", "minimum", "maximum", "total"]
    value: SqlValue


class RowCountReference(Contract):
    """A successful empty query is itself usable evidence."""

    kind: Literal["row_count"] = "row_count"
    value: int = Field(ge=0, strict=True)


DataReference = Annotated[
    CellReference | StatisticReference | RowCountReference, Field(discriminator="kind")
]


LEGACY_CLAIM_CHARS = 2000


class Claim(Contract):
    """Facts carry resolvable evidence; inference and unsupported text stay labelled."""

    schema_version: Literal[1, 2] = 2
    text: str = Field(min_length=1, max_length=4000)
    kind: ClaimKind
    data_refs: list[DataReference] = Field(default_factory=list, max_length=20)
    chunk_ids: list[UUID] = Field(default_factory=list, max_length=20)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def released_text_bound(self) -> Self:
        """V2 accepts the existing knowledge-passage bound; V1 stays unchanged."""
        if self.schema_version == 1 and len(self.text) > LEGACY_CLAIM_CHARS:
            raise PydanticCustomError("claim_text", "Historical claim exceeds its text bound")
        return self


class Conflict(Contract):
    """A pair of zero-based claim indices; no model-written resolution field."""

    schema_version: Literal[1] = 1
    left_claim: int = Field(ge=0, strict=True)
    right_claim: int = Field(ge=0, strict=True)


class SynthesisOutput(Contract):
    """The model draft; its summary is always replaced after claim validation."""

    schema_version: Literal[1] = 1
    claims: list[Claim] = Field(default_factory=list, max_length=16)
    conflicts: list[Conflict] = Field(default_factory=list, max_length=16)
    unanswered: list[Annotated[str, Field(min_length=1, max_length=1000)]] = Field(
        default_factory=list, max_length=12
    )
    summary: str = Field(default="", max_length=32_000)


class SynthesisAbstention(StrEnum):
    """Evidence absence differs from an invalid model draft or an upstream failure."""

    NO_EVIDENCE = "no_evidence"
    UNSUPPORTED = "unsupported"
    INVALID_REFERENCES = "invalid_references"


class DataGenerationView(Contract):
    """Parse the exact stored data slot without exposing the larger audit payload."""

    returned_row_count: int = Field(ge=0)
    returned_column_count: int = Field(ge=0)
    statistics_scope: Literal["returned_rows"]
    result_truncated: bool
    sample_truncated: bool
    sample_row_count: int = Field(ge=0)
    columns_omitted: int = Field(ge=0)
    statistics_fields: list[str]
    statistics: list[list[SqlValue]]
    all_null_columns: list[int]
    sample_rows: list[list[SqlValue]]
    qualification: str
