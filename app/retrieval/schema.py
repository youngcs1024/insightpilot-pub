"""Pure collection contract and strict normalization at the untyped SDK boundary."""

import json

from pydantic import BaseModel, Field, JsonValue, model_validator
from pymilvus import CollectionSchema, DataType, Function, FunctionType  # type: ignore[import-untyped]
from pymilvus.milvus_client.index import IndexParams  # type: ignore[import-untyped]

from app.core.errors import RetrievalSchemaError
from app.retrieval.config import SCHEMA_VERSION, MilvusSettings, analyzer_config

TEXT_FIELDS = {
    "chunk_uuid": 36,
    "document_id": 36,
    "document_version": 64,
    "chunking_version": 64,
    "content_sha256": 64,
    "doc_type": 32,
    "heading_path": 512,
    "parent_content": 65535,
}
SCALAR_FIELDS = ("document_id", "doc_type", "effective_from", "effective_to")


class CollectionField(BaseModel):
    """Vendor field metadata; unrelated server-generated identifiers are ignored."""

    name: str
    type: int
    params: dict[str, JsonValue] = Field(default_factory=dict)
    is_primary: bool = False
    auto_id: bool = False
    is_function_output: bool = False


class CollectionFunction(BaseModel):
    """BM25 binding, independent of generated numeric field IDs."""

    name: str
    type: int
    input_field_names: list[str]
    output_field_names: list[str]


class CollectionDescription(BaseModel):
    """Typed persisted collection description, never an SDK dictionary outside storage."""

    collection_name: str
    auto_id: bool
    enable_dynamic_field: bool
    fields: list[CollectionField]
    functions: list[CollectionFunction]
    properties: dict[str, str]


class IndexDescription(BaseModel):
    """Server-returned index type, metric and algorithm parameters."""

    field_name: str
    index_name: str
    index_type: str
    metric_type: str | None = None
    params: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_parameters(cls, value: object) -> object:
        """PyMilvus 3 returns construction parameters at the top level."""
        if isinstance(value, dict):
            return {
                **value,
                "params": {
                    key: value[key]
                    for key in ("M", "efConstruction", "drop_ratio_build")
                    if key in value
                },
            }
        return value


class CollectionReport(BaseModel):
    """Stable diagnostic result shared by operators and integration tests."""

    description: CollectionDescription
    indexes: list[IndexDescription]

    @model_validator(mode="after")
    def canonical_indexes(self) -> "CollectionReport":
        """Reject duplicate identities before making server ordering irrelevant."""
        names = [index.index_name for index in self.indexes]
        fields = [index.field_name for index in self.indexes]
        if len(set(names)) != len(names) or len(set(fields)) != len(fields):
            raise RetrievalSchemaError(reason="duplicate_index_identity")
        self.indexes = sorted(self.indexes, key=lambda index: (index.index_name, index.field_name))
        return self


def collection_schema(settings: MilvusSettings) -> CollectionSchema:
    """Build the three retrieval fields and indexed temporal/document metadata."""
    schema = CollectionSchema([], auto_id=True, enable_dynamic_field=False, check_fields=False)
    schema.add_field("pk", DataType.INT64, is_primary=True, auto_id=True)
    for name, length in TEXT_FIELDS.items():
        schema.add_field(name, DataType.VARCHAR, max_length=length)
    for name in ("effective_from", "effective_to"):
        schema.add_field(name, DataType.INT64)
    schema.add_field(
        "content",
        DataType.VARCHAR,
        max_length=65535,
        enable_analyzer=True,
        analyzer_params=analyzer_config(settings).model_dump(by_alias=True),
    )
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=settings.dimension)
    schema.add_field("sparse_learned", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("sparse_bm25", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(
        Function(
            name="bm25",
            function_type=FunctionType.BM25,
            input_field_names=["content"],
            output_field_names=["sparse_bm25"],
        )
    )
    return schema


def collection_indexes(settings: MilvusSettings) -> IndexParams:
    """Explicit algorithms and names support deterministic drift checks."""
    indexes = IndexParams()
    indexes.add_index(
        "dense",
        index_name="dense",
        index_type="HNSW",
        metric_type="IP",
        params={"M": settings.hnsw_m, "efConstruction": settings.ef_construction},
    )
    indexes.add_index(
        "sparse_learned",
        index_name="sparse_learned",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="IP",
        params={"drop_ratio_build": settings.drop_ratio_build},
    )
    indexes.add_index(
        "sparse_bm25",
        index_name="sparse_bm25",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
    )
    for name in SCALAR_FIELDS:
        indexes.add_index(name, index_name=name, index_type="INVERTED")
    return indexes


def expected_fields(settings: MilvusSettings) -> dict[str, int]:
    """Expected exact field set, including server-generated BM25 output."""
    return {
        "pk": DataType.INT64,
        **dict.fromkeys(TEXT_FIELDS, DataType.VARCHAR),
        "effective_from": DataType.INT64,
        "effective_to": DataType.INT64,
        "content": DataType.VARCHAR,
        "dense": DataType.FLOAT_VECTOR,
        "sparse_learned": DataType.SPARSE_FLOAT_VECTOR,
        "sparse_bm25": DataType.SPARSE_FLOAT_VECTOR,
    }


def schema_issues(report: CollectionReport, settings: MilvusSettings) -> list[str]:
    """Fail closed on version, field, analyzer, function or index drift."""
    description = report.description
    issues: list[str] = []
    if description.properties.get("schema_version") != SCHEMA_VERSION:
        issues.append("schema_version")
    if not description.auto_id or description.enable_dynamic_field:
        issues.append("collection_flags")
    fields = {field.name: field for field in description.fields}
    if {name: field.type for name, field in fields.items()} != expected_fields(settings):
        return [*issues, "fields"]
    issues.extend(field_issues(fields, settings))
    expected = CollectionFunction(
        name="bm25",
        type=FunctionType.BM25,
        input_field_names=["content"],
        output_field_names=["sparse_bm25"],
    )
    if description.functions != [expected]:
        issues.append("bm25_function")
    return [*issues, *index_issues(report.indexes, settings)]


def field_issues(fields: dict[str, CollectionField], settings: MilvusSettings) -> list[str]:
    """Validate field flags, dimensions, byte bounds and the actual analyzer."""
    issues: list[str] = []
    if not fields["pk"].is_primary or not fields["pk"].auto_id:
        issues.append("primary_key")
    if not fields["sparse_bm25"].is_function_output:
        issues.append("bm25_output")
    for name, length in {**TEXT_FIELDS, "content": 65535}.items():
        if str(fields[name].params.get("max_length")) != str(length):
            issues.append(name)
    if str(fields["dense"].params.get("dim")) != str(settings.dimension):
        issues.append("dimension")
    if not analyzer_matches(fields["content"], settings):
        issues.append("analyzer")
    return issues


def analyzer_matches(field: CollectionField, settings: MilvusSettings) -> bool:
    """Normalize the server's JSON string and boolean text without accepting missing keys."""
    raw = field.params.get("analyzer_params")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return False
    return field.params.get("enable_analyzer") in (True, "true") and raw == analyzer_config(
        settings
    ).model_dump(by_alias=True)


def index_issues(indexes: list[IndexDescription], settings: MilvusSettings) -> list[str]:
    """Compare all required algorithms, metrics and configured construction parameters."""
    actual = {index.field_name: index for index in indexes}
    expected = {
        "dense": ("HNSW", "IP"),
        "sparse_learned": ("SPARSE_INVERTED_INDEX", "IP"),
        "sparse_bm25": ("SPARSE_INVERTED_INDEX", "BM25"),
        **dict.fromkeys(SCALAR_FIELDS, ("INVERTED", None)),
    }
    issues = [
        name
        for name, contract in expected.items()
        if name not in actual or (actual[name].index_type, actual[name].metric_type) != contract
    ]
    params: dict[str, dict[str, int | float]] = {
        "dense": {"M": settings.hnsw_m, "efConstruction": settings.ef_construction},
        "sparse_learned": {"drop_ratio_build": settings.drop_ratio_build},
    }
    for name, required in params.items():
        if name in actual and any(
            str(actual[name].params.get(key)) != str(value) for key, value in required.items()
        ):
            issues.append(name + "_params")
    return issues
