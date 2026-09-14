"""Bounded storage settings, independent of API credentials and model libraries."""

from typing import Annotated, Literal, Self

from pydantic import Field, HttpUrl, model_validator

from app.core.settings_base import ConfigModel, require_configuration

SCHEMA_VERSION = "1"
ProtectedTerm = Annotated[str, Field(pattern=r"^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+$", max_length=128)]


class MilvusSettings(ConfigModel):
    """Changing schema/index/analyzer settings requires a fresh collection."""

    uri: HttpUrl = HttpUrl("http://milvus:19530")
    collection: str = Field(default="kb_chunks", pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
    dimension: int = Field(default=1024, ge=1024, le=1024)
    timeout_s: float = Field(default=10, ge=0.01, le=120)
    startup_timeout_s: float = Field(default=90, ge=1, le=300)
    hnsw_m: int = Field(default=16, ge=4, le=64)
    ef_construction: int = Field(default=200, ge=8, le=1024)
    search_ef: int = Field(default=64, ge=1, le=32768)
    drop_ratio_build: float = Field(default=0.2, ge=0, le=1)
    protected_terms: list[ProtectedTerm] = Field(
        default_factory=lambda: ["SKU-A1023", "SKU-B2048"], max_length=10000
    )

    @model_validator(mode="after")
    def validate_index(self) -> Self:
        """Fail on inconsistent construction settings without accepting duplicate terms."""
        require_configuration(
            self.ef_construction >= self.hnsw_m, "ef_construction must be at least hnsw_m"
        )
        require_configuration(
            len(self.protected_terms) == len(set(self.protected_terms)),
            "protected_terms must be unique",
        )
        require_configuration(
            not (self.uri.username or self.uri.password or self.uri.query or self.uri.fragment),
            "Milvus URI must not contain credentials, query or fragment",
        )
        return self


class FilterConfig(ConfigModel):
    """Normalized relevance thresholds and hard diversity/size ceilings."""

    dynamic_ratio: float = Field(default=0.5, ge=0, le=1)
    absolute_floor: float = Field(default=0.30, ge=0, le=1)
    high_ratio: float = Field(default=0.6, ge=0, le=1)
    min_per_doc: int = Field(default=1, ge=1, le=100)
    max_per_doc: int = Field(default=3, ge=1, le=100)
    gap_threshold: float = Field(default=0.15, ge=0, le=1)
    final_k: int = Field(default=8, ge=1, le=100)

    @model_validator(mode="after")
    def ordered_caps(self) -> Self:
        """Minimum allocation cannot bypass the configured hard maximum."""
        require_configuration(self.min_per_doc <= self.max_per_doc, "Invalid document caps")
        return self


class RetrievalConfig(ConfigModel):
    """Effective search parameters snapshotted with every candidate pool."""

    use_dense: bool = True
    use_sparse_learned: bool = True
    use_bm25: bool = True
    pool: int = Field(default=20, ge=1, le=100)
    hnsw_ef: int = Field(default=64, ge=1, le=32768)
    rrf_k: int = Field(default=60, ge=1, le=16384)
    drop_ratio_search: float = Field(default=0.2, ge=0, lt=1)
    record_arm_scores: bool = False
    use_rerank: bool = True
    filtering: FilterConfig = Field(default_factory=FilterConfig)

    @model_validator(mode="after")
    def search_invariants(self) -> Self:
        """Reject unusable search configurations when settings are constructed."""
        require_configuration(
            self.use_dense or self.use_sparse_learned or self.use_bm25,
            "At least one retrieval arm must be enabled",
        )
        require_configuration(self.hnsw_ef >= self.pool, "hnsw_ef must be at least pool")
        return self


class EvidenceConfig(ConfigModel):
    """The complete rendered evidence slot, measured with the bundled tokenizer."""

    max_tokens: int = Field(default=6000, ge=1, le=6000)


class RetrievalSettings(ConfigModel):
    """Storage readiness settings; knowledge graph dispatch is owned by its caller."""

    enabled: bool = False
    search: RetrievalConfig = Field(default_factory=RetrievalConfig)
    evidence: EvidenceConfig = Field(default_factory=EvidenceConfig)
    search_timeout_s: float = Field(default=10, ge=0.01, le=120)
    milvus: MilvusSettings = Field(default_factory=MilvusSettings)


class JiebaTokenizer(ConfigModel):
    """Server-side dictionary; raw content and query strings are never normalized."""

    type: Literal["jieba"] = "jieba"
    dictionary: list[str] = Field(alias="dict")
    mode: Literal["exact"] = "exact"
    hmm: bool = True


def analyzer_filters() -> list[Literal["cnalphanumonly"]]:
    """Return an independent, narrowly typed filter list for each configuration."""
    return ["cnalphanumonly"]


class AnalyzerConfig(ConfigModel):
    """cnalphanumonly retains mixed tokens; removepunct would delete whole SKU tokens."""

    tokenizer: JiebaTokenizer
    filter: list[Literal["cnalphanumonly"]] = Field(default_factory=analyzer_filters)


def analyzer_config(settings: MilvusSettings) -> AnalyzerConfig:
    """Sort the explicit vocabulary so configuration order cannot cause schema drift."""
    return AnalyzerConfig(
        tokenizer=JiebaTokenizer(dict=["_default_", *sorted(settings.protected_terms)])
    )
