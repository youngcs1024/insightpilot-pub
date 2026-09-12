"""Async shared-corpus storage with bounded calls and a fail-closed schema guard."""

import asyncio
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Annotated, Self

import structlog
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from pymilvus import AsyncMilvusClient  # type: ignore[import-untyped]
from pymilvus.exceptions import MilvusException, ParamError  # type: ignore[import-untyped]

from app.core.errors import (
    RetrievalConfigurationError,
    RetrievalSchemaError,
    RetrievalUnavailableError,
)
from app.retrieval.config import SCHEMA_VERSION, MilvusSettings, analyzer_config
from app.retrieval.schema import (
    CollectionDescription,
    CollectionReport,
    IndexDescription,
    collection_indexes,
    collection_schema,
    schema_issues,
)

logger = structlog.get_logger(__name__)
Text = Annotated[str, Field(min_length=1, max_length=65535)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class AnalyzerRequest(BaseModel):
    """Bound analyzer diagnostics to one small batch."""

    texts: list[Text] = Field(min_length=1, max_length=100)


class AnalyzedText(BaseModel):
    """The actual server tokens, never reconstructed by joining fragments."""

    text: str
    tokens: list[str]


class AnalyzerReport(BaseModel):
    """Operator evidence from the same analyzer used by the collection."""

    samples: list[AnalyzedText]


class DenseSearch(BaseModel):
    """Small storage primitive; fusion and model encoding belong to later steps."""

    vector: list[FiniteFloat] = Field(min_length=1024, max_length=1024)
    limit: int = Field(default=10, ge=1, le=100)


class HitText(BaseModel):
    """Stable corpus identity and original child/parent text returned by storage."""

    chunk_uuid: str
    document_id: str
    content: str
    parent_content: str


class StorageHit(BaseModel):
    """Normalize SDK hits without passing opaque dictionaries to callers."""

    id: int = Field(strict=True)
    distance: FiniteFloat = Field(strict=True)
    entity: HitText


class SdkHit(BaseModel):
    """The pinned SDK maps the collection primary-key name, not an id key."""

    pk: int = Field(strict=True)
    distance: FiniteFloat = Field(strict=True)
    entity: HitText


class SdkTokens(BaseModel):
    """Validate the SDK analyzer object's token attribute without iterating it."""

    tokens: list[str]


def search_hits(raw: object) -> list[StorageHit]:
    """Normalize exactly one query response and classify malformed upstream data."""
    try:
        batches = TypeAdapter(list[list[SdkHit]]).validate_python(raw)
    except ValidationError as exc:
        raise RetrievalUnavailableError(reason="invalid_search_response") from exc
    if len(batches) != 1:
        raise RetrievalUnavailableError(reason="search_result_count")
    return [StorageHit(id=hit.pk, distance=hit.distance, entity=hit.entity) for hit in batches[0]]


def analyzer_report(raw: object, request: AnalyzerRequest) -> AnalyzerReport:
    """Normalize token objects and reject incomplete or malformed batches."""
    try:
        results = TypeAdapter(list[SdkTokens]).validate_python(raw, from_attributes=True)
    except ValidationError as exc:
        raise RetrievalUnavailableError(reason="invalid_analyzer_response") from exc
    if len(results) != len(request.texts):
        raise RetrievalUnavailableError(reason="analyzer_result_count")
    return AnalyzerReport(
        samples=[
            AnalyzedText(text=text, tokens=result.tokens)
            for text, result in zip(request.texts, results, strict=True)
        ]
    )


class MilvusRepository:
    """The corpus is shared; this repository never accesses user-owned records.

    The outer asyncio deadline covers connection, SDK calls and cancellation.
    SDK timeout=None plus retry_times=0 disables its timeout-driven retry loop;
    passing both a numeric SDK timeout and retry_times=0 does NOT disable retries
    in PyMilvus 3.0.1. No application retry layer is added. Initialization can be
    explicitly repeated, but never deletes or overwrites incompatible state.
    """

    def __init__(self, settings: MilvusSettings) -> None:
        self.settings = settings.model_copy(deep=True)
        self._client = AsyncMilvusClient(uri=str(settings.uri), timeout=settings.timeout_s)
        self._validated = False

    async def __aenter__(self) -> Self:
        """Connection is lazy and covered by the first operation's deadline."""
        return self

    async def __aexit__(self, *_exc: object) -> None:
        """Close on success, typed failure and caller cancellation."""
        await self.aclose()

    async def _rpc[T](self, name: str, operation: Callable[[], Awaitable[T]]) -> T:
        try:
            async with asyncio.timeout(self.settings.timeout_s):
                return await operation()
        except ParamError as exc:
            logger.exception("milvus_configuration_failed", operation=name)
            raise RetrievalConfigurationError(operation=name) from exc
        except MilvusException as exc:
            logger.exception("milvus_sdk_failed", operation=name)
            if exc.is_input_error:
                raise RetrievalConfigurationError(operation=name) from exc
            raise RetrievalUnavailableError(operation=name) from exc
        except Exception as exc:
            logger.exception("milvus_unavailable", operation=name)
            raise RetrievalUnavailableError(operation=name) from exc

    async def aclose(self) -> None:
        """Release the dedicated client reference with a bounded wait."""
        self._validated = False
        await self._rpc("close", self._client.close)

    async def ensure_collection(self) -> CollectionReport:
        """Create an absent collection, or validate existing state before loading."""
        self._validated = False
        try:
            async with asyncio.timeout(self.settings.startup_timeout_s):
                return await self._initialize()
        except TimeoutError as exc:
            raise RetrievalUnavailableError(operation="initialize") from exc

    async def _initialize(self) -> CollectionReport:
        name = self.settings.collection
        exists = await self._rpc(
            "has_collection",
            lambda: self._client.has_collection(
                name,
                timeout=None,
                retry_times=0,
                retry_on_rate_limit=False,
            ),
        )
        if not exists:
            await self._rpc(
                "create_collection",
                lambda: self._client.create_collection(
                    collection_name=name,
                    schema=collection_schema(self.settings),
                    properties={"schema_version": SCHEMA_VERSION},
                    timeout=None,
                    retry_times=0,
                    retry_on_rate_limit=False,
                ),
            )
            await self._rpc(
                "create_index",
                lambda: self._client.create_index(
                    name,
                    collection_indexes(self.settings),
                    timeout=None,
                    retry_times=0,
                    retry_on_rate_limit=False,
                ),
            )
        report = await self.validate_collection()
        await self._rpc(
            "load_collection",
            lambda: self._client.load_collection(
                name,
                timeout=None,
                retry_times=0,
                retry_on_rate_limit=False,
            ),
        )
        self._validated = True
        logger.info("milvus_collection_ready", collection=name, schema_version=SCHEMA_VERSION)
        return report

    async def validate_collection(self) -> CollectionReport:
        """Read back persisted fields, properties, analyzer and indexes every time."""
        self._validated = False
        try:
            report = await self.describe()
            issues = schema_issues(report, self.settings)
        except ValidationError as exc:
            raise RetrievalSchemaError(reason="invalid_metadata") from exc
        if issues:
            actual = report.description.properties.get("schema_version")
            # Only validated identifiers and fixed diagnostics are logged, never corpus text.
            logger.warning(
                "milvus_schema_mismatch",
                collection=self.settings.collection,
                expected=SCHEMA_VERSION,
                actual=actual,
                issues=issues,
                recovery="python -m scripts.milvus_init --collection kb_chunks_v1",
            )
            raise RetrievalSchemaError(expected=SCHEMA_VERSION, actual=actual, issues=issues)
        return report

    async def describe(self) -> CollectionReport:
        """Reject malformed metadata without exposing validation implementation errors."""
        try:
            return await self._describe()
        except ValidationError as exc:
            raise RetrievalSchemaError(reason="invalid_metadata") from exc

    async def _describe(self) -> CollectionReport:
        """Normalize the vendor's response before it crosses the component boundary."""
        name = self.settings.collection
        raw = await self._rpc(
            "describe_collection",
            lambda: self._client.describe_collection(
                name,
                timeout=None,
                retry_times=0,
                retry_on_rate_limit=False,
            ),
        )
        description = CollectionDescription.model_validate(raw)
        names = TypeAdapter(list[str]).validate_python(
            await self._rpc(
                "list_indexes",
                lambda: self._client.list_indexes(
                    name,
                    timeout=None,
                    retry_times=0,
                    retry_on_rate_limit=False,
                ),
            )
        )
        if len(names) != len(set(names)):
            raise RetrievalSchemaError(reason="duplicate_index_names")
        indexes = []
        for index in sorted(names):
            result = await self._rpc(
                "describe_index",
                partial(
                    self._client.describe_index,
                    name,
                    index,
                    timeout=None,
                    retry_times=0,
                    retry_on_rate_limit=False,
                ),
            )
            indexes.append(IndexDescription.model_validate(result))
        return CollectionReport(description=description, indexes=indexes)

    async def analyze(self, request: AnalyzerRequest) -> AnalyzerReport:
        """Use the server analyzer configuration unchanged for every sample."""
        raw = await self._rpc(
            "run_analyzer",
            lambda: self._client.run_analyzer(
                request.texts,
                analyzer_params=analyzer_config(self.settings).model_dump(by_alias=True),
                timeout=None,
                retry_times=0,
                retry_on_rate_limit=False,
            ),
        )
        return analyzer_report(raw, request)

    async def search_dense(self, request: DenseSearch) -> list[StorageHit]:
        """Record and send search-time ef, always fetching original parent content."""
        if not self._validated:
            await self.ensure_collection()
        ef = max(self.settings.search_ef, request.limit)
        raw = await self._rpc(
            "search_dense",
            lambda: self._client.search(
                self.settings.collection,
                data=[request.vector],
                anns_field="dense",
                limit=request.limit,
                search_params={"metric_type": "IP", "params": {"ef": ef}},
                output_fields=["chunk_uuid", "document_id", "content", "parent_content"],
                timeout=None,
                retry_times=0,
                retry_on_rate_limit=False,
            ),
        )
        logger.info("milvus_dense_searched", ef=ef, limit=request.limit)
        return search_hits(raw)
