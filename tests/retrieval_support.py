"""Deterministic retrieval fixtures; no infrastructure or model work at import time."""

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
from pydantic import SecretStr

from app.clients.model_runtime import ModelRuntimeClient
from app.core.config_models import ModelRuntimeClientSettings
from app.core.deadline import Deadline
from app.db.session import Database
from app.retrieval.config import MilvusSettings, RetrievalConfig, RetrievalSettings
from app.retrieval.fusion import SearchArm, candidates
from app.retrieval.ingestion_store import IngestionStore
from app.retrieval.pipeline import RetrievalPipeline
from app.retrieval.search_store import HybridSearchStore
from app.schemas.ingestion import digest
from app.schemas.model_runtime import EmbedResult, ModelFailure, RerankResult
from app.schemas.retrieval import Candidate, EncodedQuery, PointTimeScope, RetrievalQuery
from app.services.ingestion import IngestionService
from app.services.ingestion_config import IngestionSettings
from model_runtime.errors import ModelError
from tests.corpus_support import entry, markdown_source, write_inventory
from tests.ingestion_support import clear_registry, model_metadata
from tests.milvus_support import MilvusStack
from tests.shared_database import TestPostgres


def deadline(seconds: float = 30) -> Deadline:
    return Deadline(time.monotonic() + seconds)


def query() -> RetrievalQuery:
    return RetrievalQuery(
        standalone="SKU-A1023", time_scope=PointTimeScope(as_of=date(2026, 8, 15))
    )


def encoded() -> EncodedQuery:
    return EncodedQuery(
        text="SKU-A1023", dense=[1.0, *([0.0] * 1023)], sparse={7: 1.0}, metadata=model_metadata()
    )


def raw_hit(pk: int = 1, score: float = 0.8) -> dict[str, object]:
    content = "SKU-A1023 七天退款规则。"
    return {
        "pk": pk,
        "distance": score,
        "entity": {
            "chunk_uuid": str(uuid4()),
            "document_id": str(uuid4()),
            "document_version": "a" * 64,
            "chunking_version": "b" * 64,
            "content_sha256": digest(content),
            "doc_type": "policy",
            "heading_path": "政策",
            "content": content,
            "parent_content": "完整政策章节。" + content,
            "effective_from": -1,
            "effective_to": -1,
        },
    }


def candidate() -> Candidate:
    result = candidates([[raw_hit()]], SearchArm.DENSE)[0]
    result.source_path = "source.md"
    return result


def fake_store(**methods: AsyncMock) -> HybridSearchStore:
    result = object.__new__(HybridSearchStore)
    result.settings = MilvusSettings()
    result._client = SimpleNamespace(**methods)
    result._validated = True
    return result


@dataclass
class QueryEmbeddings:
    """Fixed directions deliberately model a dense miss, not real BGE-M3 quality."""

    calls: list[dict[str, object]]
    rerank_calls: list[dict[str, object]] = field(default_factory=list)
    rerank_error: ModelError | None = None
    rerank_scores: list[float] | None = None

    async def handle(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if request.url.path == "/v1/rerank":
            self.rerank_calls.append(payload)
            if self.rerank_error is not None:
                error = self.rerank_error
                failure = ModelFailure(
                    code=error.kind,
                    message=error.user_message,
                    request_id="synthetic-rerank",
                    retryable=error.retryable,
                )
                return httpx.Response(error.http_status, json=failure.model_dump(mode="json"))
            reranked = RerankResult(
                request_id="synthetic-rerank",
                ms=1,
                queue_ms=0,
                inference_ms=1,
                metadata=model_metadata(),
                scores=self.rerank_scores
                if self.rerank_scores is not None
                else [0.8] * len(payload["passages"]),
            )
            return httpx.Response(200, json=reranked.model_dump(mode="json"))
        self.calls.append(payload)
        texts = payload["texts"]
        is_query = payload["mode"] == "query"
        vectors = []
        for value in texts:
            sku_document = any(term in value for term in ("SKU-A1023", "商品包装", "商品规则"))
            sku_document = sku_document and not is_query
            vectors.append([0.0, 1.0, *([0.0] * 1022)] if sku_document else [1.0, *([0.0] * 1023)])
        response = EmbedResult(
            request_id="synthetic-retrieval",
            ms=0,
            queue_ms=0,
            inference_ms=0,
            metadata=model_metadata(),
            dense=vectors,
            sparse=[{7: 1.0} for _ in texts],
        )
        return httpx.Response(200, json=response.model_dump(mode="json"))


def corpus(root: Path) -> None:
    """Separate July/August sources are both members of the committed corpus."""
    sources = [
        (
            "sku.md",
            "# 商品规则\n\nSKU-A1023 七天退款规则。" + "商品包装必须完整。" * 90,
            "2026-01-01",
            None,
        ),
        ("july.md", "# 七月政策\n\n七天退款规则。七月版本。", "2026-07-01", "2026-08-01"),
        ("august.md", "# 八月政策\n\n七天退款规则。八月版本。", "2026-08-01", None),
    ]
    for name, body, start, end in sources:
        source = markdown_source(body).replace("2026-01-01", start)
        if end:
            source = source.replace("effective_to: null", "effective_to: " + end)
        (root / name).write_text(source)
    write_inventory(root, [entry(name) for name, *_ in sources])


@dataclass
class RetrievalHarness:
    database: Database
    ingestion: IngestionStore
    search: HybridSearchStore
    model: ModelRuntimeClient
    model_settings: ModelRuntimeClientSettings
    embeddings: QueryEmbeddings
    root: Path

    async def ingest(self) -> None:
        result = await IngestionService(
            self.database,
            self.ingestion,
            self.model,
            IngestionSettings(),
            self.model_settings,
        ).ingest(self.root)
        assert result.successful

    def pipeline(self, config: RetrievalConfig | None = None) -> RetrievalPipeline:
        return RetrievalPipeline(
            self.database,
            self.search,
            self.model,
            RetrievalSettings(
                search=config or RetrievalConfig(use_rerank=False),
                search_timeout_s=30,
                milvus=self.search.settings,
            ),
        )


@pytest.fixture
async def harness(
    migrated_db: TestPostgres, milvus_stack: MilvusStack, tmp_path: Path
) -> AsyncIterator[RetrievalHarness]:
    database = Database(migrated_db.app)
    database.start()
    settings = ModelRuntimeClientSettings(auth_token=SecretStr("synthetic-retrieval-token"))
    storage = MilvusSettings(uri=milvus_stack.uri, collection="step36_" + uuid4().hex, timeout_s=30)
    embeddings = QueryEmbeddings(calls=[])
    corpus(tmp_path)
    try:
        await clear_registry(database)
        async with (
            httpx.AsyncClient(transport=httpx.MockTransport(embeddings.handle)) as http,
            IngestionStore(storage) as ingestion,
            HybridSearchStore(storage) as search,
        ):
            model = ModelRuntimeClient(settings, http)
            value = RetrievalHarness(
                database, ingestion, search, model, settings, embeddings, tmp_path
            )
            await value.ingest()
            await search.ensure_collection()
            yield value
    finally:
        await clear_registry(database)
        await database.aclose()
