"""Step 3.1 acceptance against pinned Milvus, etcd and MinIO from production Compose."""

import asyncio
import json
import statistics
import time
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from pymilvus import AsyncMilvusClient, DataType

from app.core.errors import RetrievalSchemaError
from app.retrieval.config import MilvusSettings
from app.retrieval.milvus_repo import AnalyzerRequest, DenseSearch, MilvusRepository
from app.retrieval.schema import SCALAR_FIELDS, TEXT_FIELDS
from scripts.analyzer_smoke import SAMPLES, require_sku_tokens
from tests.milvus_support import MilvusStack

pytestmark = [pytest.mark.integration, pytest.mark.storage]


@pytest.fixture
async def repository(
    milvus_stack: MilvusStack, request: pytest.FixtureRequest
) -> AsyncIterator[MilvusRepository]:
    async with (
        milvus_stack.collection("step31", request) as config,
        MilvusRepository(config) as repo,
    ):
        await repo.ensure_collection()
        yield repo


async def test_collection_created_with_all_fields(
    repository: MilvusRepository, milvus_stack: MilvusStack
) -> None:
    report = await repository.describe()
    fields = {field.name: field for field in report.description.fields}
    assert set(fields) == {
        "pk",
        *TEXT_FIELDS,
        "content",
        "effective_from",
        "effective_to",
        "dense",
        "sparse_learned",
        "sparse_bm25",
    }
    assert fields["dense"].params["dim"] == repository.settings.dimension
    assert report.description.properties["schema_version"] == "1"
    assert not report.description.enable_dynamic_field
    assert report.description.auto_id
    (milvus_stack.directory / "collection.json").write_text(report.model_dump_json(indent=2))


async def test_bm25_function_registered(repository: MilvusRepository) -> None:
    report = await repository.describe()
    (function,) = report.description.functions
    assert function.name == "bm25"
    assert function.input_field_names == ["content"]
    assert function.output_field_names == ["sparse_bm25"]
    assert {index.field_name for index in report.indexes} == {
        *SCALAR_FIELDS,
        "dense",
        "sparse_learned",
        "sparse_bm25",
    }


async def test_jieba_analyzer_tokenizes_chinese(
    repository: MilvusRepository, milvus_stack: MilvusStack
) -> None:
    report = await repository.analyze(AnalyzerRequest(texts=list(SAMPLES)))
    assert {"七天", "退货", "政策"} <= set(report.samples[0].tokens)
    assert {"618", "大促", "规则"} <= set(report.samples[2].tokens)
    (milvus_stack.directory / "analyzer-smoke.json").write_text(report.model_dump_json(indent=2))


async def test_analyzer_preserves_sku_codes(repository: MilvusRepository) -> None:
    report = await repository.analyze(AnalyzerRequest(texts=list(SAMPLES)))
    require_sku_tokens(report)
    assert "SKU-A1023" in report.samples[4].tokens
    assert "SKU-B2048" in report.samples[9].tokens


async def test_schema_version_mismatch_refuses_to_start(repository: MilvusRepository) -> None:
    async with AsyncMilvusClient(uri=str(repository.settings.uri), timeout=10) as client:
        await client.release_collection(repository.settings.collection, timeout=10)
        await client.alter_collection_properties(
            repository.settings.collection, properties={"schema_version": "old"}, timeout=10
        )
    with pytest.raises(RetrievalSchemaError):
        await repository.ensure_collection()
    report = await repository.describe()
    assert report.description.properties["schema_version"] == "old"


async def test_vocabulary_change_requires_new_collection(repository: MilvusRepository) -> None:
    config = MilvusSettings.model_validate(
        {
            **repository.settings.model_dump(),
            "protected_terms": ["SKU-A1023", "SKU-B2048", "SKU-X9876"],
        }
    )
    async with MilvusRepository(config) as changed:
        with pytest.raises(RetrievalSchemaError):
            await changed.ensure_collection()


async def seed_three(repo: MilvusRepository) -> None:
    rows = []
    for index, (content, kind) in enumerate(
        [
            ("SKU-A1023 七天无理由退货政策", "policy"),
            ("SKU-B2048 促销折扣规则", "promotion"),
            ("订单取消退款说明", "policy"),
        ]
    ):
        rows.append(
            {
                "chunk_uuid": str(uuid4()),
                "document_id": str(uuid4()),
                "document_version": "1",
                "chunking_version": "1",
                "content_sha256": str(index) * 64,
                "doc_type": kind,
                "effective_from": -1,
                "effective_to": -1,
                "heading_path": "退货规则",
                "content": content,
                "parent_content": "完整章节:" + content,
                "dense": [float(index == 0)] + [0.0] * 1023,
                "sparse_learned": {index + 1: 1.0},
            }
        )
    async with AsyncMilvusClient(uri=str(repo.settings.uri), timeout=10) as client:
        await client.insert(repo.settings.collection, data=rows, timeout=10)
        await client.flush(repo.settings.collection, timeout=30)


async def test_three_arms_and_index_readback(repository: MilvusRepository) -> None:
    await seed_three(repository)
    hits = await repository.search_dense(DenseSearch(vector=[1.0] + [0.0] * 1023, limit=1))
    assert "SKU-A1023" in hits[0].entity.content
    assert hits[0].entity.parent_content != hits[0].entity.content
    async with AsyncMilvusClient(uri=str(repository.settings.uri), timeout=10) as client:
        for field, data, metric in [
            ("sparse_learned", [{1: 1.0}], "IP"),
            ("sparse_bm25", ["SKU-A1023"], "BM25"),
        ]:
            result = await client.search(
                repository.settings.collection,
                data=data,
                anns_field=field,
                search_params={"metric_type": metric},
                limit=1,
                output_fields=["content"],
                timeout=10,
            )
            assert "SKU-A1023" in result[0][0]["entity"]["content"]
    indexes = {item.field_name: item for item in (await repository.describe()).indexes}
    assert indexes["dense"].params == {"M": "16", "efConstruction": "200"}
    assert indexes["sparse_learned"].params["drop_ratio_build"] == "0.2"


async def test_idempotency_and_restart_persist_schema(
    repository: MilvusRepository, milvus_stack: MilvusStack
) -> None:
    await seed_three(repository)
    before = await repository.describe()
    await repository.ensure_collection()
    # Avoid a blocking Docker subprocess inside the async test.
    await asyncio.to_thread(milvus_stack.restart)
    async with MilvusRepository(repository.settings) as restarted:
        after = await restarted.ensure_collection()
        assert before == after
        hits = await restarted.search_dense(DenseSearch(vector=[1.0] + [0.0] * 1023, limit=1))
        assert "SKU-A1023" in hits[0].entity.content


async def wait_scalar_index(client: AsyncMilvusClient, name: str) -> None:
    """Wait for background indexing, which may outlive collection loading."""
    async with asyncio.timeout(60):
        while True:
            index = await client.describe_index(name, "document_id", timeout=10)
            assert index["index_type"] == "INVERTED"
            if index["state"] == "Finished":
                return
            assert index["state"] in {"Unissued", "InProgress"}, index
            await asyncio.sleep(0.5)


async def measure_scalar(client: AsyncMilvusClient, name: str, *, indexed: bool) -> list[float]:
    """Create one sealed 50k-row cohort and retain its post-warmup RPC timings."""
    count, batch_size, warmup = 50000, 5000, 5
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("pk", DataType.INT64, is_primary=True)
    schema.add_field("document_id", DataType.VARCHAR, max_length=36)
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=2)
    indexes = client.prepare_index_params()
    indexes.add_index("dense", index_type="FLAT", metric_type="IP")
    if indexed:
        indexes.add_index("document_id", index_type="INVERTED")
    await client.create_collection(name, schema=schema, index_params=indexes, timeout=30)
    for start in range(0, count, batch_size):
        rows = [
            {"pk": i, "document_id": f"doc-{i:08d}", "dense": [0.0, 1.0]}
            for i in range(start, start + batch_size)
        ]
        await client.insert(name, rows, timeout=30)
    await client.flush(name, timeout=30)
    await client.release_collection(name, timeout=30)
    await client.load_collection(name, timeout=30)
    if indexed:
        await wait_scalar_index(client, name)
    elapsed = []
    for iteration in range(warmup + 30):
        target = (iteration * 1423) % count
        began = time.perf_counter()
        rows = await client.query(
            name, filter=f'document_id == "doc-{target:08d}"', output_fields=["pk"], timeout=10
        )
        measured = time.perf_counter() - began
        assert rows == [{"pk": target}]
        if iteration >= warmup:
            elapsed.append(measured)
    return elapsed


async def test_scalar_filter_uses_index(
    milvus_stack: MilvusStack, request: pytest.FixtureRequest
) -> None:
    """Paired sealed-segment timings; both collections are released after measurement."""
    async with (
        milvus_stack.collection("step31_scalar", request) as scan,
        milvus_stack.collection("step31_scalar", request) as indexed,
        AsyncMilvusClient(uri=milvus_stack.uri, timeout=30) as client,
    ):
        timings = {
            "scan": await measure_scalar(client, scan.collection, indexed=False),
            "indexed": await measure_scalar(client, indexed.collection, indexed=True),
        }
    (milvus_stack.directory / "scalar-timing.json").write_text(json.dumps(timings, indent=2))
    # End-to-end RPC timing includes noise. A 25% regression is not acceptable.
    assert statistics.median(timings["indexed"]) <= statistics.median(timings["scan"]) * 1.25


async def test_sequential_owned_collections_leave_no_residuals(
    milvus_stack: MilvusStack, request: pytest.FixtureRequest
) -> None:
    async with AsyncMilvusClient(uri=milvus_stack.uri, timeout=10) as client:
        before = set(await client.list_collections(timeout=10))
        for _ in range(3):
            async with (
                milvus_stack.collection("step31", request) as config,
                MilvusRepository(config) as repository,
            ):
                await repository.ensure_collection()
                await seed_three(repository)
            assert not await client.has_collection(config.collection, timeout=10)
        assert set(await client.list_collections(timeout=10)) == before
