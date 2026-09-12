"""Storage guards and deadlines without a running Milvus or model service."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError
from pymilvus.client.search_result import Hit
from pymilvus.exceptions import MilvusException, ParamError

from app.core.errors import (
    RetrievalConfigurationError,
    RetrievalSchemaError,
    RetrievalUnavailableError,
)
from app.retrieval.config import MilvusSettings, analyzer_config
from app.retrieval.milvus_repo import AnalyzerRequest, DenseSearch, MilvusRepository
from app.retrieval.schema import collection_schema
from scripts.analyzer_smoke import require_sku_tokens


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    config = MilvusSettings()
    schema = collection_schema(config)
    schema.verify()
    raw = schema.to_dict()
    raw.update(
        collection_name="kb_chunks",
        properties={"schema_version": "1"},
        auto_id=True,
        enable_dynamic_field=False,
    )
    for field in raw["fields"]:
        if field["name"] == "sparse_bm25":
            field["is_function_output"] = True
    indexes = {
        "dense": {"index_type": "HNSW", "metric_type": "IP", "M": "16", "efConstruction": "200"},
        "sparse_learned": {
            "index_type": "SPARSE_INVERTED_INDEX",
            "metric_type": "IP",
            "drop_ratio_build": "0.2",
        },
        "sparse_bm25": {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25"},
        **{
            name: {"index_type": "INVERTED"}
            for name in ("document_id", "doc_type", "effective_from", "effective_to")
        },
    }
    client = MagicMock()
    for name in ("close", "create_collection", "create_index", "load_collection"):
        setattr(client, name, AsyncMock())
    client.has_collection = AsyncMock(return_value=True)
    client.describe_collection = AsyncMock(return_value=raw)
    client.list_indexes = AsyncMock(return_value=list(indexes))
    client.describe_index = AsyncMock(
        side_effect=lambda _name, index, **_kw: {
            **indexes[index],
            "field_name": index,
            "index_name": index,
        }
    )
    client.search = AsyncMock(return_value=[[]])
    monkeypatch.setattr("app.retrieval.milvus_repo.AsyncMilvusClient", lambda **_kwargs: client)
    return client


@pytest.mark.parametrize(
    "change",
    [
        {"dimension": 768},
        {"timeout_s": 0},
        {"search_ef": 0},
        {"hnsw_m": 2},
        {"hnsw_m": 64, "ef_construction": 32},
        {"drop_ratio_build": 1.1},
        {"collection": "other;drop"},
        {"protected_terms": ["SKU-A", "SKU-A"]},
        {"protected_terms": ["bad term"]},
        {"uri": "http://user:secret@milvus:19530"},
    ],
)
def test_invalid_storage_configuration(change: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        MilvusSettings.model_validate(change)


def test_analyzer_dictionary_is_canonical_and_preserves_hyphens() -> None:
    config = MilvusSettings(protected_terms=["SKU-B2048", "SKU-A1023"])
    analyzer = analyzer_config(config).model_dump(by_alias=True)
    assert analyzer["tokenizer"]["dict"] == ["_default_", "SKU-A1023", "SKU-B2048"]
    assert analyzer["filter"] == ["cnalphanumonly"]


async def test_existing_collection_is_validated_without_writes(sdk: MagicMock) -> None:
    async with MilvusRepository(MilvusSettings()) as repository:
        await repository.ensure_collection()
        await repository.ensure_collection()
    sdk.create_collection.assert_not_called()
    sdk.create_index.assert_not_called()
    expected_reads = 2
    assert sdk.describe_collection.await_count == expected_reads
    sdk.close.assert_awaited_once()


async def test_new_collection_persists_version_before_indexes(sdk: MagicMock) -> None:
    sdk.has_collection.return_value = False
    async with MilvusRepository(MilvusSettings()) as repository:
        await repository.ensure_collection()
    assert sdk.create_collection.call_args.kwargs["properties"] == {"schema_version": "1"}
    sdk.create_index.assert_awaited_once()
    sdk.load_collection.assert_awaited_once()


@pytest.mark.parametrize(
    "drift", ["version", "missing_version", "field", "analyzer", "function", "dimension", "index"]
)
async def test_schema_drift_refuses_load(sdk: MagicMock, drift: str) -> None:
    raw = deepcopy(sdk.describe_collection.return_value)
    if drift == "version":
        raw["properties"]["schema_version"] = "old"
    elif drift == "missing_version":
        raw["properties"] = {}
    elif drift == "field":
        raw["fields"].pop()
    elif drift == "function":
        raw["functions"] = []
    elif drift == "index":
        sdk.list_indexes.return_value = ["dense"]
    else:
        target = "content" if drift == "analyzer" else "dense"
        for field in raw["fields"]:
            if field["name"] == target:
                field["params"] = {}
    sdk.describe_collection.return_value = raw
    async with MilvusRepository(MilvusSettings()) as repository:
        with pytest.raises(RetrievalSchemaError):
            await repository.ensure_collection()
    sdk.load_collection.assert_not_called()
    sdk.create_collection.assert_not_called()


async def test_incomplete_initialization_is_not_silently_repaired(sdk: MagicMock) -> None:
    sdk.has_collection.return_value = False
    sdk.create_index.side_effect = MilvusException(message="unavailable")
    async with MilvusRepository(MilvusSettings()) as repository:
        with pytest.raises(RetrievalUnavailableError):
            await repository.ensure_collection()
        sdk.has_collection.return_value = True
        sdk.list_indexes.return_value = []
        with pytest.raises(RetrievalSchemaError):
            await repository.ensure_collection()
    sdk.create_collection.assert_awaited_once()
    sdk.create_index.assert_awaited_once()


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (MilvusException(message="connection lost"), RetrievalUnavailableError),
        (OSError("connection lost"), RetrievalUnavailableError),
        (ParamError(message="bad analyzer"), RetrievalConfigurationError),
    ],
)
async def test_typed_sdk_failures(
    sdk: MagicMock, error: Exception, expected: type[Exception]
) -> None:
    sdk.has_collection.side_effect = error
    async with MilvusRepository(MilvusSettings()) as repository:
        with pytest.raises(expected):
            await repository.ensure_collection()
    sdk.close.assert_awaited_once()


async def test_timeout_cancels_call_without_retry(sdk: MagicMock) -> None:
    async def hang(*_args: object, **_kwargs: object) -> None:
        await asyncio.Event().wait()

    sdk.has_collection.side_effect = hang
    async with MilvusRepository(MilvusSettings(timeout_s=0.01)) as repository:
        with pytest.raises(RetrievalUnavailableError):
            await repository.ensure_collection()
    sdk.has_collection.assert_awaited_once()
    assert sdk.has_collection.call_args.kwargs["retry_times"] == 0
    assert sdk.has_collection.call_args.kwargs["timeout"] is None


async def test_cancellation_propagates_and_closes(sdk: MagicMock) -> None:
    sdk.has_collection.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        async with MilvusRepository(MilvusSettings()) as repository:
            await repository.ensure_collection()
    sdk.close.assert_awaited_once()


async def test_dense_search_sends_ef_and_parent_content(sdk: MagicMock) -> None:
    async with MilvusRepository(MilvusSettings(search_ef=8)) as repository:
        assert await repository.search_dense(DenseSearch(vector=[0.0] * 1024, limit=20)) == []
    kwargs = sdk.search.call_args.kwargs
    assert kwargs["search_params"] == {"metric_type": "IP", "params": {"ef": 20}}
    assert "parent_content" in kwargs["output_fields"]


async def test_analyzer_reads_tokens_without_joining_fragments(sdk: MagicMock) -> None:
    sdk.run_analyzer = AsyncMock(return_value=[SimpleNamespace(tokens=["SKU", "A1023", "退款"])])
    async with MilvusRepository(MilvusSettings()) as repository:
        report = await repository.analyze(AnalyzerRequest(texts=["SKU-A1023 退款"]))
    with pytest.raises(RetrievalConfigurationError):
        require_sku_tokens(report)


def test_data_graph_does_not_construct_storage(sdk: MagicMock) -> None:
    # API configuration and data graph imports retain no retrieval client dependency.
    from app.agents.data.graph import build  # noqa: PLC0415 -- exercise import boundary.

    assert build() is not None
    sdk.has_collection.assert_not_called()


async def test_real_sdk_nonempty_hit_maps_primary_key(sdk: MagicMock) -> None:
    sdk.search.return_value = [
        [
            Hit(
                {
                    "pk": 42,
                    "distance": 0.75,
                    "entity": {
                        "chunk_uuid": "chunk",
                        "document_id": "doc",
                        "content": "child",
                        "parent_content": "parent",
                    },
                },
                pk_name="pk",
            )
        ]
    ]
    async with MilvusRepository(MilvusSettings()) as repository:
        hits = await repository.search_dense(DenseSearch(vector=[0.0] * 1024))
    expected_pk, expected_score = 42, 0.75
    assert hits[0].id == expected_pk
    assert hits[0].distance == expected_score
    assert hits[0].entity.parent_content == "parent"
    sdk.search.assert_awaited_once()


@pytest.mark.parametrize(
    "change",
    [
        {"pk": None},
        {"pk": "42"},
        {"pk": True},
        {"distance": float("nan")},
        {"distance": float("inf")},
        {"distance": "0.5"},
        {"entity": {}},
        {"entity": {"chunk_uuid": "c", "document_id": "d", "content": 42, "parent_content": "p"}},
    ],
)
async def test_malformed_hit_is_typed_and_not_retried(
    sdk: MagicMock, change: dict[str, object]
) -> None:
    sdk.search.return_value = [
        [
            Hit(
                {
                    "pk": 42,
                    "distance": 0.75,
                    "entity": {
                        "chunk_uuid": "c",
                        "document_id": "d",
                        "content": "c",
                        "parent_content": "p",
                    },
                    **change,
                },
                pk_name="pk",
            )
        ]
    ]
    async with MilvusRepository(MilvusSettings()) as repository:
        with pytest.raises(RetrievalUnavailableError):
            await repository.search_dense(DenseSearch(vector=[0.0] * 1024))
    sdk.search.assert_awaited_once()
    sdk.close.assert_awaited_once()


@pytest.mark.parametrize("response", [[], [[], []], [[{"distance": 1.0, "entity": {}}]], None])
async def test_invalid_search_batch_is_typed(sdk: MagicMock, response: object) -> None:
    sdk.search.return_value = response
    async with MilvusRepository(MilvusSettings()) as repository:
        with pytest.raises(RetrievalUnavailableError):
            await repository.search_dense(DenseSearch(vector=[0.0] * 1024))
    sdk.search.assert_awaited_once()


@pytest.mark.parametrize("response", [None, [], [SimpleNamespace()], [SimpleNamespace(tokens=[1])]])
async def test_invalid_analyzer_batch_is_typed(sdk: MagicMock, response: object) -> None:
    sdk.run_analyzer = AsyncMock(return_value=response)
    async with MilvusRepository(MilvusSettings()) as repository:
        with pytest.raises(RetrievalUnavailableError):
            await repository.analyze(AnalyzerRequest(texts=["SKU-A1023"]))
    sdk.run_analyzer.assert_awaited_once()


async def test_index_order_is_not_part_of_persistence_contract(sdk: MagicMock) -> None:
    async with MilvusRepository(MilvusSettings()) as repository:
        before = await repository.ensure_collection()
        sdk.list_indexes.return_value.reverse()
        after = await repository.ensure_collection()
    assert before == after


@pytest.mark.parametrize("duplicate", ["name", "field"])
async def test_duplicate_index_identity_refuses_loading(sdk: MagicMock, duplicate: str) -> None:
    if duplicate == "name":
        sdk.list_indexes.return_value.append("dense")
    else:
        original = sdk.describe_index.side_effect

        def response(name: str, index: str, **kwargs: object) -> dict[str, object]:
            raw = original(name, index, **kwargs)
            if index == "document_id":
                raw["field_name"] = "doc_type"
            return raw

        sdk.describe_index.side_effect = response
    async with MilvusRepository(MilvusSettings()) as repository:
        with pytest.raises(RetrievalSchemaError):
            await repository.ensure_collection()
    sdk.load_collection.assert_not_called()
