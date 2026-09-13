"""Malformed writes, scoped paging and retry policy without external storage."""

from datetime import date
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from app.core.errors import RetrievalUnavailableError
from app.retrieval.config import MilvusSettings
from app.retrieval.ingestion_store import IngestionStore
from app.schemas.corpus import CorpusMetadata, DocumentType
from app.schemas.ingestion import PreparedChunk, VectorRow


def store(**methods: AsyncMock) -> IngestionStore:
    """Exercise the real bounded adapter around a controlled SDK object."""
    result = object.__new__(IngestionStore)
    result.settings = MilvusSettings()
    result._client = SimpleNamespace(**methods)
    result._validated = True
    return result


@pytest.fixture
def vector() -> VectorRow:
    return VectorRow(
        chunk=PreparedChunk(
            document_id=uuid4(),
            document_version="a" * 64,
            chunking_version="b" * 64,
            chunk_uuid=uuid4(),
            content_sha256="c" * 64,
            ordinal=0,
            heading_path="政策",
            char_len=2,
            content="规则",
            parent_content="完整规则",
        ),
        metadata=CorpusMetadata(
            title="政策",
            doc_type=DocumentType.POLICY,
            effective_from=date(2026, 1, 1),
            effective_to=None,
            supersedes=None,
        ),
        dense=[1.0, *([0.0] * 1023)],
        sparse_learned={1: 1.0},
    )


@pytest.mark.parametrize(
    "receipt",
    [
        {},
        {"ids": [], "insert_count": 1},
        {"ids": [1], "insert_count": 0},
        {"ids": [1, 1], "insert_count": 2},
    ],
)
async def test_incomplete_insert_receipt_rejected(vector: VectorRow, receipt: object) -> None:
    operation = AsyncMock(return_value=receipt)
    with pytest.raises(RetrievalUnavailableError):
        await store(insert=operation).insert([vector])
    operation.assert_awaited_once()


async def test_insert_does_not_retry_uncertain_timeout(vector: VectorRow) -> None:
    operation = AsyncMock(side_effect=TimeoutError())
    with pytest.raises(RetrievalUnavailableError):
        await store(insert=operation).insert([vector])
    operation.assert_awaited_once()


async def test_insert_serializes_business_bounds_and_server_bm25(vector: VectorRow) -> None:
    operation = AsyncMock(return_value={"insert_count": 1, "ids": [100]})
    receipt = await store(insert=operation).insert([vector])
    assert receipt.ids == [100]
    kwargs = operation.call_args.kwargs
    assert kwargs["timeout"] is None
    assert kwargs["retry_times"] == 0
    assert kwargs["data"][0]["effective_to"] == -1
    assert "sparse_bm25" not in kwargs["data"][0]


def physical(vector: VectorRow, pk: int) -> dict[str, object]:
    return {
        "pk": pk,
        "chunk_uuid": str(vector.chunk.chunk_uuid),
        "document_id": str(vector.chunk.document_id),
        "document_version": vector.chunk.document_version,
        "chunking_version": vector.chunk.chunking_version,
        "content_sha256": vector.chunk.content_sha256,
    }


async def test_cleanup_pages_without_order_assumption_and_keeps_current_pk(
    vector: VectorRow,
) -> None:
    query = AsyncMock(
        side_effect=[[physical(vector, 300), physical(vector, 100)], [physical(vector, 200)], []]
    )
    delete = AsyncMock(return_value={"delete_count": 2})
    repository = store(query=query, delete=delete)
    expected_stale = 2
    assert await repository.cleanup(vector.chunk.document_id, [200]) == expected_stale
    assert query.call_args_list[1].kwargs["filter_params"]["seen"] == [-1, 100, 300]
    assert delete.call_args.kwargs["ids"] == [300, 100]


async def test_repeated_page_refuses_cleanup(vector: VectorRow) -> None:
    query = AsyncMock(return_value=[physical(vector, 100)])
    delete = AsyncMock()
    with pytest.raises(RetrievalUnavailableError):
        await store(query=query, delete=delete).cleanup(vector.chunk.document_id, [])
    delete.assert_not_called()


async def test_foreign_document_response_cannot_be_deleted(vector: VectorRow) -> None:
    query = AsyncMock(return_value=[physical(vector, 100)])
    delete = AsyncMock()
    with pytest.raises(RetrievalUnavailableError):
        await store(query=query, delete=delete).cleanup(uuid4(), [])
    delete.assert_not_called()
