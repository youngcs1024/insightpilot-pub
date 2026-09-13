"""Registry metadata rejects malformed source records without exposing their payloads."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.core.errors import IngestionRegistryError
from app.repositories.chunk import ChunkRepository
from app.schemas.corpus import CorpusMetadata
from tests.knowledge_support import provenance
from tests.retrieval_support import candidate


async def test_empty_provenance_never_queries_database() -> None:
    session = SimpleNamespace(execute=AsyncMock())
    assert await ChunkRepository(session).provenance([]) == []
    session.execute.assert_not_called()


@pytest.mark.parametrize("damage", ["metadata", "page", "path"])
async def test_bad_registered_provenance_is_typed_error(damage: str) -> None:
    value = candidate()
    registered = provenance(value)
    chunk = SimpleNamespace(
        id=value.chunk_uuid, **value.model_dump(exclude={"schema_version"}), page=None
    )
    metadata = CorpusMetadata.model_validate(
        {
            "title": "规则",
            "doc_type": "policy",
            "effective_from": "2026-01-01",
            "effective_to": None,
            "supersedes": None,
        }
    )
    document = SimpleNamespace(
        business_metadata=metadata.model_dump_json(), source_path=registered.source_path
    )
    if damage == "metadata":
        document.business_metadata = "invalid CANARY_METADATA"
    elif damage == "page":
        chunk.page = 0
    else:
        document.source_path = "../secret.md"
    session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(all=lambda: [(chunk, document)]))
    )
    with pytest.raises(IngestionRegistryError) as caught:
        await ChunkRepository(session).provenance([value])
    assert "CANARY_METADATA" not in str(caught.value)
