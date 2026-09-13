"""Pure comparison, bounded storage adapters and safe operator CLI contracts."""

# ruff: noqa: PLR2004 -- fixed synthetic primary keys and pagination counts.

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.core.errors import (
    IngestionAlreadyRunning,
    IngestionConfigurationError,
    IngestionRegistryError,
    RetrievalUnavailableError,
)
from app.core.settings_base import ProcessSettings
from app.retrieval.config import MilvusSettings
from app.retrieval.consistency_store import ConsistencyStore
from app.schemas.consistency import (
    ConsistencyReport,
    Drift,
    DriftKind,
    DriftReason,
    IndexSnapshot,
    RegistrySnapshot,
    StoredChunk,
)
from app.schemas.ingestion import RegisteredChunk, canonical, digest
from app.services.consistency_check import assess, validate_registry
from app.services.consistency_plan import restore_splitter, stage_repair
from app.services.ingestion_config import IngestionSettings
from scripts import check_consistency as cli


def row(pk: int = 100) -> dict[str, object]:
    return {
        "pk": pk,
        "chunk_uuid": str(uuid4()),
        "document_id": str(uuid4()),
        "document_version": "a" * 64,
        "chunking_version": "b" * 64,
        "content_sha256": digest("private corpus text"),
        "content": "private corpus text",
    }


def store(query: AsyncMock, *, exists: object = True) -> ConsistencyStore:
    repository = object.__new__(ConsistencyStore)
    repository.settings = MilvusSettings()
    repository._client = SimpleNamespace(
        query=query,
        has_collection=AsyncMock(return_value=exists),
        delete=AsyncMock(return_value={"delete_count": 1}),
    )
    repository.validate_collection = AsyncMock()
    return repository


async def test_scan_pages_all_documents_without_order_assumption() -> None:
    query = AsyncMock(side_effect=[[row(300), row(100)], [row(200)], []])
    result = await store(query).scan()
    assert result.exists
    assert {item.milvus_pk for item in result.rows} == {100, 200, 300}
    assert "private corpus text" not in result.model_dump_json()
    assert query.call_args_list[1].kwargs["filter_params"]["seen"] == [-1, 100, 300]
    assert all(call.kwargs["consistency_level"] == "Strong" for call in query.call_args_list)
    assert "document_id" not in query.call_args.kwargs["filter"]


async def test_scan_absent_collection_does_not_create_or_query() -> None:
    query = AsyncMock()
    repository = store(query, exists=False)
    result = await repository.scan()
    assert not result.exists
    assert result.rows == []
    query.assert_not_awaited()
    repository.validate_collection.assert_not_awaited()


@pytest.mark.parametrize("raw", [{}, [{"pk": 1}], [row(), row(100)]])
async def test_malformed_or_duplicate_page_fails_closed(raw: object) -> None:
    repository = store(AsyncMock(return_value=raw))
    with pytest.raises(RetrievalUnavailableError):
        await repository.scan()
    repository._client.delete.assert_not_awaited()


async def test_repeated_page_fails_instead_of_looping() -> None:
    query = AsyncMock(return_value=[row()])
    with pytest.raises(RetrievalUnavailableError):
        await store(query).scan()
    assert query.await_count == 2


async def test_scan_sdk_failure_has_no_silent_empty_fallback() -> None:
    query = AsyncMock(side_effect=TimeoutError())
    with pytest.raises(RetrievalUnavailableError):
        await store(query).scan()
    query.assert_awaited_once()


async def test_invalid_collection_presence_response_fails() -> None:
    with pytest.raises(RetrievalUnavailableError):
        await store(AsyncMock(), exists="false").scan()


async def test_exact_key_delete_batches_and_never_retries_uncertain_outcome() -> None:
    repository = store(AsyncMock())
    keys = list(range(1001))
    assert await repository.delete_observed(keys) == len(keys)
    calls = repository._client.delete.call_args_list
    assert calls[0].kwargs["ids"] == keys[:1000]
    assert calls[1].kwargs["ids"] == keys[1000:]
    repository._client.delete = AsyncMock(side_effect=TimeoutError())
    with pytest.raises(RetrievalUnavailableError):
        await repository.delete_observed([1])
    repository._client.delete.assert_awaited_once()


@pytest.mark.parametrize("raw", ["null", "[]", "{}", "bad JSON", canonical({"child_size": 0})])
def test_invalid_or_unknown_splitter_is_typed(raw: str) -> None:
    with pytest.raises(IngestionConfigurationError):
        restore_splitter(raw, IngestionSettings())


def test_splitter_restore_preserves_semantics_and_current_operational_limits() -> None:
    frozen = IngestionSettings(child_size=200, child_overlap=20)
    limits = IngestionSettings(timeout_s=99, max_source_bytes=500)
    restored = restore_splitter(frozen.splitter_config(), limits)
    assert restored.chunking_version() == frozen.chunking_version()
    assert restored.timeout_s == 99
    assert restored.max_source_bytes == 500


def test_empty_repair_selection_does_not_read_corpus() -> None:
    result = stage_repair(
        Path("/missing-root"),
        RegistrySnapshot(documents=[], chunks=[], manifest=None),
        set(),
        IngestionSettings(),
    )
    assert not result.prepared
    assert not result.blocked


def test_missing_inventory_reports_every_selected_document() -> None:
    identifiers = {uuid4(), uuid4()}
    result = stage_repair(
        Path("/missing-root"),
        RegistrySnapshot(documents=[], chunks=[], manifest=None),
        identifiers,
        IngestionSettings(),
    )
    assert {item.document_id for item in result.blocked} == identifiers


def test_registry_without_manifest_is_rejected() -> None:
    identifier = uuid4()
    chunk = RegisteredChunk(
        chunk_uuid=uuid4(),
        document_id=identifier,
        document_version="a" * 64,
        chunking_version="b" * 64,
        content_sha256="c" * 64,
        ordinal=0,
        heading_path="",
        char_len=1,
    )
    with pytest.raises(IngestionRegistryError):
        validate_registry(
            RegistrySnapshot(documents=[], chunks=[chunk], manifest=None), "kb_chunks"
        )


def test_unknown_document_is_found_in_global_index() -> None:
    physical = StoredChunk(
        chunk_uuid=uuid4(),
        document_id=uuid4(),
        document_version="a" * 64,
        chunking_version="b" * 64,
        content_sha256="c" * 64,
        actual_sha256="c" * 64,
        milvus_pk=100,
    )
    report = assess(
        RegistrySnapshot(documents=[], chunks=[], manifest=None),
        IndexSnapshot(exists=True, rows=[physical]),
    )
    assert report.drift[0].kind is DriftKind.ORPHAN
    assert report.rebuild == set()


def test_report_counts_are_complete_and_pending_work_never_passes() -> None:
    report = ConsistencyReport()
    assert report.successful
    assert len(report.counts) == 4
    report.remaining.append(
        Drift(
            kind=DriftKind.MISSING,
            reason=DriftReason.NULL_PK,
            document_id=uuid4(),
        )
    )
    assert report.counts[DriftKind.MISSING] == 1
    assert not report.successful
    report.remaining.clear()
    report.cleanup_pending.append(uuid4())
    assert not report.successful


def test_settings_need_no_model_credentials_and_reject_foreign_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in os.environ:
        if key.startswith("IP_"):
            monkeypatch.delenv(key)
    monkeypatch.setattr(ProcessSettings, "project_root", tmp_path)
    monkeypatch.setenv("IP_DATABASE__APP_PASSWORD", "synthetic-only")
    settings = cli.ConsistencyProcessSettings.load()
    assert settings.model_runtime is None
    monkeypatch.setenv("IP_MIGRATION__PASSWORD", "must-not-leak")
    with pytest.raises(ValidationError) as error:
        cli.ConsistencyProcessSettings.load()
    assert "must-not-leak" not in str(error.value)


def test_example_is_complete_for_check_without_model_token() -> None:
    settings = cli.ConsistencyProcessSettings(_env_file=Path(".env.consistency.example"))
    assert settings.model_runtime is None
    assert settings.database.app_user == "app_rw"


@pytest.mark.parametrize("fix", [False, True])
def test_cli_passes_root_only_when_fix_is_explicit(
    monkeypatch: pytest.MonkeyPatch, fix: bool
) -> None:
    settings = object()
    run = AsyncMock(return_value=0)
    monkeypatch.setattr(cli.ConsistencyProcessSettings, "load", lambda: settings)
    monkeypatch.setattr(cli, "run", run)
    assert cli.main(["--corpus", "/configured", *(["--fix"] if fix else [])]) == 0
    run.assert_awaited_once_with(settings, Path("/configured") if fix else None)


@pytest.mark.parametrize(
    "failure", [IngestionAlreadyRunning("private prose"), TimeoutError("private prose")]
)
def test_cli_failures_are_nonzero_and_safe(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failure: Exception
) -> None:
    monkeypatch.setattr(cli.ConsistencyProcessSettings, "load", object)
    monkeypatch.setattr(cli, "run", AsyncMock(side_effect=failure))
    assert cli.main([]) == 1
    output = capsys.readouterr().err
    assert "private prose" not in output
    assert json.loads(output)["successful"] is False


async def test_check_closes_resources_without_constructing_model(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    database = SimpleNamespace(start=MagicMock(), aclose=AsyncMock())
    storage = MagicMock()
    storage.__aenter__ = AsyncMock(return_value=storage)
    storage.__aexit__ = AsyncMock()
    checker = SimpleNamespace(check=AsyncMock(return_value=ConsistencyReport()))
    monkeypatch.setattr(cli, "Database", lambda _: database)
    monkeypatch.setattr(cli, "ConsistencyStore", lambda _: storage)
    monkeypatch.setattr(cli, "ConsistencyService", lambda *args: checker)
    model = MagicMock(side_effect=AssertionError("Check must not create a model client"))
    monkeypatch.setattr(cli, "ModelRuntimeClient", model)
    settings = cli.ConsistencyProcessSettings(
        _env_file=None,
        database={"app_password": "synthetic-only"},
        model_runtime={"auth_token": "synthetic-only"},
    )
    assert await cli.run(settings, None) == 0
    model.assert_not_called()
    database.aclose.assert_awaited_once()
    storage.__aexit__.assert_awaited_once()
    assert "0 drift" in capsys.readouterr().out


def test_cli_invalid_configuration_is_a_safe_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def invalid() -> None:
        raise ValidationError.from_exception_data("Settings", [])

    monkeypatch.setattr(cli.ConsistencyProcessSettings, "load", invalid)
    assert cli.main([]) == 1
    assert json.loads(capsys.readouterr().err)["code"] == "CONSISTENCY_CONFIGURATION_INVALID"
