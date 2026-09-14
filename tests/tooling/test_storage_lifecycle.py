"""Exact collection ownership, cancellation and teardown error preservation."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from scripts.ci_storage import CleanupState, CollectionReceipt, StackEvidence, StorageIdentity
from scripts.deployment import DeploymentError
from tests import storage_lifecycle
from tests.storage_lifecycle import CollectionOwner, remove_owned


def owner(tmp_path: Path) -> CollectionOwner:
    return CollectionOwner(
        "http://127.0.0.1:19530",
        StackEvidence(identity=StorageIdentity(), project="insightpilot-test-milvus-" + "a" * 12),
        tmp_path,
    )


def client_patch(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.has_collection.side_effect = [True, False]
    monkeypatch.setattr(storage_lifecycle, "AsyncMilvusClient", lambda **kwargs: client)
    return client


@pytest.mark.parametrize("failure", [None, ValueError("body"), asyncio.CancelledError()])
async def test_registered_before_creation_and_cleaned_after_every_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: BaseException | None
) -> None:
    manager = owner(tmp_path)
    client = client_patch(monkeypatch)

    async def body() -> None:
        async with manager.collection("step31", "case") as config:
            saved = StackEvidence.model_validate_json((tmp_path / "lifecycle.json").read_text())
            assert saved.collections[0].name == config.collection
            assert saved.collections[0].state is CleanupState.PENDING
            if failure is not None:
                raise failure

    if failure is None:
        await body()
    else:
        with pytest.raises(type(failure)) as raised:
            await body()
        assert raised.value is failure
    row = manager.evidence.collections[0]
    assert row.state is CleanupState.ABSENT
    assert row.primary_failed is (failure is not None)
    client.drop_collection.assert_awaited_once_with(row.name, timeout=10)
    client.__aexit__.assert_awaited_once()


async def test_partial_setup_without_collection_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = owner(tmp_path)
    client = client_patch(monkeypatch)
    client.has_collection.side_effect = [False, False]
    with pytest.raises(ValueError):
        async with manager.collection("step31", "setup"):
            raise ValueError("initialization failed before create")
    client.drop_collection.assert_not_awaited()
    assert manager.evidence.collections[0].state is CleanupState.ABSENT


@pytest.mark.parametrize("primary", [False, True])
@pytest.mark.parametrize("cleanup", [TimeoutError(), ValueError("transport")])
async def test_cleanup_failure_is_blocking_without_hiding_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, primary: bool, cleanup: Exception
) -> None:
    manager = owner(tmp_path)
    client = client_patch(monkeypatch)
    client.drop_collection.side_effect = cleanup
    original = ValueError("original")
    with pytest.raises(ValueError if primary else DeploymentError) as raised:
        async with manager.collection("step31", "case"):
            if primary:
                raise original
    if primary:
        assert raised.value is original
        assert original.__notes__
    row = manager.evidence.collections[0]
    assert row.state is CleanupState.FAILED
    assert row.cleanup_error == type(cleanup).__name__
    client.__aexit__.assert_awaited_once()


async def test_forged_owner_and_unacknowledged_deletion_fail(tmp_path: Path) -> None:
    manager = owner(tmp_path)
    row = CollectionReceipt(name="step31_other", owner="other")
    client = AsyncMock()
    with pytest.raises(DeploymentError, match="not owned"):
        await remove_owned(client, row, manager.evidence)
    client.has_collection.assert_not_awaited()
    manager.evidence.collections.append(row)
    client.has_collection.return_value = True
    with pytest.raises(DeploymentError, match="remains"):
        await remove_owned(client, row, manager.evidence)
    assert row.state is CleanupState.PENDING


async def test_external_cancellation_completes_owned_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = owner(tmp_path)
    client = client_patch(monkeypatch)
    entered = asyncio.Event()

    async def body() -> None:
        async with manager.collection("step31", "cancelled"):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(body())
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert manager.evidence.collections[0].state is CleanupState.ABSENT
    client.__aexit__.assert_awaited_once()


async def test_pytest_failure_flag_and_per_stack_files_are_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = client_patch(monkeypatch)
    client.has_collection.side_effect = None
    client.has_collection.return_value = False
    first, second = owner(tmp_path / "one"), owner(tmp_path / "two")
    async with first.collection("step31", "first", lambda: True):
        pass
    async with second.collection("step31", "second"):
        pass
    assert first.evidence.collections[0].primary_failed
    assert not second.evidence.collections[0].primary_failed
    assert (tmp_path / "one/lifecycle.json").read_text() != (tmp_path / "two/lifecycle.json").read_text()


async def test_hanging_cleanup_obeys_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = owner(tmp_path)
    client = client_patch(monkeypatch)
    monkeypatch.setattr(storage_lifecycle, "CLEANUP_TIMEOUT_S", 0.001)

    async def hang(*args: object, **kwargs: object) -> None:
        await asyncio.Event().wait()

    client.drop_collection.side_effect = hang
    with pytest.raises(DeploymentError):
        async with manager.collection("step31", "deadline"):
            pass
    assert manager.evidence.collections[0].cleanup_error == "TimeoutError"
    client.__aexit__.assert_awaited_once()
