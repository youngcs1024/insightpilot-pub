"""Bounded exact-owner collection cleanup for isolated test stacks only."""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from pymilvus import AsyncMilvusClient

from app.retrieval.config import MilvusSettings
from scripts.ci_storage import CleanupState, CollectionReceipt, StackEvidence
from scripts.deployment import DeploymentError

CLEANUP_TIMEOUT_S = 30.0


class CollectionClient(Protocol):
    """Small cleanup contract shared by the real SDK and deterministic tests."""

    async def has_collection(self, name: str, *, timeout: float) -> bool: ...
    async def drop_collection(self, name: str, *, timeout: float) -> object: ...


async def remove_owned(
    client: CollectionClient, receipt: CollectionReceipt, evidence: StackEvidence
) -> None:
    """Never enumerate/drop other collections, even if their names look similar."""
    if not any(row is receipt for row in evidence.collections):
        raise DeploymentError("Collection is not owned by this test lifecycle")
    async with asyncio.timeout(CLEANUP_TIMEOUT_S):
        if await client.has_collection(receipt.name, timeout=10):
            await client.drop_collection(receipt.name, timeout=10)
        if await client.has_collection(receipt.name, timeout=10):
            raise DeploymentError("Owned test collection remains after cleanup")
    receipt.state = CleanupState.ABSENT


class CollectionOwner:
    """Register ownership before creation and preserve primary failures during cleanup."""

    def __init__(self, uri: str, evidence: StackEvidence, directory: Path) -> None:
        self.uri = uri
        self.evidence = evidence
        self.directory = directory

    def save(self) -> None:
        """One owner writes one stack's receipt; no cross-stack file overwrites."""
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "lifecycle.json").write_text(self.evidence.model_dump_json(indent=2))

    @asynccontextmanager
    async def collection(
        self, prefix: str, owner: str, failed: Callable[[], bool] = lambda: False
    ) -> AsyncIterator[MilvusSettings]:
        """Close a separate cleanup client even if the test closes or corrupts its own."""
        receipt = CollectionReceipt(name=prefix + "_" + uuid4().hex, owner=owner)
        self.evidence.collections.append(receipt)
        self.save()
        primary: BaseException | None = None
        try:
            yield MilvusSettings(uri=self.uri, collection=receipt.name, timeout_s=30)
        except BaseException as exc:
            primary = exc
            raise
        finally:
            receipt.primary_failed = primary is not None or failed()
            try:
                async with (
                    asyncio.timeout(35),
                    AsyncMilvusClient(uri=self.uri, timeout=10) as client,
                ):
                    await remove_owned(client, receipt, self.evidence)
            except BaseException as exc:
                receipt.state = CleanupState.FAILED
                receipt.cleanup_error = type(exc).__name__
                if primary is not None:
                    primary.add_note(
                        "Additional owned-collection cleanup failure; see lifecycle evidence."
                    )
                else:
                    raise DeploymentError("Owned test collection cleanup failed") from exc
            finally:
                self.save()
