"""Typed storage lifecycle evidence, independent of pytest and Docker clients."""

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


SERVICES = frozenset({"milvus", "etcd", "minio"})


class StorageIdentity(BaseSettings):
    """Explicit workflow identity; local diagnostic runs use the zero identity."""

    model_config = SettingsConfigDict(env_prefix="CI_STORAGE_", extra="forbid")
    tested_sha: str = Field(default="0" * 40, pattern=r"^[0-9a-f]{40}$")
    run_id: str = Field(default="0", pattern=r"^[0-9]+$")
    run_attempt: str = Field(default="0", pattern=r"^[0-9]+$")


class CleanupState(StrEnum):
    """An allocated collection is incomplete until confirmed absent."""

    PENDING = "pending"
    ABSENT = "absent"
    FAILED = "failed"


class CollectionReceipt(BaseModel):
    """Exact ownership and original failure identity, without captured test data."""

    name: str = Field(pattern=r"^step[0-9]+_[a-z0-9_]+$")
    owner: str = Field(min_length=1, max_length=1000)
    state: CleanupState = CleanupState.PENDING
    primary_failed: bool = False
    cleanup_error: str | None = None


class ContainerSample(BaseModel):
    """Docker counters only; no environment, healthcheck commands or credentials."""

    service: Literal["milvus", "etcd", "minio"]
    memory_bytes: int = Field(ge=0)
    limit_bytes: int = Field(gt=0)
    oom_killed: bool
    running: bool
    restarts: int = Field(ge=0)


class StorageSample(BaseModel):
    """Snapshots at startup, failures and before fixture teardown."""

    phase: Literal["startup", "failure", "final"]
    containers: list[ContainerSample]


class StackEvidence(BaseModel):
    """Persist before startup so crashes cannot disappear from the final assessment."""

    identity: StorageIdentity
    project: str = Field(pattern=r"^insightpilot-test-milvus-[a-f0-9]{12}$")
    completed: bool = False
    collections: list[CollectionReceipt] = Field(default_factory=list)
    samples: list[StorageSample] = Field(default_factory=list)

    @property
    def accepted(self) -> bool:
        """Require both lifecycle completion and healthy, complete resource snapshots."""
        return (
            self.completed
            and bool(self.collections)
            and len({row.name for row in self.collections}) == len(self.collections)
            and all(row.state is CleanupState.ABSENT and row.cleanup_error is None for row in self.collections)
            and {row.phase for row in self.samples} >= {"startup", "final"}
            and all(
                len(sample.containers) == len(SERVICES)
                and {row.service for row in sample.containers} == SERVICES
                and all(row.running and not row.oom_killed for row in sample.containers)
                for sample in self.samples
            )
        )


class StorageAssessment(BaseModel):
    """Invalid files and cleanup failures cannot become a successful empty assessment."""

    stacks: list[StackEvidence] = Field(default_factory=list)
    evidence_valid: bool = False

    @property
    def accepted(self) -> bool:
        """The raw storage job must still independently pass."""
        return (
            self.evidence_valid
            and bool(self.stacks)
            and len({s.project for s in self.stacks}) == len(self.stacks)
            and all(s.accepted for s in self.stacks)
        )


def assess_storage(root: Path, identity: StorageIdentity) -> StorageAssessment:
    """Read all per-stack receipts and reject stale, missing or duplicate identities."""
    result = StorageAssessment(evidence_valid=True)
    for path in sorted(root.rglob("lifecycle.json")):
        try:
            stack = StackEvidence.model_validate_json(path.read_text())
        except (OSError, ValueError):
            result.evidence_valid = False
            continue
        if stack.identity != identity:
            result.evidence_valid = False
        result.stacks.append(stack)
    if not result.stacks or len({s.project for s in result.stacks}) != len(result.stacks):
        result.evidence_valid = False
    return result
