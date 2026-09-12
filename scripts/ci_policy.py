"""Pure change classification and final CI result contracts."""

from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.errors import ValidationError
from scripts.ci_partitions import PARTITIONS

Image = Literal["api", "mcp", "model-runtime", "model-tunnel"]
ALL_IMAGES: tuple[Image, ...] = ("api", "mcp", "model-runtime", "model-tunnel")


class Reason(StrEnum):
    """Finite causes for running or omitting validation."""

    CHANGES = "changed_files"
    MANUAL = "manual_full_run"
    BASELINE = "no_trusted_baseline"
    DISCOVERY = "discovery_failed"


class Plan(BaseModel):
    """Versioned, validated output consumed by workflow conditions and the final gate."""

    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal[1] = 1
    checks: bool
    images: list[Image]
    reason: Reason
    baseline: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    tested_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    changed_paths: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def coherent_selection(self) -> "Plan":
        """Reject duplicate images or builds without quality and regression checks."""
        if len(self.images) != len(set(self.images)) or (self.images and not self.checks):
            raise ValidationError("Invalid CI task selection")
        if self.reason == Reason.CHANGES and self.baseline is None:
            raise ValidationError("Change selection requires a trusted baseline")
        if self.reason != Reason.CHANGES and (not self.checks or self.images != list(ALL_IMAGES)):
            raise ValidationError("Fallback and manual plans must run everything")
        return self


def full_plan(tested_sha: str, reason: Reason) -> Plan:
    """Fail safe by selecting every existing check and image."""
    return Plan(checks=True, images=list(ALL_IMAGES), reason=reason, tested_sha=tested_sha)


def ordinary_document(name: str) -> bool:
    """Allow only known prose locations, never all Markdown in the repository."""
    if name in {"README.md", "AGENTS.md", "PROJECT_OVERVIEW.md"}:
        return True
    path = PurePosixPath(name)
    return path.suffix == ".md" and (
        path.parent in {PurePosixPath("docs"), PurePosixPath("docs/roadmap")}
        or PurePosixPath("docs/evidence") in path.parents
    )


def path_images(name: str) -> set[Image]:  # noqa: PLR0911 -- explicit disjoint image owners.
    """Map known functional inputs; unknown paths conservatively affect both images."""
    if name.startswith("tests/"):
        return set()
    if name.startswith("model_runtime/"):
        return {"api", "model-runtime"}
    if name.startswith("model_tunnel/"):
        return {"api", "model-tunnel"}
    if name.startswith("mcp_server/"):
        return {"mcp"}
    if name == "app/__init__.py" or name.startswith(("app/core/", "app/schemas/")):
        return set(ALL_IMAGES)
    if name.startswith(("app/", "alembic/", "data/")) or name in {
        "alembic.ini",
        "pyproject.toml",
        "uv.lock",
        "app/resources/provider_capabilities.json",
    }:
        return {"api"}
    # scripts are copied into the API image, but also control builds and CI.
    # Docker/Compose, workflow, tooling, future model files and unknown inputs
    # deliberately default to full validation until explicitly classified.
    return set(ALL_IMAGES)


def classify(paths: list[str], *, baseline: str, tested_sha: str) -> Plan:
    """Union old/new paths from a complete Git diff into an execution plan."""
    changed = sorted(set(paths))
    functional = [name for name in changed if not ordinary_document(name)]
    images: set[Image] = set()
    for name in functional:
        images.update(path_images(name))
    return Plan(
        checks=bool(functional),
        images=[name for name in ALL_IMAGES if name in images],
        reason=Reason.CHANGES,
        baseline=baseline,
        tested_sha=tested_sha,
        changed_paths=changed,
    )


class Result(StrEnum):
    """GitHub needs-result values; unknown values must fail parsing."""

    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"


class JobOutputs(BaseModel):
    """Read the raw collection prerequisite and actually uploaded coverage ID."""

    collection_outcome: Result | Literal[""] = ""
    coverage_artifact_id: str = ""


class JobResult(BaseModel):
    """Read authoritative job results and prerequisite outputs from GitHub needs."""

    result: Result
    outputs: JobOutputs = Field(default_factory=JobOutputs)


class Results(BaseModel):
    """All direct dependencies of ci-result; no missing job is acceptable."""

    changes: JobResult
    quality: JobResult
    unit: JobResult
    integration: JobResult
    storage: JobResult
    coverage: JobResult
    build: JobResult


def failures(plan: Plan, results: Results) -> list[str]:
    """Reject unexpected skips, failures and cancellations, even on omitted jobs."""
    expected: dict[str, bool] = {
        "changes": True,
        "quality": plan.checks,
        "coverage": plan.checks,
        "build": bool(plan.images),
    }
    expected.update(dict.fromkeys(PARTITIONS, plan.checks))
    failed = [
        name
        for name, required in expected.items()
        if getattr(results, name).result != (Result.SUCCESS if required else Result.SKIPPED)
    ]
    if plan.checks and results.quality.outputs.collection_outcome != Result.SUCCESS:
        failed.append("quality/collection")
    return failed
