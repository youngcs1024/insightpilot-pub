"""Per-process probe settings; never import the API settings singleton."""

from pathlib import Path

from pydantic import Field

from app.core.settings_base import ConfigModel, ProcessSettings, Secret
from model_runtime.config import ModelServerSettings


class ProbeLimits(ConfigModel):
    """A user-authorized one-shot exception never changes production model settings."""

    gpu_headroom_bytes: int = Field(default=4 * 1024**3, ge=0)


class ProbeRuntimeSettings(ProcessSettings):
    """GPU probe-only runtime configuration, independent of API and deployment secrets."""

    process_name = "capacity-model"
    model_server: ModelServerSettings
    probe: ProbeLimits = Field(default_factory=ProbeLimits)


class CapacityFields(ConfigModel):
    """Bounded observation window and explicit isolated project identity."""

    project: str = Field(pattern=r"^insightpilot-(?:model-)?test-step08-[a-z0-9-]+$")
    output: Path
    stop_file: Path | None = None
    duration_s: float = Field(default=300, ge=1, le=1800)
    interval_s: float = Field(default=1, ge=0.2, le=5)
    gpu_id: str | None = Field(default=None, pattern=r"^GPU-[0-9a-fA-F-]{36}$")
    gpu_headroom_bytes: int = Field(default=4 * 1024**3, ge=0)
    reader_image: str = Field(default="insightpilot-step08-client:local", min_length=1)


class CapacitySettings(ProcessSettings):
    """Only the local or remote observer configuration, no model credentials."""

    process_name = "capacity"
    capacity: CapacityFields


class WorkloadFields(ConfigModel):
    """One disposable client, with no credentials for the existing databases."""

    output: Path
    model_url: str = "http://model-tunnel:8100"
    auth_token: Secret = Field(repr=False)
    postgres_host: str = "postgres"
    postgres_password: Secret = Field(repr=False)
    milvus_uri: str = "http://milvus:19530"
    corpus_size: int = Field(default=1024, ge=32, le=16384)
    turns: int = Field(default=50, ge=20, le=200)


class WorkloadSettings(ProcessSettings):
    """Inputs for the isolated workload job."""

    process_name = "capacity-workload"
    workload: WorkloadFields
